"""Deterministic post-hoc evaluation of Experiment 0 training checkpoints."""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

import torch
from torch.utils.data import DataLoader

from exp0.challenge_set import ChallengeSpec, generate_challenge_set
from exp0.checkpointing import load_training_checkpoint
from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.construction_strata import (
    EvaluationRecord,
    build_records,
    corruption_stratum,
    diagnose_packed,
    primary_stratum,
    summarize_strata,
)
from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
from exp0.generation import generate_protocol_packed_instances
from exp0.train import create_model, evaluate_accuracy

DIAGNOSTIC_ARTIFACT_VERSION = 1
FINGERPRINT_VERSION = 1

ConfigT = TypeVar("ConfigT", ModelConfig, Task3SumConfig, TrainConfig)


def _config_from_mapping(
    config_type: type[ConfigT],
    values: Mapping[str, Any],
    label: str,
) -> ConfigT:
    allowed = {item.name for item in fields(config_type)}
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(
            f"Checkpoint {label} contains unsupported fields: {unexpected}"
        )
    return config_type(**dict(values))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _packed_instance_payload(packed, index: int) -> dict[str, Any]:
    instance = packed.instance_at(index)
    return {
        "tuples": [list(row) for row in instance.tuples],
        "true_label": bool(instance.has_3sum),
        "planted_matching_indices": (
            list(instance.matching_indices)
            if instance.matching_indices is not None
            else None
        ),
        "construction_arm": (
            None
            if instance.construction_arm is None
            else "positive"
            if instance.construction_arm
            else "corrupted"
        ),
        "corruption_count": instance.corruption_count,
    }


def fingerprint_packed_instances(
    packed,
    *,
    namespace: str,
    generation_config: Mapping[str, Any],
) -> str:
    """Hash exact instance contents and generation provenance deterministically."""
    digest = hashlib.sha256()
    digest.update(
        _json_bytes(
            {
                "fingerprint_version": FINGERPRINT_VERSION,
                "namespace": namespace,
                "generation_config": dict(generation_config),
                "population_size": len(packed),
            }
        )
    )
    digest.update(b"\n")
    for index in range(len(packed)):
        digest.update(_json_bytes(_packed_instance_payload(packed, index)))
        digest.update(b"\n")
    return digest.hexdigest()


def structural_features(
    tuples: Sequence[Sequence[int]],
    *,
    mod: int = 10,
) -> dict[str, Any]:
    """Return prediction-independent Match-3 hardness features.

    Candidate triples follow lexicographic ``itertools.combinations`` order.
    ``first_valid_witness_position`` is zero based in that order.
    """
    if not tuples:
        return {
            "total_candidate_triples": 0,
            "candidate_coordinate_match_counts": [],
            "num_two_of_three_near_misses": None,
            "max_matched_coordinate_count_among_non_solutions": None,
            "first_valid_witness_position": None,
        }
    dimension = len(tuples[0])
    if dimension <= 0 or any(len(row) != dimension for row in tuples):
        raise ValueError("All tuples must have the same positive dimension.")

    counts = [0] * (dimension + 1)
    first_witness_position = None
    max_non_solution = None
    for position, (i, j, k) in enumerate(itertools.combinations(range(len(tuples)), 3)):
        matched = sum(
            (tuples[i][coordinate] + tuples[j][coordinate] + tuples[k][coordinate])
            % mod
            == 0
            for coordinate in range(dimension)
        )
        counts[matched] += 1
        if matched == dimension:
            if first_witness_position is None:
                first_witness_position = position
        elif max_non_solution is None or matched > max_non_solution:
            max_non_solution = matched

    return {
        "total_candidate_triples": math.comb(len(tuples), 3),
        "candidate_coordinate_match_counts": counts,
        "num_two_of_three_near_misses": counts[2] if dimension == 3 else None,
        "max_matched_coordinate_count_among_non_solutions": max_non_solution,
        "first_valid_witness_position": first_witness_position,
    }


def _predicted_labels(predicted_ids: Sequence[int], vocab) -> list[bool | None]:
    true_id = vocab.token2id["True"]
    false_id = vocab.token2id["False"]
    return [
        True if token == true_id else False if token == false_id else None
        for token in predicted_ids
    ]


def _example_record(
    packed,
    record: EvaluationRecord,
    *,
    population_id: str,
    mod: int,
) -> dict[str, Any]:
    index = record.diagnostics.index
    content = _packed_instance_payload(packed, index)
    content_hash = hashlib.sha256(_json_bytes(content)).hexdigest()
    diagnostics = asdict(record.diagnostics)
    result = {
        "index": index,
        "example_id": f"{population_id[:16]}:{index:06d}:{content_hash[:12]}",
        **content,
        "correct": record.correct,
        "predicted_label": record.predicted_label,
        "true_logit": record.true_logit,
        "false_logit": record.false_logit,
        "prediction_margin": record.prediction_margin,
        "num_valid_triples": diagnostics["num_valid_triples"],
        "first_witness": diagnostics["first_witness"],
        "multiple_witnesses": diagnostics["multiple_witnesses"],
        "primary_stratum": primary_stratum(record.diagnostics),
        "corruption_stratum": corruption_stratum(record.diagnostics),
    }
    result.update(structural_features(content["tuples"], mod=mod))
    if result["candidate_coordinate_match_counts"]:
        expected = result["candidate_coordinate_match_counts"][-1]
        if expected != result["num_valid_triples"]:
            raise AssertionError(
                "Structural feature enumeration disagrees with diagnostics."
            )
    return result


def _evaluate_population(
    model: torch.nn.Module,
    packed,
    *,
    population_id: str,
    task_cfg: Task3SumConfig,
    vocab,
    device: torch.device,
    precision: str,
    batch_size: int,
    dataset_seed: int,
) -> dict[str, Any]:
    dataset = Task3SumDataset(
        packed,
        format_type="filler",
        num_filler=task_cfg.num_filler,
        vocab=vocab,
        seed=dataset_seed,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )
    detail_sink: dict[str, list] = {}
    accuracy, prediction_counts = evaluate_accuracy(
        model,
        loader,
        device,
        vocab.token2id["ANS"],
        vocab.token2id["True"],
        vocab.token2id["False"],
        precision=precision,
        return_prediction_counts=True,
        detail_sink=detail_sink,
    )
    records = build_records(
        diagnose_packed(packed, mod=task_cfg.mod),
        _predicted_labels(detail_sink["predicted_ids"], vocab),
        detail_sink["true_logits"],
        detail_sink["false_logits"],
    )
    examples = [
        _example_record(
            packed,
            record,
            population_id=population_id,
            mod=task_cfg.mod,
        )
        for record in records
    ]
    return {
        "population_size": len(examples),
        "accuracy": accuracy,
        "error_count": sum(not item["correct"] for item in examples),
        "prediction_counts": prediction_counts,
        "stratified": summarize_strata(records),
        "records": examples,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_completed_checkpoint(
    signature: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> None:
    completed_epochs = int(progress.get("epoch", 0))
    requested_epochs = int(signature.get("epochs", 0))
    if progress.get("partial_epoch") is not None or completed_epochs < requested_epochs:
        raise ValueError(
            "Checkpoint is not a completed Experiment 0 run: "
            f"completed epoch {completed_epochs}/{requested_epochs}."
        )


def _validate_run_report_config(
    signature: Mapping[str, Any],
    training_seed: int,
    expected: Mapping[str, Any],
) -> None:
    """Verify optional run-report metadata before performing model evaluation."""
    signature_task = dict(signature["task"])
    signature_task.pop("seed", None)
    signature_training = dict(signature["training"])
    signature_training.pop("seed", None)
    comparisons = (
        ("model", dict(signature["model"]), expected.get("model")),
        ("task_config", signature_task, expected.get("task_config")),
        (
            "training_protocol",
            signature_training,
            expected.get("training_protocol"),
        ),
    )
    for label, checkpoint_value, report_value in comparisons:
        if report_value is not None and checkpoint_value != report_value:
            raise ValueError(
                f"Checkpoint {label} does not match the supplied run report."
            )
    report_seeds = expected.get("evaluation", {}).get("seeds_run")
    if report_seeds is not None and training_seed not in report_seeds:
        raise ValueError(
            f"Checkpoint seed {training_seed} is absent from run report seeds "
            f"{report_seeds}."
        )


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    device: str,
    eval_seed: int,
    val_samples: int,
    challenge_per_class: int = 0,
    challenge_seed: int | None = None,
    batch_size: int | None = None,
    precision: str | None = None,
    evaluation_provenance: Mapping[str, Any] | None = None,
    expected_run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a completed Experiment 0 checkpoint without training state restore."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    payload = load_training_checkpoint(checkpoint_path)
    signature = payload.get("signature")
    progress = payload.get("progress")
    if not isinstance(signature, Mapping) or not isinstance(progress, Mapping):
        raise ValueError("Checkpoint lacks Experiment 0 signature or progress data.")
    _validate_completed_checkpoint(signature, progress)

    model_cfg = _config_from_mapping(ModelConfig, signature["model"], "model")
    task_cfg = _config_from_mapping(Task3SumConfig, signature["task"], "task")
    train_cfg = _config_from_mapping(
        TrainConfig,
        signature["training"],
        "training protocol",
    )
    training_seed = int(train_cfg.seed)
    if expected_run_config is not None:
        _validate_run_report_config(signature, training_seed, expected_run_config)
    checkpoint_model_config = asdict(model_cfg)
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable.")

    evaluation_precision = precision
    precision_source = "cli"
    if evaluation_precision is None:
        if requested_device.type == "cuda":
            evaluation_precision = train_cfg.precision
            precision_source = "checkpoint_training_protocol"
        else:
            evaluation_precision = "fp32"
            precision_source = "cpu_fp32_fallback"
    if evaluation_precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be one of: fp32, bf16, fp16")
    if requested_device.type == "cpu" and evaluation_precision != "fp32":
        raise ValueError("CPU checkpoint evaluation currently requires fp32 precision.")

    model_cfg = replace(model_cfg, device=str(requested_device))
    vocab = build_default_vocab(
        length=task_cfg.length,
        dimension=task_cfg.dimension,
        mod=task_cfg.mod,
    )
    if model_cfg.vocab_size != len(vocab):
        raise ValueError(
            "Checkpoint vocabulary size does not match reconstructed task vocabulary: "
            f"checkpoint={model_cfg.vocab_size}, reconstructed={len(vocab)}."
        )
    compact_reduced_features = (
        task_cfg.vocab_reduction
        and signature.get("realized_format_counts", {}).get("serial_cot", 0) == 0
    )
    model = create_model(
        model_cfg,
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
        compact_reduced_features=compact_reduced_features,
    )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint lacks a model_state_dict mapping.")
    model.load_state_dict(state_dict, strict=True)
    checkpoint_metadata = {
        "checkpoint_version": payload["checkpoint_version"],
        "run_id": signature.get("run_id"),
        "training_seed": training_seed,
        "completed_epochs": int(progress["epoch"]),
        "optimizer_steps": int(progress["optimizer_steps"]),
        "initialization": payload.get("initialization"),
    }
    del state_dict, payload
    gc.collect()
    model = model.to(requested_device)

    canonical_config = {
        "eval_seed": int(eval_seed),
        "val_samples": int(val_samples),
        "length": task_cfg.length,
        "dimension": task_cfg.dimension,
        "mod": task_cfg.mod,
        "true_rate": task_cfg.true_rate,
        "generator_mode": task_cfg.generator_mode,
        "corruption_rate": task_cfg.corruption_rate,
    }
    canonical = generate_protocol_packed_instances(
        num_samples=val_samples,
        length=task_cfg.length,
        dimension=task_cfg.dimension,
        mod=task_cfg.mod,
        true_rate=task_cfg.true_rate,
        rng=random.Random(eval_seed),
        generator_mode=task_cfg.generator_mode,
        corruption_rate=task_cfg.corruption_rate,
        collect_provenance=True,
    )
    canonical_id = fingerprint_packed_instances(
        canonical,
        namespace="canonical_validation",
        generation_config=canonical_config,
    )
    resolved_batch_size = int(batch_size or train_cfg.batch_size)
    canonical_result = _evaluate_population(
        model,
        canonical,
        population_id=canonical_id,
        task_cfg=task_cfg,
        vocab=vocab,
        device=requested_device,
        precision=evaluation_precision,
        batch_size=resolved_batch_size,
        dataset_seed=eval_seed,
    )
    canonical_result.update(
        {
            "canonical_validation_id": canonical_id,
            "generation_config": canonical_config,
        }
    )

    artifact: dict[str, Any] = {
        "artifact_type": "exp0_checkpoint_diagnostics",
        "artifact_version": DIAGNOSTIC_ARTIFACT_VERSION,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _file_sha256(checkpoint_path),
            **checkpoint_metadata,
        },
        "provenance": {
            "model_config": checkpoint_model_config,
            "evaluation_model_config": asdict(model_cfg),
            "training_protocol": asdict(train_cfg),
            "task_config": asdict(task_cfg),
            "evaluation": {
                "device": str(requested_device),
                "precision": evaluation_precision,
                "precision_source": precision_source,
                "batch_size": resolved_batch_size,
                **dict(evaluation_provenance or {}),
            },
        },
        "canonical_validation": canonical_result,
    }

    if challenge_per_class:
        if challenge_seed is None:
            raise ValueError("challenge_seed is required when challenge_per_class > 0.")
        challenge_spec = ChallengeSpec(
            seed=challenge_seed,
            per_stratum=challenge_per_class,
            length=task_cfg.length,
            dimension=task_cfg.dimension,
            mod=task_cfg.mod,
            generator_mode=task_cfg.generator_mode,
            corruption_rate=task_cfg.corruption_rate,
        )
        challenge = generate_challenge_set(challenge_spec)
        challenge_content_id = fingerprint_packed_instances(
            challenge.instances,
            namespace="diagnostic_challenge_validation",
            generation_config=challenge_spec.canonical_dict(),
        )
        challenge_result = _evaluate_population(
            model,
            challenge.instances,
            population_id=challenge_content_id,
            task_cfg=task_cfg,
            vocab=vocab,
            device=requested_device,
            precision=evaluation_precision,
            batch_size=resolved_batch_size,
            dataset_seed=challenge_seed,
        )
        challenge_result.update(
            {
                "challenge_id": challenge_spec.challenge_id,
                "challenge_content_sha256": challenge_content_id,
                "challenge_seed": challenge_seed,
                "requested_counts": challenge.provenance["requested_strata"],
                "realized_counts": challenge.provenance["realized_strata"],
                "provenance": challenge.provenance,
            }
        )
        artifact["diagnostic_challenge_validation"] = challenge_result

    return artifact


def write_diagnostic_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target

