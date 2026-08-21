"""CPU tests for deterministic Experiment 0 checkpoint re-evaluation."""

from dataclasses import asdict

import pytest
import torch

from exp0.checkpoint_analysis import (
    evaluate_checkpoint,
    fingerprint_packed_instances,
    structural_features,
)
from exp0.checkpointing import CHECKPOINT_VERSION
from exp0.config import (
    ModelConfig,
    Task3SumConfig,
    TrainConfig,
    drop_identity_neutral_fields,
)
from exp0.dataset import PackedInstances, build_default_vocab
from exp0.train import create_model


def _packed(label: bool = True) -> PackedInstances:
    return PackedInstances(
        tuples=torch.tensor([[[1, 2, 3], [2, 3, 4], [7, 5, 3]]], dtype=torch.uint8),
        has_3sum=torch.tensor([label], dtype=torch.bool),
        matching_indices=torch.tensor([[0, 1, 2]], dtype=torch.int16),
        construction_arm=torch.tensor([1], dtype=torch.int8),
        corruption_count=torch.tensor([-1], dtype=torch.int8),
    )


def test_validation_fingerprint_is_deterministic_and_content_sensitive():
    config = {"eval_seed": 9999, "val_samples": 1}
    first = fingerprint_packed_instances(
        _packed(), namespace="canonical_validation", generation_config=config
    )
    second = fingerprint_packed_instances(
        _packed(), namespace="canonical_validation", generation_config=config
    )
    changed = fingerprint_packed_instances(
        _packed(label=False),
        namespace="canonical_validation",
        generation_config=config,
    )
    assert first == second
    assert first != changed
    assert len(first) == 64


def test_structural_near_match_features():
    valid = structural_features([(1, 2, 3), (2, 3, 4), (7, 5, 3)])
    assert valid == {
        "total_candidate_triples": 1,
        "candidate_coordinate_match_counts": [0, 0, 0, 1],
        "num_two_of_three_near_misses": 0,
        "max_matched_coordinate_count_among_non_solutions": None,
        "first_valid_witness_position": 0,
    }

    near = structural_features([(1, 2, 3), (2, 3, 4), (7, 5, 4)])
    assert near["candidate_coordinate_match_counts"] == [0, 0, 1, 0]
    assert near["num_two_of_three_near_misses"] == 1
    assert near["max_matched_coordinate_count_among_non_solutions"] == 2
    assert near["first_valid_witness_position"] is None


def _write_tiny_completed_checkpoint(path):
    task_cfg = Task3SumConfig(
        length=3,
        dimension=3,
        num_filler=0,
        num_samples=8,
    )
    vocab = build_default_vocab(length=3, dimension=3, mod=task_cfg.mod)
    model_cfg = ModelConfig(
        architecture="llama",
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=24,
        vocab_size=len(vocab),
        output_vocab_size=len(vocab),
        device="cpu",
    )
    train_cfg = TrainConfig(
        seed=7,
        batch_size=2,
        precision="fp32",
        epochs=1,
        immediate_protocol=False,
    )
    model = create_model(
        model_cfg,
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
        compact_reduced_features=True,
    )
    model_signature = asdict(model_cfg)
    model_signature.pop("rwkv_checkpoint")
    signature = {
        "run_id": "tiny-eval",
        "model": model_signature,
        "training": drop_identity_neutral_fields(asdict(train_cfg)),
        "task": asdict(task_cfg),
        "train_dataset_size": task_cfg.num_samples,
        "realized_format_counts": {
            "parallel_cot": 4,
            "filler": 4,
            "serial_cot": 0,
            "immediate": 0,
            "neutral": 0,
        },
        "epochs": 1,
        "steps_per_epoch": 4,
    }
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "signature": signature,
            "model_state_dict": model.state_dict(),
            "progress": {
                "epoch": 1,
                "epoch_seed": None,
                "samples_consumed_in_epoch": 0,
                "optimizer_steps": 4,
                "completed": {},
                "partial_epoch": None,
            },
            "initialization": {"mode": "random"},
        },
        path,
    )


def test_completed_checkpoint_evaluator_uses_cpu_without_optimizer(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    _write_tiny_completed_checkpoint(checkpoint)
    artifact = evaluate_checkpoint(
        checkpoint,
        device="cpu",
        eval_seed=9999,
        val_samples=6,
        batch_size=2,
    )
    canonical = artifact["canonical_validation"]
    assert artifact["checkpoint"]["training_seed"] == 7
    assert artifact["provenance"]["evaluation"]["precision"] == "fp32"
    assert canonical["population_size"] == 6
    assert len(canonical["records"]) == 6
    assert [record["index"] for record in canonical["records"]] == list(range(6))
    assert all("prediction_margin" in record for record in canonical["records"])
    assert all(
        "num_two_of_three_near_misses" in record for record in canonical["records"]
    )


def test_incomplete_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    _write_tiny_completed_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["progress"]["epoch"] = 0
    torch.save(payload, checkpoint)
    try:
        evaluate_checkpoint(
            checkpoint,
            device="cpu",
            eval_seed=9999,
            val_samples=2,
        )
    except ValueError as exc:
        assert "not a completed" in str(exc)
    else:
        raise AssertionError("Incomplete checkpoint was accepted")


def test_mismatched_run_report_config_is_rejected_before_evaluation(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    _write_tiny_completed_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    signature = payload["signature"]
    expected_task = dict(signature["task"])
    expected_task.pop("seed", None)
    expected_training = dict(signature["training"])
    expected_training.pop("seed", None)
    expected = {
        "model": signature["model"],
        "task_config": {**expected_task, "length": 4},
        "training_protocol": expected_training,
        "evaluation": {"seeds_run": [7]},
    }
    with pytest.raises(ValueError, match="task_config"):
        evaluate_checkpoint(
            checkpoint,
            device="cpu",
            eval_seed=9999,
            val_samples=2,
            expected_run_config=expected,
        )

