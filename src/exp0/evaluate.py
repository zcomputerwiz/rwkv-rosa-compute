"""Evaluation harness and metrics reporting for Experiment 0."""

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import build_default_vocab


def _resolved_model_dict(
    model_cfg: ModelConfig,
    task_cfg: Task3SumConfig,
) -> Dict[str, Any]:
    """Return model metadata with task-derived interface dimensions resolved."""
    model_dict = asdict(model_cfg)
    model_dict["vocab_size"] = len(
        build_default_vocab(
            length=task_cfg.length,
            dimension=task_cfg.dimension,
        )
    )
    return model_dict


def canonical_run_config(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    task_cfg: Task3SumConfig,
    eval_seed: int,
    val_samples: int,
    seeds_run: List[int],
) -> Dict[str, Any]:
    """Return the complete deterministic configuration used for run identity."""
    model_dict = _resolved_model_dict(model_cfg, task_cfg)
    # Checkpoint content, not its machine-specific path, defines model identity.
    model_dict.pop("rwkv_checkpoint", None)

    train_dict = asdict(train_cfg)
    train_dict.pop("seed", None)

    task_dict = asdict(task_cfg)
    task_dict.pop("seed", None)

    return {
        "model": model_dict,
        "training_protocol": train_dict,
        "task_config": task_dict,
        "evaluation": {
            "eval_seed": eval_seed,
            "val_samples": val_samples,
            "seeds_run": sorted(seeds_run),
        },
    }


def compute_run_id(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    task_cfg: Task3SumConfig,
    eval_seed: int,
    val_samples: int,
    seeds_run: List[int],
) -> str:
    """Compute a deterministic ID from the canonical scientific run config."""
    config = canonical_run_config(
        model_cfg,
        train_cfg,
        task_cfg,
        eval_seed,
        val_samples,
        seeds_run,
    )
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]


def _initialization_from_results(
    model_cfg: ModelConfig,
    per_seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    initializations = [
        result["initialization"]
        for result in per_seed_results
        if "initialization" in result
    ]
    if initializations:
        first = initializations[0]
        if any(item != first for item in initializations[1:]):
            raise ValueError("Initialization provenance differs across seeds.")
        return first

    return {
        "mode": model_cfg.init_mode,
        "pretrained_scope": (
            "backbone_only"
            if model_cfg.architecture == "rwkv"
            and model_cfg.init_mode == "pretrained"
            else None
        ),
        "checkpoint_path": model_cfg.rwkv_checkpoint,
        "checkpoint_sha256": model_cfg.rwkv_checkpoint_sha256,
    }


def compile_experiment_report(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    task_cfg: Task3SumConfig,
    per_seed_results: List[Dict[str, Any]],
    majority_class_baseline: float,
    realized_mixture_counts: Dict[str, int],
    eval_seed: int = 9999,
    val_samples: int = 2000,
) -> Dict[str, Any]:
    """Compile an Experiment 0 report with unambiguous provenance."""
    filler_accuracies = [
        res.get("best_filler_accuracy", res.get("best_val_accuracy", 0.0))
        for res in per_seed_results
    ]
    cot_accuracies = [
        res.get("best_cot_accuracy", 0.0) for res in per_seed_results
    ]

    mean_filler_acc = (
        sum(filler_accuracies) / len(filler_accuracies)
        if filler_accuracies
        else 0.0
    )
    mean_cot_acc = (
        sum(cot_accuracies) / len(cot_accuracies) if cot_accuracies else 0.0
    )
    min_acc = min(filler_accuracies) if filler_accuracies else 0.0
    max_acc = max(filler_accuracies) if filler_accuracies else 0.0

    model_dict = _resolved_model_dict(model_cfg, task_cfg)
    train_dict = asdict(train_cfg)
    task_dict = asdict(task_cfg)
    train_dict.pop("seed", None)
    task_dict.pop("seed", None)

    resolved_vocab_sizes = {
        result["resolved_vocab_size"]
        for result in per_seed_results
        if "resolved_vocab_size" in result
    }
    if resolved_vocab_sizes and resolved_vocab_sizes != {model_dict["vocab_size"]}:
        raise ValueError(
            "Training/report vocabulary resolution disagrees across seeds: "
            f"expected {model_dict['vocab_size']}, got {sorted(resolved_vocab_sizes)}."
        )

    seeds_run = [res["seed"] for res in per_seed_results]
    run_config = canonical_run_config(
        model_cfg,
        train_cfg,
        task_cfg,
        eval_seed,
        val_samples,
        seeds_run,
    )
    run_id = compute_run_id(
        model_cfg,
        train_cfg,
        task_cfg,
        eval_seed,
        val_samples,
        seeds_run,
    )

    return {
        "run_id": run_id,
        "run_config": run_config,
        "model": model_dict,
        "initialization": _initialization_from_results(
            model_cfg,
            per_seed_results,
        ),
        "training_protocol": train_dict,
        "task_config": task_dict,
        "compute_budget": {
            "num_filler": (
                task_cfg.num_filler
                if task_cfg.num_filler is not None
                else task_cfg.length**2
            ),
            "length": task_cfg.length,
            "dimension": task_cfg.dimension,
        },
        "majority_class_baseline": majority_class_baseline,
        "realized_mixture_counts": realized_mixture_counts,
        "metrics": {
            "filler_accuracy": mean_filler_acc,
            "cot_accuracy": mean_cot_acc,
            "mean_accuracy": mean_filler_acc,
            "min_accuracy": min_acc,
            "max_accuracy": max_acc,
            "per_seed_accuracies": filler_accuracies,
        },
        "eval_seed": eval_seed,
        "val_samples": val_samples,
        "seeds_run": seeds_run,
        "per_seed_details": per_seed_results,
    }
