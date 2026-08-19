"""Evaluation harness and metrics reporting for Experiment 0."""

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig


def compute_run_id(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    task_cfg: Task3SumConfig,
    eval_seed: int,
    val_samples: int,
    seeds_run: List[int],
) -> str:
    """Compute a deterministic run ID from scientific configuration."""
    config_dict = {
        "architecture": model_cfg.architecture,
        "hidden_size": model_cfg.hidden_size,
        "num_hidden_layers": model_cfg.num_hidden_layers,
        "num_attention_heads": model_cfg.num_attention_heads,
        "length": task_cfg.length,
        "dimension": task_cfg.dimension,
        "num_filler": task_cfg.num_filler,
        "vocab_reduction": task_cfg.vocab_reduction,
        "mixture": train_cfg.mixture,
        "parallel_ratio": train_cfg.parallel_ratio,
        "filler_ratio": train_cfg.filler_ratio,
        "serial_ratio": train_cfg.serial_ratio,
        "immediate_ratio": train_cfg.immediate_ratio,
        "epochs": train_cfg.epochs,
        "batch_size": train_cfg.batch_size,
        "learning_rate": train_cfg.learning_rate,
        "eval_seed": eval_seed,
        "val_samples": val_samples,
        "seeds_run": sorted(seeds_run),
        "num_samples": task_cfg.num_samples,
    }
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:8]

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
    """Compile comprehensive experiment report conforming to docs/experiments.md metadata requirements."""
    filler_accuracies = [res.get("best_filler_accuracy", res.get("best_val_accuracy", 0.0)) for res in per_seed_results]
    cot_accuracies = [res.get("best_cot_accuracy", 0.0) for res in per_seed_results]

    mean_filler_acc = sum(filler_accuracies) / len(filler_accuracies) if filler_accuracies else 0.0
    mean_cot_acc = sum(cot_accuracies) / len(cot_accuracies) if cot_accuracies else 0.0

    # Retaining mean_acc logic for backward compatibility in max/min as well? The user asked to keep `metrics.mean_accuracy` as alias to `filler_accuracy`.
    # Let's keep min/max filler accuracies as min/max_accuracy for legacy tests maybe? Wait, user said "mean_accuracy may remain only as filler alias." Nothing about min/max. Let's just alias min/max to filler min/max too to be safe, or just remove if tests pass. The issue just says:
    # Use mean per-seed best values. Legacy mean_accuracy may remain only as filler alias.
    min_acc = min(filler_accuracies) if filler_accuracies else 0.0
    max_acc = max(filler_accuracies) if filler_accuracies else 0.0

    model_dict = asdict(model_cfg)
    train_dict = asdict(train_cfg)
    task_dict = asdict(task_cfg)

    train_dict.pop("seed", None)
    task_dict.pop("seed", None)

    seeds_run = [res["seed"] for res in per_seed_results]
    run_id = compute_run_id(model_cfg, train_cfg, task_cfg, eval_seed, val_samples, seeds_run)

    report = {
        "run_id": run_id,
        "model": model_dict,
        "training_protocol": train_dict,
        "task_config": task_dict,
        "compute_budget": {
            "num_filler": task_cfg.num_filler if task_cfg.num_filler is not None else task_cfg.length**2,
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
        "seeds_run": [res["seed"] for res in per_seed_results],
        "per_seed_details": per_seed_results,
    }
    return report
