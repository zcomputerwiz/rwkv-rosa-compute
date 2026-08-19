"""Evaluation harness and metrics reporting for Experiment 0."""

from dataclasses import asdict
from typing import Any, Dict, List

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig


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

    report = {
        "model": asdict(model_cfg),
        "training_protocol": asdict(train_cfg),
        "task_config": asdict(task_cfg),
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
