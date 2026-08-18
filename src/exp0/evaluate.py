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
) -> Dict[str, Any]:
    """Compile comprehensive experiment report conforming to docs/experiments.md metadata requirements."""
    accuracies = [res["best_val_accuracy"] for res in per_seed_results]

    mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    min_acc = min(accuracies) if accuracies else 0.0
    max_acc = max(accuracies) if accuracies else 0.0

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
            "mean_accuracy": mean_acc,
            "min_accuracy": min_acc,
            "max_accuracy": max_acc,
            "per_seed_accuracies": accuracies,
        },
        "seeds_run": [res["seed"] for res in per_seed_results],
        "per_seed_details": per_seed_results,
    }
    return report
