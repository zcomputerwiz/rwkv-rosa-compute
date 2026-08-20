"""Evaluation harness and metrics reporting for Experiment 0."""

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import FORMAT_NAMES, build_default_vocab


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


def _mean_present(values: List[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _aggregate_per_pair(
    per_seed_results: List[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    per_seed_pairs = [
        result.get("best_cot_diagnostics", {}).get("cot_per_pair", [])
        for result in per_seed_results
    ]
    per_seed_pairs = [pairs for pairs in per_seed_pairs if pairs]
    if not per_seed_pairs:
        return []

    pair_count = len(per_seed_pairs[0])
    if any(len(pairs) != pair_count for pairs in per_seed_pairs):
        raise ValueError("CoT per-pair diagnostic layouts differ across seeds.")

    metrics = (
        "pair_semantic_accuracy",
        "sum_semantic_accuracy",
        "match_index_accuracy",
        "result_semantic_accuracy",
    )
    aggregated = []
    for index in range(pair_count):
        template = per_seed_pairs[0][index]
        item: Dict[str, Any] = {
            "pair_index": template["pair_index"],
            "i": template["i"],
            "j": template["j"],
            "result_count": template.get("result_count"),
        }
        for name in metrics:
            item[name] = _mean_present(
                [pairs[index].get(name) for pairs in per_seed_pairs]
            )
        aggregated.append(item)
    return aggregated


def _aggregate_cot_diagnostics(
    per_seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metric_names = [
        "cot_answer_given_cot_accuracy",
        "cot_pair_position_token_accuracy",
        "cot_pair_position_semantic_accuracy",
        "cot_sum_token_accuracy",
        "cot_sum_semantic_accuracy",
        "cot_match_index_accuracy",
        "cot_result_semantic_accuracy",
        "cot_result_nll",
        "cot_result_nll_floor",
    ]
    aggregated: Dict[str, Any] = {}
    for name in metric_names:
        aggregated[name] = _mean_present(
            [
                result.get("best_cot_diagnostics", {}).get(name)
                for result in per_seed_results
            ]
        )

    baselines = [
        result.get("best_cot_diagnostics", {}).get("cot_chance_baselines")
        for result in per_seed_results
    ]
    baselines = [item for item in baselines if item]
    if baselines:
        baseline_keys = baselines[0].keys()
        aggregated["cot_chance_baselines"] = {
            key: _mean_present([item.get(key) for item in baselines])
            for key in baseline_keys
        }
    else:
        aggregated["cot_chance_baselines"] = {}
    aggregated["cot_per_pair"] = _aggregate_per_pair(per_seed_results)
    return aggregated


def _aggregate_training_by_format(
    per_seed_results: List[Dict[str, Any]],
) -> Dict[str, float | None]:
    return {
        name: _mean_present(
            [
                result.get("best_online_train_answer_accuracy_by_format", {}).get(
                    name
                )
                for result in per_seed_results
            ]
        )
        for name in FORMAT_NAMES
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
    online_train_answer_accuracies = [
        res.get("best_online_train_answer_accuracy", 0.0)
        for res in per_seed_results
    ]

    mean_filler_acc = (
        sum(filler_accuracies) / len(filler_accuracies)
        if filler_accuracies
        else 0.0
    )
    mean_online_train_answer_acc = (
        sum(online_train_answer_accuracies) / len(online_train_answer_accuracies)
        if online_train_answer_accuracies
        else 0.0
    )
    min_acc = min(filler_accuracies) if filler_accuracies else 0.0
    max_acc = max(filler_accuracies) if filler_accuracies else 0.0
    cot_diagnostics = _aggregate_cot_diagnostics(per_seed_results)
    training_by_format = _aggregate_training_by_format(per_seed_results)

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

    expected_output_vocab_size = (
        model_dict["output_vocab_size"]
        if model_dict["output_vocab_size"] is not None
        else model_dict["vocab_size"]
    )
    reported_output_vocab_sizes = {
        result["output_vocab_size"]
        for result in per_seed_results
        if "output_vocab_size" in result
    }
    if reported_output_vocab_sizes and reported_output_vocab_sizes != {
        expected_output_vocab_size
    }:
        raise ValueError(
            "Training/report output head resolution disagrees across seeds: "
            f"expected {expected_output_vocab_size}, "
            f"got {sorted(reported_output_vocab_sizes)}."
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

    metrics: Dict[str, Any] = {
        "filler_accuracy": mean_filler_acc,
        "mean_accuracy": mean_filler_acc,
        "min_accuracy": min_acc,
        "max_accuracy": max_acc,
        "per_seed_accuracies": filler_accuracies,
        "best_online_training_answer_accuracy": mean_online_train_answer_acc,
        "best_online_training_answer_accuracy_by_format": training_by_format,
        **cot_diagnostics,
    }

    return {
        "run_id": run_id,
        "run_config": run_config,
        "model": model_dict,
        "initialization": _initialization_from_results(model_cfg, per_seed_results),
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
        "metrics": metrics,
        "metric_semantics": {
            "best_online_training_answer_accuracy": (
                "online fit diagnostic: best per-epoch answer accuracy accumulated "
                "from each training batch's existing forward pass before that "
                "batch's optimizer update. Mixed-format runs can be shortcut-"
                "contaminated, so use the per-format breakdown for interpretation."
            ),
            "best_online_training_answer_accuracy_by_format": (
                "Best online answer accuracy split by realized training format."
            ),
            "cot_answer_given_cot_accuracy": (
                "Final True/False accuracy when the ground-truth CoT prefix is "
                "teacher-forced; not evidence of independent 3SUM computation."
            ),
            "cot_pair_position_token_accuracy": (
                "Exact next-token accuracy on randomized reduced CoT pair tokens."
            ),
            "cot_pair_position_semantic_accuracy": (
                "Pair-position accuracy accepting either valid summand token."
            ),
            "cot_sum_token_accuracy": (
                "Exact next-token accuracy on sampled unmatched pair-sum tokens."
            ),
            "cot_sum_semantic_accuracy": (
                "Unmatched pair-sum accuracy accepting any coordinate digit valid "
                "for the pair under vocabulary reduction."
            ),
            "cot_match_index_accuracy": (
                "Exact accuracy for source-faithful matched-pair third-index tokens."
            ),
            "cot_result_semantic_accuracy": (
                "Semantic accuracy across all CoT result slots (sum or match)."
            ),
            "cot_result_nll": (
                "Mean teacher-forced NLL over CoT result slots only."
            ),
            "cot_result_nll_floor": (
                "Expected irreducible result-slot NLL from randomized coordinate "
                "selection; match-index targets are deterministic."
            ),
            "cot_chance_baselines": (
                "Structured random baselines computed from the fixed validation "
                "layout and source-faithful k>j candidate sets."
            ),
            "cot_per_pair": (
                "Per-(i,j) specialization metrics in lexicographic pair order."
            ),
        },
        "eval_seed": eval_seed,
        "val_samples": val_samples,
        "seeds_run": seeds_run,
        "per_seed_details": per_seed_results,
    }
