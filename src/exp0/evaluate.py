"""Evaluation harness and metrics reporting for Experiment 0."""

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List

from exp0.config import (
    ModelConfig,
    Task3SumConfig,
    TrainConfig,
    drop_disabled_early_stop_fields,
)
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

    train_dict = drop_disabled_early_stop_fields(asdict(train_cfg))
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
        "cot_pair_position_semantic_ceiling",
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
    ambiguity_flags = {
        bool(
            result.get("best_cot_diagnostics", {}).get(
                "cot_first_slot_format_ambiguous", False
            )
        )
        for result in per_seed_results
        if result.get("best_cot_diagnostics")
    }
    if len(ambiguity_flags) > 1:
        raise ValueError(
            "Seeds disagree on first-slot format ambiguity; the mixture ratios "
            "must be identical across seeds of one run."
        )
    aggregated["cot_first_slot_format_ambiguous"] = (
        ambiguity_flags.pop() if ambiguity_flags else False
    )
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

    filler_prediction_counts = [
        res.get("best_filler_answer_prediction_counts")
        for res in per_seed_results
    ]
    filler_prediction_counts = [item for item in filler_prediction_counts if item]

    early_stopping_per_seed = [
        res.get("early_stopping") for res in per_seed_results
    ]
    early_stopping_per_seed = [item for item in early_stopping_per_seed if item]

    def _epoch_budget(item: Dict[str, Any]) -> Any:
        """The budget the loop actually ran against.

        The immediate-answer protocol multiplies the requested epochs, so a run
        that stopped early must be compared against the effective ceiling.
        Comparing against the requested value would report an early-stopped N=0
        run as fixed-budget. Older reports carry only the requested key.
        """
        effective = item.get("epochs_effective")
        return effective if effective is not None else item.get("epochs_requested")

    any_early_stopped = any(
        item.get("epochs_trained") is not None
        and _epoch_budget(item) is not None
        and item["epochs_trained"] < _epoch_budget(item)
        for item in early_stopping_per_seed
    )

    immediate_protocol_per_seed = [
        res.get("immediate_protocol") for res in per_seed_results
    ]
    immediate_protocol_per_seed = [
        item for item in immediate_protocol_per_seed if item
    ]
    any_immediate_protocol = any(
        item.get("applied", False) for item in immediate_protocol_per_seed
    )

    metrics: Dict[str, Any] = {
        "filler_accuracy": mean_filler_acc,
        "early_stopping_per_seed": early_stopping_per_seed,
        "fixed_budget_run": not any_early_stopped,
        "immediate_protocol_per_seed": immediate_protocol_per_seed,
        "immediate_protocol_applied_any_seed": any_immediate_protocol,
        "filler_answer_prediction_counts_per_seed": filler_prediction_counts,
        "filler_answer_is_degenerate_any_seed": any(
            item.get("degenerate_predictor", False)
            for item in filler_prediction_counts
        ),
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
            "early_stopping_per_seed": (
                "Per-seed early-stopping settings and outcome. criterion_reached "
                "records whether the patience rule was satisfied; triggered is "
                "true only when that criterion actually shortened the requested "
                "budget. Epoch-number fields are 1-based."
            ),
            "fixed_budget_run": (
                "False only when at least one seed trained fewer epochs than "
                "requested. Reaching an early-stop criterion on the final "
                "requested epoch remains a fixed-budget run. Early-stopped runs "
                "select the epoch at which the metric first reached target, so "
                "their reported accuracy is an upward-biased peak rather than an "
                "end-of-budget measurement. Do NOT place them on the same "
                "accuracy-vs-compute curve as fixed-budget runs, and do not "
                "compare them across arms unless every arm stopped on the same "
                "rule."
            ),
            "immediate_protocol_per_seed": (
                "Per-seed record of the immediate-answer protocol substitution, "
                "which fires when num_filler is 0 or the mixture is "
                "'immediate'. It multiplies the requested epochs and replaces "
                "weight decay and gradient clip, so each value is reported both "
                "as requested and as it actually ran."
            ),
            "immediate_protocol_applied_any_seed": (
                "True when at least one seed ran under the immediate-answer "
                "protocol. Such a run trained more epochs than the canonical run "
                "config records, under a different weight decay and gradient "
                "clip. An N=0 arm is therefore NOT a compute-matched control for "
                "an N>0 arm from the same command line; read "
                "immediate_protocol_per_seed before comparing them."
            ),
            "filler_answer_prediction_counts_per_seed": (
                "Predicted True/False/other histogram at the validation ANS "
                "position. Read this before interpreting any accuracy at or "
                "near majority_class_baseline."
            ),
            "filler_answer_is_degenerate_any_seed": (
                "True when at least one seed emitted the same answer token for "
                "every validation example. Such a run has no accuracy signal, "
                "whatever the reported accuracy is."
            ),
            "cot_pair_position_token_accuracy": (
                "STRUCTURAL, not computational: exact next-token accuracy on "
                "randomized reduced CoT pair tokens. Measures layout and "
                "positional counting, not 3SUM."
            ),
            "cot_pair_position_semantic_accuracy": (
                "STRUCTURAL, not computational: pair-position accuracy accepting "
                "either valid summand token. Emitting labels[i] at every pair "
                "slot scores 1.0 without performing any pairwise computation, "
                "so this belongs with cot_answer_given_cot_accuracy rather than "
                "with the result metrics."
            ),
            "cot_pair_position_semantic_ceiling": (
                "Hard ceiling on cot_pair_position_semantic_accuracy for this "
                "mixture. Formats share the tuple prefix and the ':' separator, "
                "so when a non-CoT format outweighs parallel_ratio/2 the first "
                "pair slot cannot be predicted and the ceiling is "
                "(pair_count - 1) / pair_count. A value at the ceiling means "
                "saturated, not partially learned."
            ),
            "cot_first_slot_format_ambiguous": (
                "Whether the mixture makes the first post-separator target "
                "unpredictable from context; sets the ceiling above."
            ),
            "cot_sum_token_accuracy": (
                "Exact next-token accuracy on sampled unmatched pair-sum tokens."
            ),
            "cot_sum_semantic_accuracy": (
                "Unmatched pair-sum accuracy accepting any coordinate digit valid "
                "for the pair under vocabulary reduction."
            ),
            "cot_match_index_accuracy": (
                "Exact accuracy for source-faithful matched-pair third-index "
                "tokens. Match slots are a small minority of result slots, so a "
                "model that has not learned 3SUM emits a sum digit everywhere "
                "and scores exactly 0.0; compare against the unconditional "
                "baseline, not the given-match-known one."
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
                "layout and source-faithful k>j candidate sets. "
                "match_index_accuracy is the unconditional baseline (uniform "
                "choice over the result-slot vocabulary) and is the correct "
                "comparison for cot_match_index_accuracy. "
                "match_index_accuracy_given_match_known conditions on already "
                "knowing the pair matches and is reported for reference only; "
                "it is not a baseline any measured model competes against."
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
