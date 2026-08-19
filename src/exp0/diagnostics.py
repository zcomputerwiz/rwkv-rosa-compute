"""Generation diagnostics for Experiment 0 parallel chain-of-thought data."""

from contextlib import nullcontext
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from exp0.dataset import (
    COT_DIAG_MATCH_RESULT,
    COT_DIAG_PAIR_POSITION,
    COT_DIAG_SUM_RESULT,
)


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def evaluate_cot_diagnostics(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ans_token_id: int,
    ans_true_id: int,
    ans_false_id: int,
    *,
    precision: str = "fp32",
    non_blocking: bool = False,
) -> Dict[str, Any]:
    """Measure teacher-forced intermediate-token generation on parallel CoT.

    Final answer accuracy with a ground-truth CoT prefix is retained under the
    explicit name ``cot_answer_given_cot_accuracy`` because it is not evidence
    that the model independently computed 3SUM. The informative diagnostics
    score next-token predictions at the intermediate pair/result slots.

    Reduced-vocabulary CoT intentionally randomizes which member of a pair and
    which coordinate digit are emitted. Exact-token metrics mirror the paper's
    evaluator, while semantic metrics accept any token that represents the same
    valid pair/result and therefore remove that artificial stochastic ceiling.
    """
    model.eval()

    counter_names = (
        "pair_exact",
        "pair_semantic",
        "pair_count",
        "sum_exact",
        "sum_semantic",
        "sum_count",
        "match_exact",
        "match_count",
        "result_semantic",
        "result_count",
        "answer_correct",
        "answer_count",
    )
    counters = {
        name: torch.zeros((), dtype=torch.int64, device=device)
        for name in counter_names
    }
    result_nll_sum = torch.zeros((), dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch in val_loader:
            targets_cpu = batch["targets"]
            ans_mask_cpu = targets_cpu[:, :-1].eq(ans_token_id)
            ans_counts = ans_mask_cpu.sum(dim=1)
            bad = ans_counts.ne(1)
            if torch.any(bad):
                bad_count = int(ans_counts[bad][0].item())
                raise ValueError(
                    "CoT sequence must have exactly one ANS prediction position. "
                    f"Found {bad_count}."
                )
            ans_positions_cpu = ans_mask_cpu.to(dtype=torch.int64).argmax(dim=1)

            input_tuples = batch["input_tuples"].to(
                device,
                non_blocking=non_blocking,
            )
            targets = targets_cpu.to(device, non_blocking=non_blocking)
            has_3sum = batch["has_3sum"].to(
                device,
                non_blocking=non_blocking,
            )
            diag_type = batch["cot_diag_type"].to(
                device,
                non_blocking=non_blocking,
            )
            valid_ids = batch["cot_valid_ids"].to(
                device,
                non_blocking=non_blocking,
            )

            with _autocast_context(device, precision):
                if hasattr(model, "loss_logits"):
                    logits = model.loss_logits(input_tuples, targets)
                else:
                    logits = model(input_tuples, targets)[:, :-1, :]

            next_targets = targets[:, 1:]
            next_types = diag_type[:, 1:]
            next_valid_ids = valid_ids[:, 1:, :]
            predictions = logits.argmax(dim=-1)
            exact = predictions.eq(next_targets)
            semantic = predictions.unsqueeze(-1).eq(next_valid_ids).any(dim=-1)

            pair_mask = next_types.eq(COT_DIAG_PAIR_POSITION)
            sum_mask = next_types.eq(COT_DIAG_SUM_RESULT)
            match_mask = next_types.eq(COT_DIAG_MATCH_RESULT)
            result_mask = sum_mask | match_mask

            counters["pair_exact"].add_(exact[pair_mask].sum())
            counters["pair_semantic"].add_(semantic[pair_mask].sum())
            counters["pair_count"].add_(pair_mask.sum())

            counters["sum_exact"].add_(exact[sum_mask].sum())
            counters["sum_semantic"].add_(semantic[sum_mask].sum())
            counters["sum_count"].add_(sum_mask.sum())

            counters["match_exact"].add_(exact[match_mask].sum())
            counters["match_count"].add_(match_mask.sum())

            counters["result_semantic"].add_(semantic[result_mask].sum())
            counters["result_count"].add_(result_mask.sum())

            per_token_nll = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                next_targets.reshape(-1),
                reduction="none",
            ).view_as(next_targets)
            result_nll_sum.add_(per_token_nll[result_mask].sum().to(torch.float64))

            ans_positions = ans_positions_cpu.to(
                device,
                non_blocking=non_blocking,
            )
            batch_indices = torch.arange(targets.shape[0], device=device)
            answer_predictions = predictions[batch_indices, ans_positions]
            expected_answers = torch.where(
                has_3sum,
                torch.full_like(answer_predictions, ans_true_id),
                torch.full_like(answer_predictions, ans_false_id),
            )
            counters["answer_correct"].add_(
                answer_predictions.eq(expected_answers).sum()
            )
            counters["answer_count"].add_(targets.shape[0])

    # One host transfer at the end keeps CUDA validation asynchronous across
    # batches while still returning ordinary JSON-serializable Python values.
    counter_values = {
        name: int(value.item())
        for name, value in counters.items()
    }
    result_nll_value = float(result_nll_sum.item())

    return {
        "cot_answer_given_cot_accuracy": _safe_ratio(
            counter_values["answer_correct"],
            counter_values["answer_count"],
        ),
        "cot_pair_position_token_accuracy": _safe_ratio(
            counter_values["pair_exact"],
            counter_values["pair_count"],
        ),
        "cot_pair_position_semantic_accuracy": _safe_ratio(
            counter_values["pair_semantic"],
            counter_values["pair_count"],
        ),
        "cot_sum_token_accuracy": _safe_ratio(
            counter_values["sum_exact"],
            counter_values["sum_count"],
        ),
        "cot_sum_semantic_accuracy": _safe_ratio(
            counter_values["sum_semantic"],
            counter_values["sum_count"],
        ),
        "cot_match_index_accuracy": _safe_ratio(
            counter_values["match_exact"],
            counter_values["match_count"],
        ),
        "cot_result_semantic_accuracy": _safe_ratio(
            counter_values["result_semantic"],
            counter_values["result_count"],
        ),
        "cot_result_nll": (
            result_nll_value / counter_values["result_count"]
            if counter_values["result_count"]
            else None
        ),
        "cot_diagnostic_counts": {
            "pair_position_tokens": counter_values["pair_count"],
            "sum_results": counter_values["sum_count"],
            "match_results": counter_values["match_count"],
            "all_results": counter_values["result_count"],
            "answers": counter_values["answer_count"],
        },
    }
