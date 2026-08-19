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

    counters = {
        "pair_exact": 0,
        "pair_semantic": 0,
        "pair_count": 0,
        "sum_exact": 0,
        "sum_semantic": 0,
        "sum_count": 0,
        "match_exact": 0,
        "match_count": 0,
        "result_semantic": 0,
        "result_count": 0,
        "answer_correct": 0,
        "answer_count": 0,
    }
    result_nll_sum = 0.0

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

            counters["pair_exact"] += int(exact[pair_mask].sum().item())
            counters["pair_semantic"] += int(semantic[pair_mask].sum().item())
            counters["pair_count"] += int(pair_mask.sum().item())

            counters["sum_exact"] += int(exact[sum_mask].sum().item())
            counters["sum_semantic"] += int(semantic[sum_mask].sum().item())
            counters["sum_count"] += int(sum_mask.sum().item())

            counters["match_exact"] += int(exact[match_mask].sum().item())
            counters["match_count"] += int(match_mask.sum().item())

            counters["result_semantic"] += int(
                semantic[result_mask].sum().item()
            )
            result_count = int(result_mask.sum().item())
            counters["result_count"] += result_count
            if result_count:
                result_nll_sum += float(
                    F.cross_entropy(
                        logits[result_mask].float(),
                        next_targets[result_mask],
                        reduction="sum",
                    ).item()
                )

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
            counters["answer_correct"] += int(
                answer_predictions.eq(expected_answers).sum().item()
            )
            counters["answer_count"] += targets.shape[0]

    return {
        "cot_answer_given_cot_accuracy": _safe_ratio(
            counters["answer_correct"],
            counters["answer_count"],
        ),
        "cot_pair_position_token_accuracy": _safe_ratio(
            counters["pair_exact"],
            counters["pair_count"],
        ),
        "cot_pair_position_semantic_accuracy": _safe_ratio(
            counters["pair_semantic"],
            counters["pair_count"],
        ),
        "cot_sum_token_accuracy": _safe_ratio(
            counters["sum_exact"],
            counters["sum_count"],
        ),
        "cot_sum_semantic_accuracy": _safe_ratio(
            counters["sum_semantic"],
            counters["sum_count"],
        ),
        "cot_match_index_accuracy": _safe_ratio(
            counters["match_exact"],
            counters["match_count"],
        ),
        "cot_result_semantic_accuracy": _safe_ratio(
            counters["result_semantic"],
            counters["result_count"],
        ),
        "cot_result_nll": (
            result_nll_sum / counters["result_count"]
            if counters["result_count"]
            else None
        ),
        "cot_diagnostic_counts": {
            "pair_position_tokens": counters["pair_count"],
            "sum_results": counters["sum_count"],
            "match_results": counters["match_count"],
            "all_results": counters["result_count"],
            "answers": counters["answer_count"],
        },
    }
