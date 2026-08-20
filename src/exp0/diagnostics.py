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


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _pair_list(length: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(length) for j in range(i + 1, length)]


def evaluate_cot_diagnostics(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ans_token_id: int,
    ans_true_id: int,
    ans_false_id: int,
    *,
    task_length: int,
    task_mod: int = 10,
    precision: str = "fp32",
    non_blocking: bool = False,
) -> Dict[str, Any]:
    """Measure teacher-forced intermediate-token generation on parallel CoT.

    In addition to aggregate metrics, report per-pair specialization, explicit
    structured-chance baselines, and the irreducible NLL floor introduced by
    randomized coordinate selection in reduced-vocabulary sum targets.
    """
    model.eval()
    pairs = _pair_list(task_length)
    pair_count = len(pairs)
    pair_j = torch.tensor(
        [j for _, j in pairs],
        dtype=torch.float64,
        device=device,
    )

    scalar_names = (
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
        "result_nll_sum",
        "result_nll_floor_sum",
        "pair_exact_chance_sum",
        "pair_semantic_chance_sum",
        "sum_exact_chance_sum",
        "sum_semantic_chance_sum",
        "match_exact_chance_sum",
        "result_semantic_chance_sum",
    )
    counters = {
        name: torch.zeros((), dtype=torch.float64, device=device)
        for name in scalar_names
    }
    per_pair_names = (
        "pair_semantic_correct",
        "pair_semantic_count",
        "sum_semantic_correct",
        "sum_semantic_count",
        "match_exact_correct",
        "match_exact_count",
        "result_semantic_correct",
        "result_semantic_count",
    )
    per_pair = {
        name: torch.zeros(pair_count, dtype=torch.float64, device=device)
        for name in per_pair_names
    }

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
                device, non_blocking=non_blocking
            )
            targets = targets_cpu.to(device, non_blocking=non_blocking)
            has_3sum = batch["has_3sum"].to(device, non_blocking=non_blocking)
            diag_type = batch["cot_diag_type"].to(device, non_blocking=non_blocking)
            valid_ids = batch["cot_valid_ids"].to(device, non_blocking=non_blocking)
            pair_indices = batch["cot_pair_index"].to(
                device, non_blocking=non_blocking
            )
            stochastic_floor = batch["cot_stochastic_nll_floor"].to(
                device, non_blocking=non_blocking
            )

            with _autocast_context(device, precision):
                if hasattr(model, "loss_logits"):
                    logits = model.loss_logits(input_tuples, targets)
                else:
                    logits = model(input_tuples, targets)[:, :-1, :]

            next_targets = targets[:, 1:]
            next_types = diag_type[:, 1:]
            next_valid_ids = valid_ids[:, 1:, :]
            next_pair_indices = pair_indices[:, 1:].to(dtype=torch.long)
            next_floor = stochastic_floor[:, 1:]
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

            valid_counts = next_valid_ids.ge(0).sum(dim=-1).to(torch.float64)
            counters["pair_exact_chance_sum"].add_(
                pair_mask.sum().to(torch.float64) / task_length
            )
            counters["pair_semantic_chance_sum"].add_(
                (valid_counts * pair_mask).sum() / task_length
            )
            counters["sum_exact_chance_sum"].add_(
                sum_mask.sum().to(torch.float64) / task_mod
            )
            sum_sem_chance = (valid_counts / task_mod) * sum_mask
            counters["sum_semantic_chance_sum"].add_(sum_sem_chance.sum())

            # Empty advanced-index tensors and empty scatter_add inputs are safe;
            # avoiding Python truth-tests here prevents per-batch CUDA syncs.
            matched_pair_indices = next_pair_indices[match_mask]
            eligible = task_length - pair_j[matched_pair_indices] - 1.0
            match_chance = eligible.reciprocal()
            counters["match_exact_chance_sum"].add_(match_chance.sum())
            counters["result_semantic_chance_sum"].add_(
                match_chance.sum() + sum_sem_chance.sum()
            )

            per_token_nll = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                next_targets.reshape(-1),
                reduction="none",
            ).view_as(next_targets)
            counters["result_nll_sum"].add_(per_token_nll[result_mask].sum())
            counters["result_nll_floor_sum"].add_(next_floor[result_mask].sum())

            for mask, correct, count_name, correct_name in (
                (pair_mask, semantic, "pair_semantic_count", "pair_semantic_correct"),
                (sum_mask, semantic, "sum_semantic_count", "sum_semantic_correct"),
                (match_mask, exact, "match_exact_count", "match_exact_correct"),
                (
                    result_mask,
                    semantic,
                    "result_semantic_count",
                    "result_semantic_correct",
                ),
            ):
                indices = next_pair_indices[mask]
                ones = torch.ones_like(indices, dtype=torch.float64)
                per_pair[count_name].scatter_add_(0, indices, ones)
                per_pair[correct_name].scatter_add_(
                    0, indices, correct[mask].to(torch.float64)
                )

            ans_positions = ans_positions_cpu.to(device, non_blocking=non_blocking)
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

    # Preserve the CUDA harness's single result synchronization by flattening
    # every scalar and per-pair accumulator into one host transfer.
    scalar_tensor = torch.stack([counters[name] for name in scalar_names])
    pair_tensor = torch.cat([per_pair[name] for name in per_pair_names])
    host_values = torch.cat((scalar_tensor, pair_tensor)).cpu().tolist()

    scalar_values = dict(zip(scalar_names, host_values[: len(scalar_names)]))
    offset = len(scalar_names)
    pair_values: dict[str, list[float]] = {}
    for name in per_pair_names:
        pair_values[name] = host_values[offset : offset + pair_count]
        offset += pair_count

    per_pair_report = []
    for index, (i, j) in enumerate(pairs):
        result_count = pair_values["result_semantic_count"][index]
        per_pair_report.append(
            {
                "pair_index": index,
                "i": i,
                "j": j,
                "pair_semantic_accuracy": _safe_ratio(
                    pair_values["pair_semantic_correct"][index],
                    pair_values["pair_semantic_count"][index],
                ),
                "sum_semantic_accuracy": _safe_ratio(
                    pair_values["sum_semantic_correct"][index],
                    pair_values["sum_semantic_count"][index],
                ),
                "match_index_accuracy": _safe_ratio(
                    pair_values["match_exact_correct"][index],
                    pair_values["match_exact_count"][index],
                ),
                "result_semantic_accuracy": _safe_ratio(
                    pair_values["result_semantic_correct"][index], result_count
                ),
                "result_count": int(result_count),
            }
        )

    result_count = scalar_values["result_count"]
    return {
        "cot_answer_given_cot_accuracy": _safe_ratio(
            scalar_values["answer_correct"], scalar_values["answer_count"]
        ),
        "cot_pair_position_token_accuracy": _safe_ratio(
            scalar_values["pair_exact"], scalar_values["pair_count"]
        ),
        "cot_pair_position_semantic_accuracy": _safe_ratio(
            scalar_values["pair_semantic"], scalar_values["pair_count"]
        ),
        "cot_sum_token_accuracy": _safe_ratio(
            scalar_values["sum_exact"], scalar_values["sum_count"]
        ),
        "cot_sum_semantic_accuracy": _safe_ratio(
            scalar_values["sum_semantic"], scalar_values["sum_count"]
        ),
        "cot_match_index_accuracy": _safe_ratio(
            scalar_values["match_exact"], scalar_values["match_count"]
        ),
        "cot_result_semantic_accuracy": _safe_ratio(
            scalar_values["result_semantic"], result_count
        ),
        "cot_result_nll": _safe_ratio(
            scalar_values["result_nll_sum"], result_count
        ),
        "cot_result_nll_floor": _safe_ratio(
            scalar_values["result_nll_floor_sum"], result_count
        ),
        "cot_chance_baselines": {
            "pair_position_token_accuracy": _safe_ratio(
                scalar_values["pair_exact_chance_sum"], scalar_values["pair_count"]
            ),
            "pair_position_semantic_accuracy": _safe_ratio(
                scalar_values["pair_semantic_chance_sum"], scalar_values["pair_count"]
            ),
            "sum_token_accuracy": _safe_ratio(
                scalar_values["sum_exact_chance_sum"], scalar_values["sum_count"]
            ),
            "sum_semantic_accuracy": _safe_ratio(
                scalar_values["sum_semantic_chance_sum"], scalar_values["sum_count"]
            ),
            "match_index_accuracy": _safe_ratio(
                scalar_values["match_exact_chance_sum"], scalar_values["match_count"]
            ),
            "result_semantic_accuracy": _safe_ratio(
                scalar_values["result_semantic_chance_sum"], result_count
            ),
        },
        "cot_per_pair": per_pair_report,
        "cot_diagnostic_counts": {
            "pair_position_tokens": int(scalar_values["pair_count"]),
            "sum_results": int(scalar_values["sum_count"]),
            "match_results": int(scalar_values["match_count"]),
            "all_results": int(result_count),
            "answers": int(scalar_values["answer_count"]),
        },
    }
