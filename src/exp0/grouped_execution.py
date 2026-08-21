"""Length-aware execution of one Experiment 0 optimizer batch.

The mixed 50/50 CoT/filler batch is padded to its longest member. Parallel CoT
is a fixed length regardless of N, so at low N every filler example is carried
through a rectangle sized for CoT and the fused recurrence rounds time up again
to CHUNK_LEN. At N=0 barely half the executed recurrent transitions are logical.

This module runs the *same* optimizer batch as several length-homogeneous
subgroups. It changes execution only:

    one AdamW update per original batch
    one LR scheduler step per original batch
    one global gradient clip over the complete accumulated gradient
    the same token-weighted global loss

The loss must be weighted by tokens, not by group. Averaging subgroup means
would silently reweight the objective, because CoT and filler groups contain
different numbers of supervised positions:

    correct   (cot_sum + filler_sum) / (cot_tokens + filler_tokens)
    wrong     (cot_mean + filler_mean) / 2

Grouping is safe because nothing in these models mixes information across
examples: there is no BatchNorm, no dropout, and the RWKV GroupNorm is applied
over flattened (sample, token) rows so each row normalizes independently.
Trimming trailing padding is safe because both backbones are causal, so a
position cannot be influenced by anything after it.

Results are mathematically equivalent but not bitwise identical to the padded
path: summation order differs, which moves valid-position logits by float32
epsilon (order 1e-07).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def sequence_lengths(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Unpadded target length per sample.

    Recovered from ``loss_mask`` rather than carried alongside it: the mask
    holds real target ids in ``[:seq_len]`` and -100 afterwards, and target ids
    are vocabulary indices which are never negative, so the count is exact.
    """
    return (batch["loss_mask"] != IGNORE_INDEX).sum(dim=1)


def group_by_length(batch: Dict[str, torch.Tensor]) -> List[Tuple[int, torch.Tensor]]:
    """Split batch indices into length-homogeneous groups, longest first.

    Exact length rather than format: it generalizes to future formats and gives
    identical results where formats already share a length. Longest first so the
    largest allocation happens while the allocator is least fragmented.
    """
    lengths = sequence_lengths(batch)
    groups: List[Tuple[int, torch.Tensor]] = []
    for length in sorted({int(v) for v in lengths.tolist()}, reverse=True):
        index = torch.nonzero(lengths == length, as_tuple=False).flatten()
        groups.append((length, index))
    return groups


def supervised_token_count(batch: Dict[str, torch.Tensor]) -> int:
    """Global CE denominator: supervised positions after the next-token shift."""
    return int((batch["loss_mask"][:, 1:] != IGNORE_INDEX).sum())


def grouped_loss_backward(
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    autocast: Optional[Callable[[], Any]] = None,
    backward: bool = True,
    on_group: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]] = None,
) -> Dict[str, Any]:
    """Run one optimizer batch as length groups, accumulating gradients.

    ``forward_fn`` takes ``(input_tuples, targets)`` and returns loss-position
    logits, matching ``InputEmbedWrapper.loss_logits``. Nothing here steps an
    optimizer or scheduler, and nothing clips: the caller does that once, over
    the complete accumulated gradient, exactly as the unrouped path does.

    ``on_group`` is called as ``on_group(logits, index, sub_targets)`` once per
    group, after the backward for that group. The padded path derives per-format
    training accuracy from the batch logits; grouping never materializes a full
    batch of logits, so that telemetry has to be accumulated per group instead of
    dropped. ``index`` maps rows back to positions in the original batch so the
    caller can align its own per-sample tensors.

    Returns the global token-weighted loss plus per-group execution counts, so
    the padding actually avoided is reportable rather than assumed. The counts
    include the head-projection positions that grouping still leaves
    unsupervised, which is what decides whether a masked head is worth building.
    """
    denominator = supervised_token_count(batch)
    if denominator == 0:
        raise ValueError("batch contains no supervised positions")

    total_loss = 0.0
    executed_positions = 0
    head_positions = 0
    supervised_head_positions = 0
    group_records: List[Dict[str, int]] = []

    for length, index in group_by_length(batch):
        if length < 2:
            # A single target position yields no next-token prediction.
            continue
        sub_tuples = batch["input_tuples"][index].to(device, non_blocking=True)
        sub_targets = batch["targets"][index, :length].to(device, non_blocking=True)
        sub_mask = batch["loss_mask"][index, :length].to(device, non_blocking=True)

        context = autocast() if autocast is not None else _null_context()
        with context:
            logits = forward_fn(sub_tuples, sub_targets)
            # reduction="sum" then a single global divide: this is what keeps
            # the objective token-weighted instead of group-weighted.
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                sub_mask[:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        scaled = loss_sum / denominator
        if backward:
            scaled.backward()

        if on_group is not None:
            on_group(logits.detach(), index, sub_targets)

        group_supervised = int((sub_mask[:, 1:] != IGNORE_INDEX).sum())
        # The head projects every shifted position in the group, supervised or
        # not. Grouping removes cross-format padding but not the positions a
        # sequence pads to its own end, so this is the waste a masked head would
        # still be able to recover.
        group_head_positions = int(index.numel()) * (length - 1)

        total_loss += float(loss_sum.detach())
        executed_positions += int(index.numel()) * length
        head_positions += group_head_positions
        supervised_head_positions += group_supervised
        group_records.append({
            "target_length": length,
            "samples": int(index.numel()),
            "supervised_tokens": group_supervised,
            "head_positions": group_head_positions,
        })

    padded_positions = int(batch["targets"].shape[0] * batch["targets"].shape[1])
    return {
        "loss": total_loss / denominator,
        "supervised_tokens": denominator,
        "groups": group_records,
        "executed_target_positions": executed_positions,
        "padded_target_positions": padded_positions,
        "positions_saved": padded_positions - executed_positions,
        "head_positions": head_positions,
        "supervised_head_positions": supervised_head_positions,
        "unsupervised_head_positions": head_positions - supervised_head_positions,
    }


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
