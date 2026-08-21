"""Parity ladder for length-aware grouped execution (CPU).

Gate 1 same loss, Gate 2 same gradients, Gate 3 same optimizer update. The
grouped path is mathematically equivalent, not bitwise identical: summation
order differs, so comparisons use float32-epsilon tolerances and the observed
deviations are asserted to stay well inside them.
"""

import random

import pytest
import torch
import torch.nn.functional as F

from exp0.config import ModelConfig, Task3SumConfig
from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
from exp0.generation import generate_protocol_packed_instances
from exp0.grouped_execution import (
    group_by_length,
    grouped_loss_backward,
    sequence_lengths,
    supervised_token_count,
)
from exp0.train import create_model

pytestmark = pytest.mark.exp0

DEVICE = torch.device("cpu")
ATOL = 1e-5


def _batch(num_filler=0, count=8, seed=1):
    vocab = build_default_vocab(length=6, dimension=3)
    packed = generate_protocol_packed_instances(
        count, length=6, dimension=3, rng=random.Random(seed))
    dataset = Task3SumDataset(packed, num_filler=num_filler, vocab=vocab,
                              parallel_ratio=0.5, filler_ratio=0.5, seed=seed)
    return vocab, pad_collate_fn([dataset[i] for i in range(count)])


def _model(vocab, arch="llama"):
    torch.manual_seed(0)
    extra = {"num_attention_heads": 2} if arch == "llama" else {"head_dim": 64}
    return create_model(
        ModelConfig(architecture=arch, hidden_size=64, num_hidden_layers=2,
                    intermediate_size=256, device="cpu", vocab_size=len(vocab),
                    **extra),
        d_input=36, vocab=vocab,
        task_cfg=Task3SumConfig(length=6, dimension=3, num_filler=0))


def _baseline_loss(model, batch):
    """The current padded path: one rectangle, CE mean over supervised tokens."""
    logits = model.loss_logits(batch["input_tuples"], batch["targets"])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                           batch["loss_mask"][:, 1:].reshape(-1),
                           ignore_index=-100)


# --- structure ---------------------------------------------------------------

def test_lengths_are_recovered_exactly_from_the_mask():
    vocab, batch = _batch()
    lengths = sequence_lengths(batch)
    for row, length in enumerate(lengths.tolist()):
        assert int((batch["loss_mask"][row] != -100).sum()) == length
        assert batch["loss_mask"][row, length:].eq(-100).all()


def test_groups_partition_the_batch_without_loss():
    vocab, batch = _batch()
    groups = group_by_length(batch)
    covered = torch.cat([index for _, index in groups]).sort().values
    assert torch.equal(covered, torch.arange(batch["targets"].shape[0]))
    assert len(groups) >= 2, "the mixed batch should contain distinct lengths"


def test_mixed_batch_at_n0_splits_into_filler_and_cot_lengths():
    vocab, batch = _batch(num_filler=0)
    lengths = sorted({length for length, _ in group_by_length(batch)})
    assert lengths == [4, 34], "N=0 filler is T=4 targets, parallel CoT is 34"


# --- Gate 1: same loss -------------------------------------------------------

@pytest.mark.parametrize("arch", ["llama", "rwkv"])
def test_gate1_grouped_loss_matches_the_padded_path(arch):
    vocab, batch = _batch()
    model = _model(vocab, arch).eval()
    with torch.no_grad():
        expected = float(_baseline_loss(model, batch))
    result = grouped_loss_backward(model.loss_logits, batch, DEVICE, backward=False)
    assert result["loss"] == pytest.approx(expected, abs=ATOL)


def test_gate1_denominator_is_token_weighted_not_group_weighted():
    """Averaging group means would reweight the objective; assert it differs."""
    vocab, batch = _batch()
    model = _model(vocab).eval()
    result = grouped_loss_backward(model.loss_logits, batch, DEVICE, backward=False)

    group_means = []
    for length, index in group_by_length(batch):
        if length < 2:
            continue
        with torch.no_grad():
            logits = model.loss_logits(batch["input_tuples"][index],
                                       batch["targets"][index, :length])
        group_means.append(float(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            batch["loss_mask"][index, :length][:, 1:].reshape(-1),
            ignore_index=-100)))
    naive = sum(group_means) / len(group_means)
    assert result["loss"] != pytest.approx(naive, abs=1e-4), (
        "token-weighted and group-weighted losses must differ on an unbalanced "
        "batch, otherwise this test cannot detect the mistake"
    )
    assert result["supervised_tokens"] == supervised_token_count(batch)


# --- Gate 2: same gradients --------------------------------------------------

@pytest.mark.parametrize("arch", ["llama", "rwkv"])
def test_gate2_gradients_match_the_padded_path(arch):
    vocab, batch = _batch()

    baseline = _model(vocab, arch)
    _baseline_loss(baseline, batch).backward()
    reference = {n: p.grad.clone() for n, p in baseline.named_parameters()
                 if p.grad is not None}

    grouped = _model(vocab, arch)
    grouped_loss_backward(grouped.loss_logits, batch, DEVICE)
    actual = {n: p.grad for n, p in grouped.named_parameters() if p.grad is not None}

    assert set(actual) == set(reference)
    worst = max(float((actual[n] - reference[n]).abs().max()) for n in reference)
    assert worst < ATOL, f"largest gradient deviation {worst:.2e}"


def test_gate2_global_gradient_norm_matches():
    """Clipping happens once over the whole accumulated gradient, so the norm
    it sees must match the padded path."""
    vocab, batch = _batch()

    baseline = _model(vocab)
    _baseline_loss(baseline, batch).backward()
    expected = float(torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1e9))

    grouped = _model(vocab)
    grouped_loss_backward(grouped.loss_logits, batch, DEVICE)
    actual = float(torch.nn.utils.clip_grad_norm_(grouped.parameters(), 1e9))
    assert actual == pytest.approx(expected, rel=1e-4)


# --- Gate 3: same optimizer update -------------------------------------------

def test_gate3_one_optimizer_step_produces_the_same_parameters():
    vocab, batch = _batch()

    def step(model, use_groups):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                      betas=(0.9, 0.95), weight_decay=0.01)
        optimizer.zero_grad(set_to_none=True)
        if use_groups:
            grouped_loss_backward(model.loss_logits, batch, DEVICE)
        else:
            _baseline_loss(model, batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return optimizer

    baseline, grouped = _model(vocab), _model(vocab)
    baseline_opt = step(baseline, use_groups=False)
    grouped_opt = step(grouped, use_groups=True)

    worst = max(float((a - b).abs().max().detach()) for a, b in
                zip(baseline.parameters(), grouped.parameters()))
    assert worst < ATOL, f"largest parameter deviation after one step {worst:.2e}"

    for base_group, group_group in zip(baseline_opt.param_groups,
                                       grouped_opt.param_groups):
        assert base_group["lr"] == group_group["lr"]
    base_state = list(baseline_opt.state.values())[0]
    grouped_state = list(grouped_opt.state.values())[0]
    assert base_state["step"] == grouped_state["step"] == 1, "exactly one update"
    assert torch.allclose(base_state["exp_avg"], grouped_state["exp_avg"], atol=ATOL)
    assert torch.allclose(base_state["exp_avg_sq"], grouped_state["exp_avg_sq"],
                          atol=ATOL)


# --- reported savings --------------------------------------------------------

def test_reported_savings_reflect_the_padding_actually_avoided():
    vocab, batch = _batch(num_filler=0)
    model = _model(vocab).eval()
    result = grouped_loss_backward(model.loss_logits, batch, DEVICE, backward=False)
    assert result["padded_target_positions"] > result["executed_target_positions"]
    assert result["positions_saved"] > 0
    assert sum(g["samples"] for g in result["groups"]) == batch["targets"].shape[0]
