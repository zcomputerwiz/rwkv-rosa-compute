"""Unit tests for Milestone 1 & T2/T4: 3SUM task generation, sequence formatting, dataset tensorization, vocabulary, and mixture sampling."""

import random

import pytest

from exp0.dataset import Task3SumDataset, build_default_vocab
from exp0.sequences import (
    format_a_parallel_cot,
    format_b_filler,
    format_c_immediate,
    format_d_serial_cot,
    format_e_neutral,
    format_inputs,
)
from exp0.task3sum import Instance3Sum, check_3sum, generate_instance


@pytest.mark.exp0
def test_3sum_check_and_generation():
    rng = random.Random(123)
    pos_inst = generate_instance(length=8, dimension=3, target_has_3sum=True, rng=rng)
    assert pos_inst.has_3sum is True
    has_sol, indices = check_3sum(pos_inst.tuples)
    assert has_sol is True
    assert indices == pos_inst.matching_indices

    neg_inst = generate_instance(length=8, dimension=3, target_has_3sum=False, rng=rng)
    assert neg_inst.has_3sum is False
    has_sol_neg, indices_neg = check_3sum(neg_inst.tuples)
    assert has_sol_neg is False
    assert indices_neg is None


@pytest.mark.exp0
def test_sequence_formats():
    inst = Instance3Sum(
        tuples=[(0, 5, 0), (7, 3, 0), (3, 2, 0), (1, 1, 1)],
        has_3sum=True,
        matching_indices=(0, 1, 2),
    )

    prefix = format_inputs(inst)
    assert prefix == "A050 B730 C320 D111 : "

    fmt_a = format_a_parallel_cot(inst, vocab_reduction=False)
    assert "ANS True" in fmt_a

    fmt_b = format_b_filler(inst, num_filler=16)
    assert ". . . . . . . . . . . . . . . ." in fmt_b
    assert fmt_b.endswith("ANS True")

    fmt_c = format_c_immediate(inst)
    assert fmt_c == prefix + "ANS True"

    fmt_d = format_d_serial_cot(inst)
    assert "DIM 0" in fmt_d
    assert fmt_d.endswith("ANS True")

    fmt_e = format_e_neutral(inst, num_filler=5, neutral_token="#")
    assert "# # # # #" in fmt_e
    assert fmt_e.endswith("ANS True")


@pytest.mark.exp0
def test_seeding_reproducibility():
    inst1 = generate_instance(length=10, dimension=3, rng=random.Random(42))
    inst2 = generate_instance(length=10, dimension=3, rng=random.Random(42))
    assert inst1.tuples == inst2.tuples
    assert inst1.has_3sum == inst2.has_3sum


@pytest.mark.exp0
def test_dataset_mixture_ratios_and_vocab_freeze():
    rng = random.Random(42)
    instances = [generate_instance(length=6, dimension=3, rng=rng) for _ in range(100)]
    vocab = build_default_vocab(length=6, dimension=3)

    dataset = Task3SumDataset(
        instances=instances,
        num_filler=36,
        vocab=vocab,
        seed=42,
        parallel_ratio=0.5,
        filler_ratio=0.5,
    )

    # Assert 50/50 mixture counts roughly equal across 100 instances
    counts = dataset.realized_counts
    assert counts["parallel_cot"] + counts["filler"] == 100
    assert 40 <= counts["parallel_cot"] <= 60

    # Test that vocabulary is frozen and does not mutate during dataset access
    initial_vocab_len = len(vocab)
    _ = dataset[0]
    _ = dataset[1]
    assert len(vocab) == initial_vocab_len
