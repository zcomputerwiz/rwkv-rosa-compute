"""Unit tests for Milestone 1: 3SUM task generation, sequence formatting, dataset tensorization, and seeding."""

import random

import torch

from exp0.dataset import (
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.sequences import (
    format_a_parallel_cot,
    format_b_filler,
    format_c_immediate,
    format_d_serial_cot,
    format_e_neutral,
    format_inputs,
)
from exp0.task3sum import Instance3Sum, check_3sum, generate_instance


def test_3sum_check_and_generation():
    # Test positive instance generation
    rng = random.Random(123)
    pos_inst = generate_instance(length=8, dimension=3, target_has_3sum=True, rng=rng)
    assert pos_inst.has_3sum is True
    has_sol, indices = check_3sum(pos_inst.tuples)
    assert has_sol is True
    assert indices == pos_inst.matching_indices

    # Test negative instance generation
    neg_inst = generate_instance(length=8, dimension=3, target_has_3sum=False, rng=rng)
    assert neg_inst.has_3sum is False
    has_sol_neg, indices_neg = check_3sum(neg_inst.tuples)
    assert has_sol_neg is False
    assert indices_neg is None


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


def test_seeding_reproducibility():
    inst1 = generate_instance(length=10, dimension=3, rng=random.Random(42))
    inst2 = generate_instance(length=10, dimension=3, rng=random.Random(42))
    assert inst1.tuples == inst2.tuples
    assert inst1.has_3sum == inst2.has_3sum


def test_dataset_tensorization():
    rng = random.Random(42)
    instances = [generate_instance(length=6, dimension=3, rng=rng) for _ in range(5)]
    vocab = build_default_vocab(length=6, dimension=3)

    dataset = Task3SumDataset(
        instances=instances,
        format_type="filler",
        num_filler=36,
        vocab=vocab,
        seed=42,
    )
    assert len(dataset) == 5
    sample = dataset[0]

    # Input tuples shape check: n=6, d_input = 10*3 + 6 = 36
    assert sample["input_tuples"].shape == (6, 36)
    assert sample["targets"].ndim == 1
    assert sample["has_3sum"].dtype == torch.bool

    # Collate function check
    batch = [dataset[0], dataset[1]]
    collated = pad_collate_fn(batch)
    assert collated["input_tuples"].shape == (2, 6, 36)
    assert collated["targets"].shape[0] == 2
    assert collated["loss_mask"].shape == collated["targets"].shape
