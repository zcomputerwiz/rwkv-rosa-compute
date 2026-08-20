"""Tests for Experiment 0 task generation, formatting, and datasets."""

import random
from math import comb

import pytest
import torch

from exp0.dataset import Task3SumDataset, build_default_vocab
from exp0.sequences import (
    format_a_parallel_cot,
    format_b_filler,
    format_c_immediate,
    format_d_serial_cot,
    format_e_neutral,
    format_inputs,
)
from exp0.task3sum import (
    LEGACY_GENERATOR,
    Instance3Sum,
    check_3sum,
    generate_instance,
)


@pytest.mark.exp0
def test_3sum_check_and_generation():
    rng = random.Random(123)
    pos_inst = generate_instance(
        length=8,
        dimension=3,
        target_has_3sum=True,
        rng=rng,
    )
    assert pos_inst.has_3sum is True
    has_sol, indices = check_3sum(pos_inst.tuples)
    assert has_sol is True
    assert indices == pos_inst.matching_indices

    # source_corrupted=False selects the corrupted construction arm and may
    # legitimately remain positive. Use the explicitly conditioned legacy
    # generator when a unit test requires a guaranteed negative instance.
    neg_inst = generate_instance(
        length=8,
        dimension=3,
        target_has_3sum=False,
        rng=rng,
        generator_mode=LEGACY_GENERATOR,
    )
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
    instances = [
        generate_instance(length=6, dimension=3, rng=rng) for _ in range(100)
    ]
    vocab = build_default_vocab(length=6, dimension=3)

    dataset = Task3SumDataset(
        instances=instances,
        num_filler=36,
        vocab=vocab,
        seed=42,
        parallel_ratio=0.5,
        filler_ratio=0.5,
    )

    counts = dataset.realized_counts
    assert counts["parallel_cot"] + counts["filler"] == 100
    assert 40 <= counts["parallel_cot"] <= 60

    initial_vocab_len = len(vocab)
    _ = dataset[0]
    _ = dataset[1]
    assert len(vocab) == initial_vocab_len


@pytest.mark.exp0
def test_parallel_cot_length_invariant():
    """Parallel CoT must have one class-independent whitespace-token length."""
    for vocab_red in [True, False]:
        for length, dimension in [(8, 2), (12, 3), (16, 4)]:
            pos_lengths = set()
            neg_lengths = set()
            generation_rng = random.Random(42)

            for item_idx in range(100):
                pos_inst = generate_instance(
                    length=length,
                    dimension=dimension,
                    target_has_3sum=True,
                    rng=generation_rng,
                )
                neg_inst = generate_instance(
                    length=length,
                    dimension=dimension,
                    target_has_3sum=False,
                    rng=generation_rng,
                )

                pos_fmt = format_a_parallel_cot(
                    pos_inst,
                    vocab_reduction=vocab_red,
                    rng=random.Random(f"pos_{vocab_red}_{length}_{item_idx}"),
                )
                neg_fmt = format_a_parallel_cot(
                    neg_inst,
                    vocab_reduction=vocab_red,
                    rng=random.Random(f"neg_{vocab_red}_{length}_{item_idx}"),
                )

                pos_lengths.add(len(pos_fmt.split()))
                neg_lengths.add(len(neg_fmt.split()))

            assert pos_lengths
            assert neg_lengths

            combined_lengths = pos_lengths | neg_lengths
            assert len(combined_lengths) == 1
            common_len = next(iter(combined_lengths))

            expected_len = length + 1 + 2 * comb(length, 2) + 2
            assert common_len == expected_len

            if length == 12 and dimension == 3:
                assert common_len == 147


@pytest.mark.exp0
def test_serial_cot_length_distributions():
    """Positive and corrupted-arm serial CoT examples share observed lengths."""
    rng = random.Random(42)
    for length, dimension in [(8, 2), (12, 3)]:
        pos_lengths = []
        corrupted_lengths = []
        for _ in range(200):
            pos_inst = generate_instance(
                length=length,
                dimension=dimension,
                target_has_3sum=True,
                rng=rng,
            )
            corrupted_inst = generate_instance(
                length=length,
                dimension=dimension,
                target_has_3sum=False,
                rng=rng,
            )

            pos_lengths.append(len(format_d_serial_cot(pos_inst).split()))
            corrupted_lengths.append(len(format_d_serial_cot(corrupted_inst).split()))

        assert pos_lengths
        assert corrupted_lengths
        assert set(pos_lengths) & set(corrupted_lengths)


@pytest.mark.exp0
def test_dataset_determinism():
    """Per-item formatting must be deterministic and access-order independent."""
    source_rng = random.Random(1234)
    instances = [
        generate_instance(length=8, dimension=2, rng=source_rng) for _ in range(20)
    ]
    vocab = build_default_vocab(length=8, dimension=2)

    ds1 = Task3SumDataset(
        instances=instances,
        format_type="parallel_cot",
        vocab=vocab,
        seed=42,
    )
    ds2 = Task3SumDataset(
        instances=instances,
        format_type="parallel_cot",
        vocab=vocab,
        seed=42,
    )
    ds3 = Task3SumDataset(
        instances=instances,
        format_type="parallel_cot",
        vocab=vocab,
        seed=99,
    )

    out_0_read1 = ds1[0]["targets"].clone()
    out_0_read2 = ds1[0]["targets"].clone()
    assert torch.equal(out_0_read1, out_0_read2)

    _ = ds2[5]
    _ = ds2[1]
    out_0_ds2 = ds2[0]["targets"].clone()
    assert torch.equal(out_0_read1, out_0_ds2)

    assert any(
        not torch.equal(ds1[i]["targets"], ds3[i]["targets"])
        for i in range(len(instances))
    )
