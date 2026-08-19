"""CPU equivalence and compact-storage tests for Experiment 0 optimizations."""

import random

import torch
import torch.nn as nn

from exp0.config import ModelConfig, TrainConfig
from exp0.dataset import (
    Task3SumDataset,
    build_default_vocab,
    generate_packed_instances,
)
from exp0.task3sum import check_3sum, generate_instance
from exp0.train import _create_loader, create_model, evaluate_accuracy


def _slow_check_3sum(tuples, mod=10):
    n = len(tuples)
    d = len(tuples[0])
    for i in range(n):
        for j in range(i + 1, n):
            target = tuple(
                (-tuples[i][dim] - tuples[j][dim]) % mod
                for dim in range(d)
            )
            for k in range(n):
                if k != i and k != j and tuples[k] == target:
                    return True, tuple(sorted((i, j, k)))
    return False, None


def test_fast_3sum_matches_legacy_ordering():
    rng = random.Random(9917)
    for _ in range(250):
        tuples = [
            tuple(rng.randrange(10) for _ in range(3))
            for _ in range(12)
        ]
        assert check_3sum(tuples) == _slow_check_3sum(tuples)


def test_packed_generation_matches_legacy_rng_stream():
    count = 32
    legacy_rng = random.Random(12345)
    legacy = [
        generate_instance(length=8, dimension=3, rng=legacy_rng)
        for _ in range(count)
    ]
    packed = generate_packed_instances(
        count,
        length=8,
        dimension=3,
        rng=random.Random(12345),
    )

    for idx, expected in enumerate(legacy):
        actual = packed.instance_at(idx)
        assert actual == expected


def test_packed_default_storage_is_43_bytes_per_instance():
    count = 100
    packed = generate_packed_instances(
        count,
        length=12,
        dimension=3,
        rng=random.Random(7),
    )

    # 12*3 uint8 tuple values + 1 bool label + 3 int16 match indices.
    assert packed.storage_nbytes == count * 43

    dataset = Task3SumDataset(
        packed,
        format_type="filler",
        num_filler=144,
        seed=7,
    )
    # One additional uint8 format code per sample.
    assert dataset.packed_storage_nbytes == count * 44


def test_packed_and_list_backed_dataset_outputs_match():
    count = 16
    legacy_rng = random.Random(81)
    legacy = [
        generate_instance(length=8, dimension=2, rng=legacy_rng)
        for _ in range(count)
    ]
    packed = generate_packed_instances(
        count,
        length=8,
        dimension=2,
        rng=random.Random(81),
    )
    vocab = build_default_vocab(length=8, dimension=2)

    list_dataset = Task3SumDataset(
        legacy,
        format_type="parallel_cot",
        vocab=vocab,
        seed=123,
    )
    packed_dataset = Task3SumDataset(
        packed,
        format_type="parallel_cot",
        vocab=vocab,
        seed=123,
    )

    for idx in range(count):
        expected = list_dataset[idx]
        actual = packed_dataset[idx]
        assert expected["format"] == actual["format"]
        assert torch.equal(expected["input_tuples"], actual["input_tuples"])
        assert torch.equal(expected["targets"], actual["targets"])
        assert torch.equal(expected["has_3sum"], actual["has_3sum"])


def test_loss_and_answer_projection_match_full_logits():
    torch.manual_seed(9)
    model = create_model(
        ModelConfig(
            architecture="llama",
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            vocab_size=31,
            device="cpu",
        ),
        d_input=10,
    )
    model.eval()

    input_tuples = torch.randn(3, 4, 10)
    targets = torch.randint(0, 31, (3, 7))
    answer_positions = torch.tensor([0, 3, 5])

    with torch.no_grad():
        full_logits = model(input_tuples, targets)
        loss_logits = model.loss_logits(input_tuples, targets)
        answer_logits = model.answer_logits(
            input_tuples,
            targets,
            answer_positions,
        )

    assert torch.equal(loss_logits, full_logits[:, :-1, :])
    expected_answers = full_logits[
        torch.arange(full_logits.shape[0]),
        answer_positions,
    ]
    # The answer-only projection uses a smaller GEMM than the full-vocabulary
    # projection, so different BLAS kernels may differ by a few FP32 ulps.
    torch.testing.assert_close(
        answer_logits,
        expected_answers,
        rtol=1e-6,
        atol=1e-7,
    )
    assert torch.equal(
        answer_logits.argmax(dim=-1),
        expected_answers.argmax(dim=-1),
    )


class _AnswerOnlyModel(nn.Module):
    def answer_logits(self, input_tuples, targets, answer_positions):
        del input_tuples, answer_positions
        logits = torch.zeros(targets.shape[0], 8)
        logits[:, 1] = 10.0
        return logits

    def forward(self, input_tuples, targets):
        del input_tuples, targets
        raise AssertionError("full logits path should not run")


def test_evaluation_uses_answer_only_projection_when_available():
    loader = [
        {
            "input_tuples": torch.zeros(2, 1, 1),
            "targets": torch.tensor(
                [
                    [0, 5, 1, 0],
                    [0, 5, 2, 0],
                ]
            ),
            "has_3sum": torch.tensor([True, False]),
        }
    ]

    accuracy = evaluate_accuracy(
        _AnswerOnlyModel(),
        loader,
        torch.device("cpu"),
        ans_token_id=5,
        ans_true_id=1,
        ans_false_id=2,
    )
    assert accuracy == 0.5


def test_validation_loader_defaults_to_no_workers():
    packed = generate_packed_instances(
        4,
        length=6,
        dimension=2,
        rng=random.Random(4),
    )
    dataset = Task3SumDataset(packed, format_type="filler", num_filler=4)
    cfg = TrainConfig(num_workers=2, val_num_workers=0, batch_size=2)
    loader = _create_loader(
        dataset,
        cfg,
        torch.device("cpu"),
        num_workers=cfg.val_num_workers,
    )

    assert loader.num_workers == 0
