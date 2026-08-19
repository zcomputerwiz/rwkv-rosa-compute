"""Regression gates for the Experiment 0 transformer positive control."""

import random

import pytest
import torch

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import (
    COT_DIAG_MATCH_RESULT,
    COT_DIAG_PAIR_POSITION,
    COT_DIAG_SUM_RESULT,
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.diagnostics import evaluate_cot_diagnostics
from exp0.models.llama import RotaryEmbedding
from exp0.task3sum import generate_instance
from exp0.train import train_model
from scripts.run_experiment import build_configs, get_parser

pytestmark = pytest.mark.exp0


def test_rope_matches_llama_split_half_rotation_and_is_position_sensitive():
    rotary = RotaryEmbedding(head_dim=4, base=10000.0)
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    q = base.view(1, 1, 1, 4).expand(1, 1, 2, 4).clone()
    k = q.clone()

    q_rot, k_rot = rotary(q, k)

    torch.testing.assert_close(q_rot, k_rot)
    torch.testing.assert_close(q_rot[0, 0, 0], base)

    angle0 = torch.tensor(1.0)
    angle1 = torch.tensor(0.01)
    expected_pos1 = torch.stack(
        [
            1.0 * torch.cos(angle0) - 3.0 * torch.sin(angle0),
            2.0 * torch.cos(angle1) - 4.0 * torch.sin(angle1),
            3.0 * torch.cos(angle0) + 1.0 * torch.sin(angle0),
            4.0 * torch.cos(angle1) + 2.0 * torch.sin(angle1),
        ]
    )
    torch.testing.assert_close(q_rot[0, 0, 1], expected_pos1)
    assert not torch.equal(q_rot[0, 0, 0], q_rot[0, 0, 1])


def test_runner_retains_paper_n_squared_filler_budget():
    parser = get_parser()
    args = parser.parse_args(
        [
            "--architecture",
            "llama",
            "--length",
            "6",
            "--dimension",
            "3",
            "--device",
            "cpu",
        ]
    )
    task_cfg, model_cfg, _ = build_configs(args)

    assert task_cfg.num_filler == 36
    assert task_cfg.include_separator_token is True
    assert model_cfg.llama_rope_theta == 10000.0


def test_separator_dropping_protocol_is_rejected():
    with pytest.raises(ValueError, match="requires the supervised continuation separator"):
        Task3SumConfig(include_separator_token=False)


def test_dataset_keeps_separator_and_marks_cot_semantics():
    rng = random.Random(77)
    positive = generate_instance(
        length=5,
        dimension=3,
        target_has_3sum=True,
        rng=rng,
    )
    negative = generate_instance(
        length=5,
        dimension=3,
        target_has_3sum=False,
        rng=rng,
    )
    vocab = build_default_vocab(length=5, dimension=3)
    dataset = Task3SumDataset(
        [positive, negative],
        format_type="parallel_cot",
        vocab=vocab,
        seed=12,
        vocab_reduction=True,
    )

    items = [dataset[0], dataset[1]]
    for item in items:
        targets = item["targets"]
        diag_type = item["cot_diag_type"]
        valid_ids = item["cot_valid_ids"]

        assert targets[0].item() == vocab.token2id[":"]
        assert int((diag_type == COT_DIAG_PAIR_POSITION).sum()) == 10
        assert int(
            (
                (diag_type == COT_DIAG_SUM_RESULT)
                | (diag_type == COT_DIAG_MATCH_RESULT)
            ).sum()
        ) == 10

        diagnostic_positions = diag_type.ne(0).nonzero(as_tuple=True)[0]
        for position in diagnostic_positions:
            actual = targets[position]
            valid = valid_ids[position]
            assert bool((valid == actual).any())

    assert bool((items[0]["cot_diag_type"] == COT_DIAG_MATCH_RESULT).any())
    assert not bool((items[1]["cot_diag_type"] == COT_DIAG_MATCH_RESULT).any())


def test_cot_diagnostics_separate_answer_leakage_from_generation():
    rng = random.Random(123)
    instances = [
        generate_instance(length=5, dimension=2, rng=rng) for _ in range(8)
    ]
    vocab = build_default_vocab(length=5, dimension=2)
    dataset = Task3SumDataset(
        instances,
        format_type="parallel_cot",
        vocab=vocab,
        seed=55,
    )
    batch = pad_collate_fn([dataset[idx] for idx in range(len(dataset))])

    targets = batch["targets"]
    diag_type = batch["cot_diag_type"]
    batch_size, target_len = targets.shape
    vocab_size = len(vocab)
    scripted_logits = torch.full((batch_size, target_len - 1, vocab_size), -10.0)

    wrong_id = vocab.token2id[":"]
    scripted_logits[..., wrong_id] = 10.0

    next_types = diag_type[:, 1:]
    pair_mask = next_types.eq(COT_DIAG_PAIR_POSITION)
    next_targets = targets[:, 1:]
    pair_rows, pair_cols = pair_mask.nonzero(as_tuple=True)
    scripted_logits[pair_rows, pair_cols, next_targets[pair_mask]] = 20.0

    ans_id = vocab.token2id["ANS"]
    true_id = vocab.token2id["True"]
    false_id = vocab.token2id["False"]
    ans_positions = targets[:, :-1].eq(ans_id).to(torch.int64).argmax(dim=1)
    for row, answer_position in enumerate(ans_positions.tolist()):
        expected = true_id if instances[row].has_3sum else false_id
        scripted_logits[row, answer_position, expected] = 30.0

    class ScriptedModel(torch.nn.Module):
        def loss_logits(self, input_tuples, target_ids):
            return scripted_logits

    diagnostics = evaluate_cot_diagnostics(
        ScriptedModel(),
        [batch],
        torch.device("cpu"),
        ans_id,
        true_id,
        false_id,
    )

    assert diagnostics["cot_answer_given_cot_accuracy"] == 1.0
    assert diagnostics["cot_pair_position_token_accuracy"] == 1.0
    assert diagnostics["cot_result_semantic_accuracy"] == 0.0
    assert diagnostics["cot_match_index_accuracy"] == 0.0


def test_tiny_immediate_dataset_can_be_overfit_on_cpu():
    """A tiny fixed set must be fit before larger 3SUM failures are interpretable."""
    task_cfg = Task3SumConfig(
        length=3,
        dimension=1,
        num_filler=0,
        num_samples=16,
    )
    rng = random.Random(2024)
    instances = [
        generate_instance(
            length=3,
            dimension=1,
            rng=rng,
        )
        for _ in range(task_cfg.num_samples)
    ]
    vocab = build_default_vocab(length=3, dimension=1)
    dataset = Task3SumDataset(
        instances,
        format_type="immediate",
        num_filler=0,
        vocab=vocab,
        seed=2024,
    )
    model_cfg = ModelConfig(
        architecture="llama",
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        device="cpu",
    )
    train_cfg = TrainConfig(
        seed=2024,
        batch_size=16,
        learning_rate=3e-3,
        epochs=20,
        num_workers=0,
        mixture="immediate",
    )

    _, history = train_model(
        model_cfg,
        train_cfg,
        task_cfg,
        dataset,
        dataset,
        cot_val_dataset=None,
    )

    assert history["best_train_answer_accuracy"] >= 0.95
    assert history["best_filler_accuracy"] >= 0.95
