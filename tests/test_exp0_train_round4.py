import random

import pytest
import torch
import torch.nn as nn

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset
from exp0.task3sum import generate_instance
from exp0.train import evaluate_accuracy, train_model

pytestmark = pytest.mark.exp0


def get_tiny_configs():
    task_cfg = Task3SumConfig(num_samples=16)
    model_cfg = ModelConfig(
        architecture="llama",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        device="cpu",
    )
    train_cfg = TrainConfig(batch_size=8, epochs=1, num_workers=0)
    return task_cfg, model_cfg, train_cfg


def _generate_instances(task_cfg: Task3SumConfig, seed: int):
    rng = random.Random(seed)
    return [
        generate_instance(
            length=task_cfg.length,
            dimension=task_cfg.dimension,
            mod=task_cfg.mod,
            rng=rng,
        )
        for _ in range(task_cfg.num_samples)
    ]


def test_train_model_dual_validation():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()

    train_instances = _generate_instances(task_cfg, seed=100)
    val_instances = _generate_instances(task_cfg, seed=101)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    cot_ds = Task3SumDataset(
        val_instances,
        format_type="parallel_cot",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    _, history = train_model(
        model_cfg,
        train_cfg,
        task_cfg,
        train_ds,
        val_ds,
        cot_val_dataset=cot_ds,
    )

    assert len(history["epoch_train_losses"]) == train_cfg.epochs
    assert len(history["epoch_online_train_answer_accuracies"]) == train_cfg.epochs
    assert len(history["epoch_filler_accuracies"]) == train_cfg.epochs
    assert len(history["epoch_cot_diagnostics"]) == train_cfg.epochs

    assert "best_online_train_answer_accuracy" in history
    assert "best_filler_accuracy" in history
    assert "best_cot_diagnostics" in history
    assert "cot_answer_given_cot_accuracy" in history["best_cot_diagnostics"]
    assert "cot_result_semantic_accuracy" in history["best_cot_diagnostics"]
    assert "cot_result_nll" in history["best_cot_diagnostics"]
    assert history["adam_betas"] == [0.9, 0.95]
    assert history["lr_schedule"] == "linear_warmup_decay"
    assert "best_cot_accuracy" not in history
    assert history["best_val_accuracy"] == history["best_filler_accuracy"]


def test_train_model_single_validation():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()

    train_instances = _generate_instances(task_cfg, seed=200)
    val_instances = _generate_instances(task_cfg, seed=201)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    assert "epoch_train_losses" in history
    assert "epoch_online_train_answer_accuracies" in history
    assert "epoch_filler_accuracies" in history
    assert "epoch_cot_diagnostics" not in history
    assert "best_cot_diagnostics" not in history


def test_missing_ids_fails():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    train_instances = _generate_instances(task_cfg, seed=300)
    val_instances = _generate_instances(task_cfg, seed=301)
    train_ds = Task3SumDataset(
        train_instances,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )

    if "ANS" in train_ds.vocab.token2id:
        del train_ds.vocab.token2id["ANS"]

    with pytest.raises(
        ValueError,
        match="Vocabulary must contain 'ANS', 'True', and 'False'",
    ):
        train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)


class MockModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.call_count = 0

    def forward(self, input_tuples, targets):
        batch_size = targets.size(0)
        out = self.logits[self.call_count : self.call_count + batch_size]
        self.call_count += batch_size
        return out


def test_evaluate_accuracy_exact_scoring():
    ans_token_id = 10
    ans_true_id = 1
    ans_false_id = 2
    unrelated_id = 3
    device = torch.device("cpu")

    targets = torch.tensor(
        [
            [0, 0, ans_token_id, ans_true_id, 0],
            [0, 0, ans_token_id, ans_false_id, 0],
            [0, 0, ans_token_id, ans_false_id, 0],
            [0, 0, ans_token_id, ans_true_id, 0],
        ]
    )
    has_3sum = torch.tensor([True, False, False, True])

    logits = torch.zeros(4, 5, 20)
    logits[0, 2, ans_true_id] = 10.0
    logits[1, 2, ans_false_id] = 10.0
    logits[2, 2, unrelated_id] = 10.0
    logits[3, 2, ans_false_id] = 10.0

    mock_model = MockModel(logits)
    mock_loader = [
        {
            "input_tuples": torch.zeros(4, 1, 1),
            "targets": targets,
            "has_3sum": has_3sum,
        }
    ]

    accuracy = evaluate_accuracy(
        mock_model,
        mock_loader,
        device,
        ans_token_id,
        ans_true_id,
        ans_false_id,
    )

    assert accuracy == 0.5


def test_evaluate_accuracy_malformed_ans():
    ans_token_id = 10
    ans_true_id = 1
    ans_false_id = 2
    device = torch.device("cpu")

    mock_model = MockModel(torch.zeros(1, 5, 20))

    targets_zero = torch.tensor([[0, 0, 0, 0, 0]])
    mock_loader_zero = [
        {
            "input_tuples": torch.zeros(1, 1, 1),
            "targets": targets_zero,
            "has_3sum": torch.tensor([True]),
        }
    ]

    with pytest.raises(ValueError, match="Sequence must have exactly one ANS token"):
        evaluate_accuracy(
            mock_model,
            mock_loader_zero,
            device,
            ans_token_id,
            ans_true_id,
            ans_false_id,
        )

    mock_model.call_count = 0
    targets_multiple = torch.tensor([[0, ans_token_id, ans_token_id, 0, 0]])
    mock_loader_multiple = [
        {
            "input_tuples": torch.zeros(1, 1, 1),
            "targets": targets_multiple,
            "has_3sum": torch.tensor([True]),
        }
    ]

    with pytest.raises(ValueError, match="Sequence must have exactly one ANS token"):
        evaluate_accuracy(
            mock_model,
            mock_loader_multiple,
            device,
            ans_token_id,
            ans_true_id,
            ans_false_id,
        )
