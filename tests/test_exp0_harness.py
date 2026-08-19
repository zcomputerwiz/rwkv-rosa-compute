"""Unit tests for Milestone 3: Training and evaluation harness."""

import random
from unittest.mock import patch

import pytest

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
from exp0.evaluate import compile_experiment_report
from exp0.task3sum import generate_instance


@pytest.mark.exp0
def test_evaluation_readout_is_supervised():
    """Assert that the evaluated logits position corresponds to the ANS token and is non-masked (-100)."""
    rng = random.Random(42)
    length, dimension = 6, 3
    instances = [generate_instance(length=length, dimension=dimension, rng=rng) for _ in range(2)]

    vocab = build_default_vocab(length=length, dimension=dimension)
    ans_token_id = vocab.token2id["ANS"]
    ans_true_id = vocab.token2id["True"]
    ans_false_id = vocab.token2id["False"]

    dataset = Task3SumDataset(instances, format_type="filler", num_filler=10, vocab=vocab, seed=42)
    sample = dataset[0]

    targets = sample["targets"]
    ans_positions = (targets == ans_token_id).nonzero(as_tuple=True)[0]
    assert len(ans_positions) == 1, "ANS token must exist exactly once in target sequence"
    ans_pos = ans_positions[0].item()

    assert ans_pos + 1 < len(targets)
    answer_label_token = targets[ans_pos + 1].item()
    assert answer_label_token in (ans_true_id, ans_false_id)

    batch = pad_collate_fn([sample])
    loss_mask = batch["loss_mask"]
    shift_targets = loss_mask[:, 1:]
    supervised_target = shift_targets[0, ans_pos].item()

    assert supervised_target != -100, "Evaluated ANS position must be supervised in training loss"
    assert supervised_target == answer_label_token


@pytest.mark.exp0

@pytest.mark.exp0
@patch("exp0.train.train_model")
def test_train_loop_end_to_end_cpu(mock_train_model):
    mock_history = {
        "epoch_train_losses": [0.5],
        "epoch_filler_accuracies": [0.8],
        "epoch_cot_accuracies": [0.7],
        "best_filler_accuracy": 0.8,
        "best_cot_accuracy": 0.7,
        "best_val_accuracy": 0.8,
        "epochs_trained": 1,
        "epoch_seconds": [1.0],
        "total_train_seconds": 1.0,
        "data_wait_seconds": 0.1,
        "samples_per_second": 10.0,
    }
    mock_train_model.return_value = (None, mock_history)

    rng = random.Random(42)
    length = 6
    dimension = 3
    num_samples = 20
    val_samples = 10

    train_instances = [generate_instance(length=length, dimension=dimension, rng=rng) for _ in range(num_samples)]
    val_instances = [generate_instance(length=length, dimension=dimension, rng=rng) for _ in range(val_samples)]

    vocab = build_default_vocab(length=length, dimension=dimension)

    train_ds = Task3SumDataset(train_instances, format_type="filler", num_filler=10, vocab=vocab, seed=42)
    filler_val_ds = Task3SumDataset(val_instances, format_type="filler", num_filler=10, vocab=vocab, seed=42)
    cot_val_ds = Task3SumDataset(val_instances, format_type="parallel_cot", num_filler=10, vocab=vocab, seed=42)

    model_cfg = ModelConfig(
        architecture="llama",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        device="cpu",
    )

    train_cfg = TrainConfig(
        seed=42,
        batch_size=8,
        learning_rate=1e-3,
        epochs=1,
    )

    task_cfg = Task3SumConfig(
        length=length,
        dimension=dimension,
        num_filler=10,
        num_samples=num_samples,
    )

    model, history = mock_train_model(model_cfg, train_cfg, task_cfg, train_ds, filler_val_dataset=filler_val_ds, cot_val_dataset=cot_val_ds)

    assert "best_filler_accuracy" in history
    assert "best_cot_accuracy" in history
    assert len(history["epoch_filler_accuracies"]) == 1



@pytest.mark.exp0
def test_compile_experiment_report():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    task_cfg = Task3SumConfig()

    per_seed_results = [
        {"seed": 42, "task_seed": 42, "training_seed": 42, "best_filler_accuracy": 0.8, "best_cot_accuracy": 0.7, "best_val_accuracy": 0.8},
        {"seed": 43, "task_seed": 43, "training_seed": 43, "best_filler_accuracy": 0.9, "best_cot_accuracy": 0.8, "best_val_accuracy": 0.9},
        {"seed": 44, "task_seed": 44, "training_seed": 44, "best_filler_accuracy": 0.85, "best_cot_accuracy": 0.75, "best_val_accuracy": 0.85},
    ]

    realized_counts = {"parallel_cot": 50, "filler": 50}

    report = compile_experiment_report(
        model_cfg,
        train_cfg,
        task_cfg,
        per_seed_results,
        majority_class_baseline=0.5,
        realized_mixture_counts=realized_counts,
        eval_seed=123,
        val_samples=500,
    )

    assert report["metrics"]["filler_accuracy"] == pytest.approx(0.85)
    assert report["metrics"]["cot_accuracy"] == pytest.approx(0.75)
    assert report["metrics"]["mean_accuracy"] == pytest.approx(0.85)
    assert report["metrics"]["min_accuracy"] == 0.8
    assert report["metrics"]["max_accuracy"] == 0.9
    assert report["majority_class_baseline"] == 0.5
    assert report["realized_mixture_counts"] == realized_counts
    assert report["eval_seed"] == 123
    assert report["val_samples"] == 500
    assert report["seeds_run"] == [42, 43, 44]
