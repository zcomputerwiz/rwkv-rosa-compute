"""Unit tests for Milestone 3: Training and evaluation harness."""

import random

import pytest

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab
from exp0.evaluate import compile_experiment_report
from exp0.task3sum import generate_instance
from exp0.train import train_model


def test_train_loop_end_to_end_cpu():
    rng = random.Random(42)
    length = 6
    dimension = 3
    num_samples = 20
    val_samples = 10

    train_instances = [generate_instance(length=length, dimension=dimension, rng=rng) for _ in range(num_samples)]
    val_instances = [generate_instance(length=length, dimension=dimension, rng=rng) for _ in range(val_samples)]

    vocab = build_default_vocab(length=length, dimension=dimension)

    train_ds = Task3SumDataset(train_instances, format_type="filler", num_filler=10, vocab=vocab, seed=42)
    val_ds = Task3SumDataset(val_instances, format_type="filler", num_filler=10, vocab=vocab, seed=42)

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

    model, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    assert "best_val_accuracy" in history
    assert len(history["epoch_val_accuracies"]) == 1
    assert 0.0 <= history["best_val_accuracy"] <= 1.0


def test_compile_experiment_report():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    task_cfg = Task3SumConfig()

    per_seed_results = [
        {"seed": 42, "best_val_accuracy": 0.8},
        {"seed": 43, "best_val_accuracy": 0.9},
        {"seed": 44, "best_val_accuracy": 0.85},
    ]

    report = compile_experiment_report(model_cfg, train_cfg, task_cfg, per_seed_results, majority_class_baseline=0.5)

    assert report["metrics"]["mean_accuracy"] == pytest.approx(0.85)
    assert report["metrics"]["min_accuracy"] == 0.8
    assert report["metrics"]["max_accuracy"] == 0.9
    assert report["majority_class_baseline"] == 0.5
