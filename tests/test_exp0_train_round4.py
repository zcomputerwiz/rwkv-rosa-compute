import pytest
import torch

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset
from exp0.task3sum import generate_instance
from exp0.train import train_model

def get_tiny_configs():
    task_cfg = Task3SumConfig(num_samples=16)
    model_cfg = ModelConfig(
        architecture="llama",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        device="cpu"
    )
    train_cfg = TrainConfig(batch_size=8, epochs=1, num_workers=0)
    return task_cfg, model_cfg, train_cfg

def test_train_model_dual_validation():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()

    train_instances = [generate_instance(task_cfg.length, task_cfg.dimension, task_cfg.mod) for _ in range(task_cfg.num_samples)]
    val_instances = [generate_instance(task_cfg.length, task_cfg.dimension, task_cfg.mod) for _ in range(task_cfg.num_samples)]
    cot_instances = [generate_instance(task_cfg.length, task_cfg.dimension, task_cfg.mod) for _ in range(task_cfg.num_samples)]
    train_ds = Task3SumDataset(train_instances, vocab_reduction=task_cfg.vocab_reduction)
    val_ds = Task3SumDataset(val_instances, vocab=train_ds.vocab, vocab_reduction=task_cfg.vocab_reduction)
    cot_ds = Task3SumDataset(cot_instances, vocab=train_ds.vocab, vocab_reduction=task_cfg.vocab_reduction)

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds, cot_val_dataset=cot_ds)

    assert "epoch_train_losses" in history
    assert len(history["epoch_train_losses"]) == train_cfg.epochs
    assert "epoch_filler_accuracies" in history
    assert len(history["epoch_filler_accuracies"]) == train_cfg.epochs
    assert "epoch_cot_accuracies" in history
    assert len(history["epoch_cot_accuracies"]) == train_cfg.epochs

    assert "best_filler_accuracy" in history
    assert "best_cot_accuracy" in history
    assert "best_val_accuracy" in history
    assert history["best_val_accuracy"] == history["best_filler_accuracy"]

def test_train_model_single_validation():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()

    train_instances = [generate_instance(task_cfg.length, task_cfg.dimension, task_cfg.mod) for _ in range(task_cfg.num_samples)]
    val_instances = [generate_instance(task_cfg.length, task_cfg.dimension, task_cfg.mod) for _ in range(task_cfg.num_samples)]
    train_ds = Task3SumDataset(train_instances, vocab_reduction=task_cfg.vocab_reduction)
    val_ds = Task3SumDataset(val_instances, vocab=train_ds.vocab, vocab_reduction=task_cfg.vocab_reduction)

    _, history = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)

    assert "epoch_train_losses" in history
    assert "epoch_filler_accuracies" in history
    assert "epoch_cot_accuracies" not in history
    assert "best_cot_accuracy" not in history

def test_missing_ids_fails():
    task_cfg, model_cfg, train_cfg = get_tiny_configs()
    train_instances = [generate_instance(task_cfg.length, task_cfg.dimension, task_cfg.mod) for _ in range(task_cfg.num_samples)]
    val_instances = [generate_instance(task_cfg.length, task_cfg.dimension, task_cfg.mod) for _ in range(task_cfg.num_samples)]
    train_ds = Task3SumDataset(train_instances, vocab_reduction=task_cfg.vocab_reduction)
    val_ds = Task3SumDataset(val_instances, vocab=train_ds.vocab, vocab_reduction=task_cfg.vocab_reduction)

    # Mutate vocabulary intentionally to break it
    if "ANS" in train_ds.vocab.token2id:
        del train_ds.vocab.token2id["ANS"]

    with pytest.raises(ValueError, match="Vocabulary must contain 'ANS', 'True', and 'False'"):
        train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)
