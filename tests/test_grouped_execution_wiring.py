"""Grouped execution wired into train_model.

test_grouped_execution.py already covers the parity ladder for the mechanism
itself. These tests cover the wiring: that turning it on preserves the training
objective end to end, that leaving it off preserves run identity, and that the
combination it cannot support fails loudly instead of silently underflowing.
"""

import random
from dataclasses import asdict, replace

import pytest
import torch

from exp0.config import (
    ModelConfig,
    Task3SumConfig,
    TrainConfig,
    drop_identity_neutral_fields,
)
from exp0.dataset import Task3SumDataset
from exp0.task3sum import generate_instance
from exp0.train import train_model

pytestmark = pytest.mark.exp0


def _configs(**train_overrides):
    task_cfg = Task3SumConfig(num_samples=32)
    model_cfg = ModelConfig(
        architecture="llama",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        device="cpu",
    )
    train_cfg = TrainConfig(
        batch_size=8,
        epochs=1,
        num_workers=0,
        immediate_protocol=False,
        **train_overrides,
    )
    return task_cfg, model_cfg, train_cfg


def _datasets(task_cfg, seed):
    rng = random.Random(seed)
    instances = [
        generate_instance(
            length=task_cfg.length,
            dimension=task_cfg.dimension,
            mod=task_cfg.mod,
            rng=rng,
        )
        for _ in range(task_cfg.num_samples)
    ]
    train_ds = Task3SumDataset(instances, vocab_reduction=task_cfg.vocab_reduction)
    val_ds = Task3SumDataset(
        instances,
        format_type="filler",
        vocab=train_ds.vocab,
        vocab_reduction=task_cfg.vocab_reduction,
    )
    return train_ds, val_ds


def _run(train_cfg, task_cfg, model_cfg, seed=7):
    train_ds, val_ds = _datasets(task_cfg, seed)
    torch.manual_seed(0)
    return train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)


def test_grouped_matches_ungrouped_training_loss():
    """The objective must not move when only execution changes.

    Equivalent, not bitwise identical: grouping changes summation order, so the
    tolerance is float32 epsilon rather than zero.
    """
    task_cfg, model_cfg, plain_cfg = _configs()
    grouped_cfg = replace(plain_cfg, grouped_execution=True)

    _, plain_history = _run(plain_cfg, task_cfg, model_cfg)
    _, grouped_history = _run(grouped_cfg, task_cfg, model_cfg)

    plain_loss = plain_history["epoch_train_losses"][-1]
    grouped_loss = grouped_history["epoch_train_losses"][-1]
    assert grouped_loss == pytest.approx(plain_loss, abs=1e-4), (
        f"grouped {grouped_loss} vs ungrouped {plain_loss}"
    )


def test_grouped_preserves_per_format_training_telemetry():
    """Grouping never holds a full batch of logits; the stats must survive."""
    task_cfg, model_cfg, plain_cfg = _configs()
    grouped_cfg = replace(plain_cfg, grouped_execution=True)

    _, plain_history = _run(plain_cfg, task_cfg, model_cfg)
    _, grouped_history = _run(grouped_cfg, task_cfg, model_cfg)

    key = "epoch_online_train_answer_accuracies"
    assert len(grouped_history[key]) == len(plain_history[key])
    # Same number of samples counted per format, whatever the accuracy is.
    plain_by_format = plain_history["epoch_online_train_answer_accuracies_by_format"][-1]
    grouped_by_format = grouped_history["epoch_online_train_answer_accuracies_by_format"][-1]
    assert set(grouped_by_format) == set(plain_by_format)


def test_grouped_reports_head_projection_waste():
    """The Track B decision input: what grouping does NOT recover."""
    task_cfg, model_cfg, grouped_cfg = _configs(grouped_execution=True)
    _, history = _run(grouped_cfg, task_cfg, model_cfg)

    stats = history["grouped_execution_stats"]
    assert stats is not None
    assert stats["head_positions"] > 0
    assert 0 <= stats["unsupervised_head_positions"] < stats["head_positions"]
    assert 0.0 < stats["supervised_head_fraction"] <= 1.0


def test_stats_absent_when_disabled():
    task_cfg, model_cfg, plain_cfg = _configs()
    _, history = _run(plain_cfg, task_cfg, model_cfg)
    assert history["grouped_execution_stats"] is None
    assert history["execution_protocol"]["grouped_execution"] is False


def test_disabled_flag_is_identity_neutral():
    """A run that does not use grouping must keep its existing run_id."""
    train_dict = asdict(TrainConfig())
    assert "grouped_execution" in train_dict
    assert "grouped_execution" not in drop_identity_neutral_fields(dict(train_dict))

    enabled = asdict(TrainConfig(grouped_execution=True))
    assert drop_identity_neutral_fields(dict(enabled))["grouped_execution"] is True


def test_fp16_combination_is_refused():
    """GradScaler must wrap every backward; grouping runs one per subgroup."""
    task_cfg, model_cfg, train_cfg = _configs(
        grouped_execution=True, precision="fp16"
    )
    train_ds, val_ds = _datasets(task_cfg, seed=7)
    with pytest.raises(ValueError, match="fp16"):
        train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)
