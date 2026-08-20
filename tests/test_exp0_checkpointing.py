"""Micro tests for Experiment 0 checkpoint save and exact resume."""

import dataclasses
import random
from pathlib import Path

import pytest
import torch

import exp0.train as train_module
from exp0.checkpointing import (
    CHECKPOINT_VERSION,
    ResumableRandomSampler,
    atomic_torch_save,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    validate_checkpoint_signature,
)
from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab
from exp0.generation import generate_protocol_packed_instances
from exp0.train import train_model
from scripts import run_experiment


def test_atomic_checkpoint_save_replaces_destination_without_temp_files(tmp_path):
    checkpoint = tmp_path / "latest.pt"

    atomic_torch_save(
        {"checkpoint_version": CHECKPOINT_VERSION, "value": torch.tensor([1, 2])},
        checkpoint,
    )
    atomic_torch_save(
        {"checkpoint_version": CHECKPOINT_VERSION, "value": torch.tensor([3, 4])},
        checkpoint,
    )

    loaded = load_training_checkpoint(checkpoint)
    assert torch.equal(loaded["value"], torch.tensor([3, 4]))
    assert list(tmp_path.glob(".*.tmp-*")) == []


def test_rng_state_round_trip_restores_python_and_torch_streams():
    random.seed(1234)
    torch.manual_seed(5678)
    state = capture_rng_state()

    expected_python = [random.random() for _ in range(3)]
    expected_torch = torch.rand(4)

    for _ in range(5):
        random.random()
        torch.rand(2)

    restore_rng_state(state)

    assert [random.random() for _ in range(3)] == expected_python
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)


def test_resumable_random_sampler_replays_exact_remaining_suffix():
    data = list(range(23))
    sampler = ResumableRandomSampler(data, epoch_seed=987654321)
    full_order = list(sampler)

    sampler.set_state(epoch_seed=987654321, start_index=9)
    resumed_order = list(sampler)

    assert resumed_order == full_order[9:]
    assert sorted(full_order) == list(range(len(data)))
    assert len(sampler) == len(data) - 9


def test_checkpoint_signature_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Differing signature sections: training"):
        validate_checkpoint_signature(
            {"model": {"hidden": 16}, "training": {"batch": 4}},
            {"model": {"hidden": 16}, "training": {"batch": 8}},
        )


def _tiny_training_fixture():
    length, dimension = 4, 2
    vocab = build_default_vocab(length=length, dimension=dimension)
    task_cfg = Task3SumConfig(
        length=length,
        dimension=dimension,
        num_filler=4,
        num_samples=12,
    )
    model_cfg = ModelConfig(
        architecture="llama",
        init_mode="random",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        head_dim=16,
        output_vocab_size=len(vocab),
        device="cpu",
    )
    train_cfg = TrainConfig(
        seed=17,
        batch_size=3,
        learning_rate=1e-3,
        epochs=2,
        mixture="filler",
        parallel_ratio=0.0,
        filler_ratio=1.0,
        num_workers=0,
        val_num_workers=0,
        pin_memory=False,
    )

    train_instances = generate_protocol_packed_instances(
        num_samples=12,
        length=length,
        dimension=dimension,
        true_rate=0.5,
        rng=random.Random(17),
    )
    val_instances = generate_protocol_packed_instances(
        num_samples=6,
        length=length,
        dimension=dimension,
        true_rate=0.5,
        rng=random.Random(9999),
    )
    train_ds = Task3SumDataset(
        train_instances,
        format_type="filler",
        num_filler=4,
        vocab=vocab,
        seed=17,
    )
    val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        num_filler=4,
        vocab=vocab,
        seed=9999,
    )
    return model_cfg, train_cfg, task_cfg, train_ds, val_ds


def test_mid_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    model_cfg, train_cfg, task_cfg, train_ds, val_ds = _tiny_training_fixture()

    baseline_model, baseline_history = train_model(
        model_cfg,
        train_cfg,
        task_cfg,
        train_ds,
        filler_val_dataset=val_ds,
        checkpoint_dir=tmp_path / "baseline",
        checkpoint_every_steps=2,
        checkpoint_run_id="micro-resume",
    )

    interrupted_dir = tmp_path / "interrupted"
    real_save = train_module._save_training_checkpoint

    class SimulatedInterruption(RuntimeError):
        pass

    def save_then_interrupt(path, **kwargs):
        saved_path = real_save(path, **kwargs)
        progress = kwargs["progress"]
        if Path(path).name == "latest.pt" and progress["optimizer_steps"] == 2:
            raise SimulatedInterruption("simulated process loss after durable save")
        return saved_path

    monkeypatch.setattr(train_module, "_save_training_checkpoint", save_then_interrupt)
    with pytest.raises(SimulatedInterruption):
        train_model(
            model_cfg,
            train_cfg,
            task_cfg,
            train_ds,
            filler_val_dataset=val_ds,
            checkpoint_dir=interrupted_dir,
            checkpoint_every_steps=2,
            checkpoint_run_id="micro-resume",
        )

    latest = interrupted_dir / "latest.pt"
    assert latest.is_file()
    interrupted_payload = load_training_checkpoint(latest)
    assert interrupted_payload["progress"]["epoch"] == 0
    assert interrupted_payload["progress"]["optimizer_steps"] == 2
    assert interrupted_payload["progress"]["samples_consumed_in_epoch"] == 6

    monkeypatch.setattr(train_module, "_save_training_checkpoint", real_save)
    resumed_model, resumed_history = train_model(
        model_cfg,
        train_cfg,
        task_cfg,
        train_ds,
        filler_val_dataset=val_ds,
        checkpoint_dir=interrupted_dir,
        checkpoint_every_steps=2,
        resume_checkpoint=latest,
        checkpoint_run_id="micro-resume",
    )

    baseline_state = baseline_model.state_dict()
    resumed_state = resumed_model.state_dict()
    assert baseline_state.keys() == resumed_state.keys()
    for name in baseline_state:
        torch.testing.assert_close(
            resumed_state[name],
            baseline_state[name],
            rtol=0,
            atol=0,
            msg=lambda message, name=name: f"{name}: {message}",
        )

    for key in (
        "epoch_train_losses",
        "epoch_online_train_answer_accuracies",
        "epoch_online_train_answer_accuracies_by_format",
        "epoch_filler_accuracies",
        "epoch_filler_answer_prediction_counts",
        "best_filler_accuracy",
        "best_filler_answer_prediction_counts",
        "best_online_train_answer_accuracy",
        "best_online_train_answer_accuracy_by_format",
        "optimizer_steps",
        "epoch_end_learning_rates",
    ):
        assert resumed_history[key] == baseline_history[key]

    assert resumed_history["checkpointing"]["resumed_from"] == str(latest.resolve())
    assert (interrupted_dir / "epoch_001.pt").is_file()
    assert (interrupted_dir / "epoch_002.pt").is_file()


def test_resume_from_qualifying_epoch_checkpoint_does_not_train_extra_epoch(tmp_path):
    model_cfg, train_cfg, task_cfg, train_ds, val_ds = _tiny_training_fixture()
    train_cfg = dataclasses.replace(
        train_cfg,
        epochs=3,
        early_stop_metric="filler_accuracy",
        early_stop_target=0.0,
        early_stop_patience=1,
    )
    checkpoint_dir = tmp_path / "early-stop-resume"

    _, stopped_history = train_model(
        model_cfg,
        train_cfg,
        task_cfg,
        train_ds,
        filler_val_dataset=val_ds,
        checkpoint_dir=checkpoint_dir,
        checkpoint_run_id="early-stop-resume",
    )
    assert stopped_history["epochs_trained"] == 1
    assert stopped_history["early_stopping"]["triggered"] is True

    epoch_checkpoint = checkpoint_dir / "epoch_001.pt"
    assert epoch_checkpoint.is_file()
    saved_payload = load_training_checkpoint(epoch_checkpoint)
    saved_steps = saved_payload["progress"]["optimizer_steps"]
    assert not (checkpoint_dir / "epoch_002.pt").exists()

    _, resumed_history = train_model(
        model_cfg,
        train_cfg,
        task_cfg,
        train_ds,
        filler_val_dataset=val_ds,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=epoch_checkpoint,
        checkpoint_run_id="early-stop-resume",
    )

    assert resumed_history["optimizer_steps"] == saved_steps
    assert resumed_history["epochs_trained"] == 1
    assert resumed_history["early_stopping"]["criterion_reached"] is True
    assert resumed_history["early_stopping"]["triggered"] is True
    assert resumed_history["early_stopping"]["stopped_after_epoch"] == 1
    assert not (checkpoint_dir / "epoch_002.pt").exists()


def test_runner_checkpoint_defaults_and_resume_seed_guard(tmp_path, monkeypatch):
    args = run_experiment.get_parser().parse_args([])
    assert args.checkpoint_every_steps == 5000

    argv = [
        "run_experiment.py",
        "--device",
        "cpu",
        "--num_samples",
        "2",
        "--val_samples",
        "2",
        "--epochs",
        "0",
        "--seeds",
        "1",
        "2",
        "--resume_checkpoint",
        str(tmp_path / "latest.pt"),
        "--out_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError, match="requires exactly one training seed"):
        run_experiment.main()


def test_fixed_budget_checkpoint_signature_omits_early_stop_fields():
    """Enabling the option must not invalidate checkpoints written without it."""
    model_cfg, train_cfg, task_cfg, train_ds, _ = _tiny_training_fixture()

    def signature_for(cfg):
        return train_module._checkpoint_signature(
            model_cfg,
            cfg,
            task_cfg,
            train_ds,
            epochs=cfg.epochs,
            steps_per_epoch=4,
            checkpoint_run_id="run",
        )

    fixed_budget = signature_for(train_cfg)
    assert "early_stop_metric" not in fixed_budget["training"]

    enabled = signature_for(
        dataclasses.replace(train_cfg, early_stop_metric="filler_accuracy")
    )
    assert enabled["training"]["early_stop_metric"] == "filler_accuracy"
    with pytest.raises(ValueError):
        validate_checkpoint_signature(enabled, fixed_budget)
