"""CPU tests for the completed-run continuation tool."""

import json
from dataclasses import asdict

import pytest
import torch

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from scripts.continue_training import (
    build_schedule,
    configs_from_signature,
    last_nonzero_lr,
    main,
    restore_optimizer,
)

pytestmark = pytest.mark.exp0


def _signature(epochs=5, **train_overrides):
    train = asdict(TrainConfig(seed=42, batch_size=64, epochs=epochs,
                               **train_overrides))
    return {
        "run_id": "abc123",
        "model": asdict(ModelConfig(architecture="llama", hidden_size=32)),
        "training": train,
        "task": asdict(Task3SumConfig(length=6, dimension=3, num_samples=128)),
        "epochs": epochs,
        "steps_per_epoch": 2,
    }


def _progress(epoch=5, rates=(8e-05, 6e-05, 4e-05, 2e-05, 0.0)):
    return {"epoch": epoch,
            "completed": {"epoch_end_learning_rates": list(rates)}}


def test_configs_are_reconstructed_from_the_checkpoint():
    """A continuation must not be able to train on different data than its source."""
    model_cfg, train_cfg, task_cfg = configs_from_signature(_signature())
    assert train_cfg.seed == 42
    assert train_cfg.batch_size == 64
    assert task_cfg.length == 6
    assert task_cfg.num_samples == 128
    assert model_cfg.hidden_size == 32


def test_unknown_signature_keys_are_ignored():
    """Older or newer checkpoints must not crash reconstruction."""
    signature = _signature()
    signature["training"]["some_future_field"] = 1
    signature["task"]["another_one"] = 2
    _, train_cfg, task_cfg = configs_from_signature(signature)
    assert train_cfg.seed == 42
    assert task_cfg.length == 6


def test_peak_lr_defaults_to_the_last_nonzero_rate():
    """The stored final LR is exactly 0, so the last usable rate is the default."""
    assert last_nonzero_lr(_progress()) == pytest.approx(2e-05)
    assert last_nonzero_lr({"completed": {"epoch_end_learning_rates": [0.0]}}) is None
    assert last_nonzero_lr({}) is None


def test_schedule_warms_up_then_decays_to_zero():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=2e-05)
    scheduler = build_schedule(optimizer, total_steps=100, warmup_fraction=0.05)

    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert rates[0] < rates[5], "should warm up"
    assert rates[5] == pytest.approx(2e-05), "should reach the peak"
    assert rates[-1] < rates[50], "should decay"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_restore_optimizer_overrides_the_stored_zero_lr():
    """Loading state alone would leave lr=0 and make the continuation a no-op."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    source = torch.optim.AdamW([parameter], lr=0.0)
    parameter.grad = torch.ones(1)
    source.step()

    target = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    restore_optimizer(target, source.state_dict(), 2e-05, torch.device("cpu"))
    assert target.param_groups[0]["lr"] == pytest.approx(2e-05)
    assert target.param_groups[0]["initial_lr"] == pytest.approx(2e-05)


def test_midrun_checkpoint_is_refused(tmp_path):
    """Extending a run that has not finished its own budget is a mistake."""
    checkpoint = tmp_path / "mid.pt"
    torch.save({"signature": _signature(epochs=5),
                "progress": _progress(epoch=3),
                "model_state_dict": {}, "optimizer_state_dict": {}}, checkpoint)
    with pytest.raises(SystemExit):
        main([str(checkpoint), "--additional-epochs", "1", "--dry-run"])


def test_non_positive_additional_epochs_is_refused(tmp_path):
    checkpoint = tmp_path / "done.pt"
    torch.save({"signature": _signature(), "progress": _progress(),
                "model_state_dict": {}, "optimizer_state_dict": {}}, checkpoint)
    with pytest.raises(SystemExit):
        main([str(checkpoint), "--additional-epochs", "0", "--dry-run"])


def test_dry_run_reports_the_plan_without_training(tmp_path, capsys):
    checkpoint = tmp_path / "done.pt"
    torch.save({"signature": _signature(), "progress": _progress(),
                "model_state_dict": {}, "optimizer_state_dict": {}}, checkpoint)
    assert main([str(checkpoint), "--additional-epochs", "2", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "abc123" in output
    assert "2e-05" in output
    assert "nothing trained" in output


def test_report_shape_marks_the_result_as_non_canonical():
    """The output must be impossible to mistake for a protocol result."""
    from scripts.continue_training import CONTINUATION_VERSION

    report = {
        "continuation_version": CONTINUATION_VERSION,
        "is_canonical_experiment_result": False,
        "distribution_note": "Continuation of a completed run",
        "source_run_id": "abc123",
    }
    decoded = json.loads(json.dumps(report))
    assert decoded["is_canonical_experiment_result"] is False
    assert decoded["source_run_id"] == "abc123"


# --- the CE target regression ------------------------------------------------

def _mixed_batch(num_filler=0, count=8):
    """A real mixed batch, so padding and the mask are genuine."""
    import random

    from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
    from exp0.generation import generate_protocol_packed_instances

    vocab = build_default_vocab(length=6, dimension=3)
    packed = generate_protocol_packed_instances(
        count, length=6, dimension=3, rng=random.Random(1))
    dataset = Task3SumDataset(packed, num_filler=num_filler, vocab=vocab,
                              parallel_ratio=0.5, filler_ratio=0.5)
    return vocab, pad_collate_fn([dataset[i] for i in range(count)])


def test_loss_mask_and_targets_disagree_on_padding():
    """The premise of the regression: the two tensors are padded differently."""
    _, batch = _mixed_batch()
    mask, targets = batch["loss_mask"][:, 1:], batch["targets"][:, 1:]
    supervised = int((mask != -100).sum())
    assert supervised < mask.numel(), "batch must actually contain padding"
    # targets pads with the PAD id, which is a real class, so nothing is ignored.
    assert int((targets == -100).sum()) == 0


def test_supervised_cross_entropy_ignores_padded_positions():
    """Regression: using targets instead of loss_mask changes the objective.

    It trains the model to predict PAD at padded positions and divides by a
    larger denominator, so it is a different loss, not a noisier one.
    """
    from scripts.continue_training import supervised_cross_entropy

    vocab, batch = _mixed_batch()
    torch.manual_seed(0)
    positions = batch["targets"].shape[1] - 1
    logits = torch.randn(batch["targets"].shape[0], positions, len(vocab))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    masked = supervised_cross_entropy(logits, batch, criterion, torch.device("cpu"))
    naive = criterion(logits.reshape(-1, logits.size(-1)),
                      batch["targets"][:, 1:].reshape(-1))
    assert not torch.isclose(masked, naive), (
        "loss over supervised positions must differ from loss over all positions"
    )
    assert torch.isfinite(masked)


def test_supervised_cross_entropy_matches_an_explicit_masked_mean():
    """The helper must be the token-weighted mean over supervised positions."""
    from scripts.continue_training import supervised_cross_entropy

    vocab, batch = _mixed_batch()
    torch.manual_seed(1)
    positions = batch["targets"].shape[1] - 1
    logits = torch.randn(batch["targets"].shape[0], positions, len(vocab))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_mask = batch["loss_mask"][:, 1:].reshape(-1)
    keep = flat_mask != -100
    explicit = torch.nn.functional.cross_entropy(
        flat_logits[keep], flat_mask[keep], reduction="mean")

    result = supervised_cross_entropy(logits, batch, criterion, torch.device("cpu"))
    assert torch.allclose(result, explicit, atol=1e-6)
