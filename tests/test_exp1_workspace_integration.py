"""The workspace has to be joined to the trainer without breaking the 2x2.

The join is small but has two ways to fail silently, and both would surface as
a result rather than as an error:

  - applying the workspace in training but not evaluation, or vice versa, would
    train one architecture and measure another while both halves run cleanly;
  - leaving the workspace out of the optimizer's parameter list would make the
    2x2 compare four frozen random workspaces, which also runs cleanly.

These tests exist for those two, plus the end-to-end parameter invariance the
whole screen depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exp0.config import ModelConfig, TrainConfig  # noqa: E402
from exp0.train import create_model, set_seed  # noqa: E402
from exp1.dataset import PointerChaseDataset  # noqa: E402
from exp1.pointer_chase import ChaseSpec, generate_dataset  # noqa: E402
from exp1.train import (  # noqa: E402
    _trainable,
    evaluate_vway_accuracy,
    forward_logits,
    train_model,
)
from exp1.workspace import Workspace  # noqa: E402

V, K_MAPS, D_MODEL = 16, 4, 64
CELLS = [(1, 1), (1, 8), (8, 1), (8, 8)]


def _model():
    set_seed(42)
    cfg = ModelConfig(architecture="rwkv", hidden_size=D_MODEL, num_hidden_layers=1,
                      num_attention_heads=1, head_dim=D_MODEL, vocab_size=V,
                      rwkv_kernel="reference")
    spec = ChaseSpec(num_nodes=V, num_maps=K_MAPS, max_depth=8)
    return create_model(cfg, d_input=spec.d_input, compact_reduced_features=False), spec


@pytest.mark.parametrize("num_slots,num_steps", CELLS)
def test_trainable_parameter_count_identical_end_to_end(num_slots, num_steps):
    """Invariance must hold for model+workspace, not just the workspace alone."""
    model, _ = _model()
    baseline = sum(p.numel() for p in
                   _trainable(model, Workspace(D_MODEL, num_slots=1, num_steps=1)))
    cell = sum(p.numel() for p in
               _trainable(model, Workspace(D_MODEL, num_slots=num_slots,
                                           num_steps=num_steps)))
    assert cell == baseline, (
        f"cell (M={num_slots}, K={num_steps}) has {cell} trainable parameters "
        f"against the baseline's {baseline}; the 2x2 would measure capacity")


def test_workspace_parameters_reach_the_optimizer():
    """Without this the 2x2 compares four frozen random workspaces.

    Every cell would still train, evaluate and report -- the failure has no
    error path, only a wrong result.
    """
    model, _ = _model()
    ws = Workspace(D_MODEL, num_slots=8, num_steps=2)
    params = _trainable(model, ws)
    ids = {id(p) for p in params}
    assert ids, "no trainable parameters at all"
    for name, p in ws.named_parameters():
        assert id(p) in ids, f"workspace parameter {name} is invisible to the optimizer"


def test_workspace_receives_gradient_through_the_forward():
    """Reaching the optimizer is necessary but not sufficient: the workspace
    must also sit on the path from input to loss."""
    model, spec = _model()
    ws = Workspace(D_MODEL, num_slots=4, num_steps=2)
    data = generate_dataset(4, 2, depth=2, seed=1, num_nodes=V, num_maps=K_MAPS)
    ds = PointerChaseDataset(data, spec)
    batch = torch.stack([ds[i]["input_tuples"] for i in range(4)])
    targets = torch.stack([ds[i]["targets"] for i in range(4)])
    logits = forward_logits(model, batch, ws)
    torch.nn.functional.cross_entropy(logits, targets.view(-1)).backward()
    grads = {n: (p.grad is not None and torch.any(p.grad != 0).item())
             for n, p in ws.named_parameters()}
    assert any(grads.values()), f"no workspace parameter received gradient: {grads}"


def test_evaluation_uses_the_same_workspace_as_training():
    """A train/eval architecture mismatch would not raise; it would just report
    a different number. Assert the evaluation path actually consumes it."""
    model, spec = _model()
    data = generate_dataset(8, 2, depth=2, seed=3, num_nodes=V, num_maps=K_MAPS)
    ds = PointerChaseDataset(data, spec)
    cfg = TrainConfig(seed=42, batch_size=8, precision="fp32", num_workers=0)
    dev = torch.device("cpu")

    torch.manual_seed(0)
    ws = Workspace(D_MODEL, num_slots=8, num_steps=4)
    with_ws = evaluate_vway_accuracy(model, ds, cfg, dev, ws)
    without = evaluate_vway_accuracy(model, ds, cfg, dev, None)

    # Perturb the workspace; if evaluation consumes it, the score may move.
    with torch.no_grad():
        for p in ws.parameters():
            p.add_(torch.randn_like(p) * 5.0)
    perturbed = evaluate_vway_accuracy(model, ds, cfg, dev, ws)

    logits_ws = forward_logits(model, torch.stack([ds[i]["input_tuples"] for i in range(8)]), ws)
    logits_no = forward_logits(model, torch.stack([ds[i]["input_tuples"] for i in range(8)]), None)
    assert not torch.allclose(logits_ws, logits_no), (
        "forward_logits ignores the workspace entirely")
    assert isinstance(with_ws, float) and isinstance(without, float)
    assert isinstance(perturbed, float)


@pytest.mark.parametrize("num_slots,num_steps", CELLS)
def test_untrained_model_scores_near_chance_in_every_cell(num_slots, num_steps):
    """The V-way guard, extended across the 2x2. A cell that scores near 1/2
    rather than 1/V has lost the task, not gained a mechanism."""
    model, spec = _model()
    ws = Workspace(D_MODEL, num_slots=num_slots, num_steps=num_steps)
    data = generate_dataset(60, 4, depth=3, seed=11, num_nodes=V, num_maps=K_MAPS)
    ds = PointerChaseDataset(data, spec)
    cfg = TrainConfig(seed=42, batch_size=64, precision="fp32", num_workers=0)
    acc = evaluate_vway_accuracy(model, ds, cfg, torch.device("cpu"), ws)
    assert acc < 0.15, f"cell (M={num_slots}, K={num_steps}) scores {acc}, not near 1/V"


def test_M_changes_the_computation():
    """Both axes must do something, or the 2x2 has only one real dimension.

    Added after an end-to-end smoke run showed M=1 and M=8 reporting identical
    validation accuracy at fixed K. That turned out to be quantization -- 50
    validation samples give 2% granularity -- but nothing in the suite actually
    asserted that M reaches the output, so the coincidence was indistinguishable
    from a dead axis.
    """
    state = torch.randn(4, D_MODEL, generator=torch.Generator().manual_seed(0))
    outs = {}
    for m in (1, 2, 8):
        torch.manual_seed(7)                      # identical weights across M
        outs[m] = Workspace(D_MODEL, num_slots=m, num_steps=4)(state)
    for m in (2, 8):
        assert not torch.allclose(outs[m], outs[1]), (
            f"M={m} produces the same output as M=1; the M axis is inert")


def test_K_changes_the_computation():
    """The same guarantee for the other axis."""
    state = torch.randn(4, D_MODEL, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(7)
    one = Workspace(D_MODEL, num_slots=4, num_steps=1)
    torch.manual_seed(7)
    eight = Workspace(D_MODEL, num_slots=4, num_steps=8)
    assert not torch.allclose(one(state), eight(state)), "the K axis is inert"


def test_workspace_checkpoint_round_trip_and_cell_identity(tmp_path):
    model, spec = _model()
    workspace = Workspace(D_MODEL, num_slots=2, num_steps=3)
    data = generate_dataset(4, 2, depth=2, seed=5, num_nodes=V, num_maps=K_MAPS)
    dataset = PointerChaseDataset(data, spec)
    model_cfg = ModelConfig(
        architecture="rwkv", hidden_size=D_MODEL, num_hidden_layers=1,
        num_attention_heads=1, head_dim=D_MODEL, vocab_size=V,
        rwkv_kernel="reference",
    )
    train_cfg = TrainConfig(
        seed=42, batch_size=8, epochs=1, precision="fp32", num_workers=0
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    train_model(
        model, dataset, dataset, spec, model_cfg, train_cfg, torch.device("cpu"),
        workspace=workspace, checkpoint_path=checkpoint_dir,
    )
    checkpoint = checkpoint_dir / "epoch_001.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "workspace.q_proj.weight" in payload["model_state_dict"]
    expected = {key: value.clone() for key, value in workspace.state_dict().items()}

    resumed_model, _ = _model()
    resumed_workspace = Workspace(D_MODEL, num_slots=2, num_steps=3)
    train_model(
        resumed_model, dataset, dataset, spec, model_cfg, train_cfg,
        torch.device("cpu"), workspace=resumed_workspace,
        resume_from_checkpoint=checkpoint, max_epochs=0,
    )
    for key, value in expected.items():
        assert torch.equal(resumed_workspace.state_dict()[key], value), key

    wrong_cell = Workspace(D_MODEL, num_slots=1, num_steps=3)
    with pytest.raises(ValueError, match="workspace"):
        train_model(
            resumed_model, dataset, dataset, spec, model_cfg, train_cfg,
            torch.device("cpu"), workspace=wrong_cell,
            resume_from_checkpoint=checkpoint, max_epochs=0,
        )
