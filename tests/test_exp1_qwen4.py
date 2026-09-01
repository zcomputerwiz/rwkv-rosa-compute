import hashlib

import pytest
import torch

from exp0.config import TrainConfig
from exp0.train import set_seed
from exp1.dataset import PointerChaseDataset
from exp1.pointer_chase import ChaseSpec, generate_dataset
from exp1.qwen4_micro import Qwen4MicroConfig, create_qwen4_micro_model
from exp1.train import _model_signature, forward_logits, train_model
from exp1.workspace import Workspace


def _model(config: Qwen4MicroConfig, spec: ChaseSpec):
    return create_qwen4_micro_model(config, d_input=spec.d_input)


def _nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in module.parameters()
    )


def test_registered_qwen4_micro_config_is_complete_and_fixed():
    hybrid = Qwen4MicroConfig(vocab_size=16)
    recurrent = Qwen4MicroConfig(vocab_size=16, variant="all-gdn")

    assert hybrid.layer_types == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    )
    assert recurrent.layer_types == ("linear_attention",) * 4
    assert hybrid.resolved()["ple_layer_ids"] == []
    assert hybrid.resolved()["use_cache"] is False

    with pytest.raises(ValueError, match="registered Qwen4-Exp pilot"):
        Qwen4MicroConfig(vocab_size=16, hidden_size=64)


def test_qwen4_micro_forward_backward_reaches_every_component():
    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    set_seed(7)
    model = _model(Qwen4MicroConfig(vocab_size=16), spec)
    workspace = Workspace(128, num_slots=8, num_steps=8)
    inputs = torch.randn(2, spec.seq_len(0), spec.d_input)

    logits = forward_logits(model, inputs, workspace)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2]))
    loss.backward()

    assert logits.shape == (2, 16)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert _nonzero_gradient(model.input_proj)
    assert _nonzero_gradient(model.head)
    layers = model.backbone.model.layers
    assert _nonzero_gradient(layers[0].linear_attn)
    assert _nonzero_gradient(layers[3].self_attn)
    assert _nonzero_gradient(layers[0].mlp.experts)
    assert _nonzero_gradient(layers[0].attn_hyper_connection)
    assert _nonzero_gradient(workspace)
    assert any("ple" in name for name, _ in model.named_modules()) is False

    model_parameters = sum(p.numel() for p in model.parameters())
    totals = {
        model_parameters + sum(p.numel() for p in Workspace(128, num_slots=m, num_steps=k).parameters())
        for m, k in ((1, 1), (1, 8), (8, 1), (8, 8))
    }
    assert len(totals) == 1


def test_qwen4_micro_initialization_is_seeded():
    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    config = Qwen4MicroConfig(vocab_size=16, variant="all-gdn")

    def digest(seed: int) -> str:
        set_seed(seed)
        model = _model(config, spec)
        sha = hashlib.sha256()
        for name, parameter in sorted(model.state_dict().items()):
            sha.update(name.encode())
            sha.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return sha.hexdigest()

    assert digest(41) == digest(41)
    assert digest(41) != digest(42)


def test_qwen4_signature_records_every_resolved_field():
    hybrid = Qwen4MicroConfig(vocab_size=16)
    recurrent = Qwen4MicroConfig(vocab_size=16, variant="all-gdn")

    signature = _model_signature(hybrid)
    assert signature == {"qwen4_exp": hybrid.resolved()}
    assert _model_signature(recurrent) != signature


def test_qwen4_cpu_checkpoint_resume_is_exact(tmp_path):
    spec = ChaseSpec(num_nodes=4, num_maps=2, max_depth=2)
    train_ds = PointerChaseDataset(
        generate_dataset(1, 1, depth=1, seed=1, num_nodes=4, num_maps=2),
        spec,
    )
    val_ds = PointerChaseDataset(
        generate_dataset(1, 1, depth=1, seed=2, num_nodes=4, num_maps=2),
        spec,
    )
    config = Qwen4MicroConfig(vocab_size=4, variant="all-gdn")
    train_config = TrainConfig(
        seed=42,
        batch_size=1,
        precision="fp32",
        num_workers=0,
        epochs=2,
        learning_rate=1e-3,
    )
    common = dict(
        depth=1,
        train_data_seed=1,
        val_data_seed=2,
        train_size=1,
        val_size=1,
        queries_per_memory=1,
    )
    device = torch.device("cpu")

    set_seed(42)
    full, full_history = train_model(
        _model(config, spec), train_ds, val_ds, spec, config, train_config,
        device, **common,
    )

    checkpoint_dir = tmp_path / "checkpoints"
    set_seed(42)
    train_model(
        _model(config, spec), train_ds, val_ds, spec, config, train_config,
        device, checkpoint_path=checkpoint_dir, max_epochs=1, **common,
    )

    set_seed(999)
    resumed, resumed_history = train_model(
        _model(config, spec), train_ds, val_ds, spec, config, train_config,
        device, resume_from_checkpoint=checkpoint_dir / "latest.pt", **common,
    )

    assert (
        resumed_history["epoch_train_losses"]
        == full_history["epoch_train_losses"]
    )
    assert (
        resumed_history["epoch_val_accuracies"]
        == full_history["epoch_val_accuracies"]
    )
    assert resumed_history["best_val_accuracy"] == full_history["best_val_accuracy"]
    for (full_name, full_value), (resumed_name, resumed_value) in zip(
        full.state_dict().items(), resumed.state_dict().items()
    ):
        assert resumed_name == full_name
        assert torch.equal(resumed_value, full_value), full_name

    wrong = Qwen4MicroConfig(vocab_size=4, variant="hybrid")
    with pytest.raises(ValueError, match="qwen4_exp"):
        train_model(
            _model(wrong, spec), train_ds, val_ds, spec, wrong, train_config,
            device, resume_from_checkpoint=checkpoint_dir / "latest.pt",
            max_epochs=0, **common,
        )
