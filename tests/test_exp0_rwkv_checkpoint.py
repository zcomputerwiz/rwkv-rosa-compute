"""Synthetic tests for Experiment 0B stock RWKV-7 checkpoint loading."""

import torch

from exp0.config import ModelConfig
from exp0.models.rwkv import RWKV7Backbone
from exp0.rwkv_checkpoint import (
    infer_checkpoint_architecture,
    load_pretrained_backbone,
    sha256_file,
)


def _target_to_source_key(target_key: str) -> str:
    if target_key.startswith("ln_out."):
        return target_key

    parts = target_key.split(".")
    assert parts[0] == "layers"
    layer_id = parts[1]
    component = parts[2]
    suffix = ".".join(parts[3:])

    if component == "time_mix":
        return f"blocks.{layer_id}.att.{suffix}"
    if component == "channel_mix":
        return f"blocks.{layer_id}.ffn.{suffix}"
    if component in {"ln0", "ln1", "ln2"}:
        return f"blocks.{layer_id}.{component}.{suffix}"
    raise AssertionError(f"Unhandled target key: {target_key}")


def _synthetic_stock_checkpoint(
    hidden_size: int = 128,
    num_layers: int = 2,
    intermediate_size: int = 256,
    head_dim: int = 64,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(123)
    template = RWKV7Backbone(
        hidden_size=hidden_size,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
    )

    source: dict[str, torch.Tensor] = {}
    expected_target: dict[str, torch.Tensor] = {}
    for target_key, tensor in template.state_dict().items():
        if target_key in {
            "layers.0.time_mix.v0",
            "layers.0.time_mix.v1",
            "layers.0.time_mix.v2",
        }:
            # Stock x070 checkpoints may omit these unused first-layer values.
            continue
        value = torch.randn_like(tensor)
        source_key = _target_to_source_key(target_key)
        source[source_key] = value
        expected_target[target_key] = value

    # Original LM interface is intentionally replaced by Experiment 0.
    source["emb.weight"] = torch.randn(32, hidden_size)
    source["head.weight"] = torch.randn(32, hidden_size)

    # Upstream Block instantiates ln0 at every layer, though only block 0 uses it.
    for layer_id in range(1, num_layers):
        source[f"blocks.{layer_id}.ln0.weight"] = torch.randn(hidden_size)
        source[f"blocks.{layer_id}.ln0.bias"] = torch.randn(hidden_size)

    return source, expected_target


def _pretrained_config(checkpoint_path, **overrides) -> ModelConfig:
    values = {
        "architecture": "rwkv",
        "init_mode": "pretrained",
        "rwkv_checkpoint": str(checkpoint_path),
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "head_dim": 64,
        "device": "cpu",
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_infer_stock_checkpoint_architecture():
    state, _ = _synthetic_stock_checkpoint()
    inferred = infer_checkpoint_architecture(state)

    assert inferred == {
        "architecture": "rwkv7_x070",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "head_dim": 64,
        "num_attention_heads": 2,
    }


def test_strict_stock_checkpoint_load(tmp_path):
    state, expected_target = _synthetic_stock_checkpoint()
    checkpoint_path = tmp_path / "stock_rwkv7.pth"
    torch.save(state, checkpoint_path)

    model_cfg = _pretrained_config(
        checkpoint_path,
        rwkv_checkpoint_sha256=sha256_file(checkpoint_path),
    )
    backbone = RWKV7Backbone(
        hidden_size=128,
        num_layers=2,
        intermediate_size=256,
        head_dim=64,
    )
    before = {
        key: value.clone()
        for key, value in backbone.state_dict().items()
        if key
        in {
            "layers.0.time_mix.v0",
            "layers.0.time_mix.v1",
            "layers.0.time_mix.v2",
        }
    }

    provenance = load_pretrained_backbone(backbone, model_cfg)
    loaded = backbone.state_dict()

    for key, expected in expected_target.items():
        assert torch.equal(loaded[key], expected), key
    for key, expected in before.items():
        assert torch.equal(loaded[key], expected), key

    assert provenance["mode"] == "pretrained"
    assert provenance["pretrained_scope"] == "backbone_only"
    assert provenance["strict_backbone_load"] is True
    assert provenance["checkpoint_sha256"] == sha256_file(checkpoint_path)
    assert "emb.weight" in provenance["ignored_source_keys"]
    assert "head.weight" in provenance["ignored_source_keys"]
    assert provenance["retained_target_defaults"] == [
        "layers.0.time_mix.v0",
        "layers.0.time_mix.v1",
        "layers.0.time_mix.v2",
    ]


def test_checkpoint_missing_required_backbone_key_fails(tmp_path):
    state, _ = _synthetic_stock_checkpoint()
    del state["blocks.0.att.key.weight"]
    checkpoint_path = tmp_path / "missing.pth"
    torch.save(state, checkpoint_path)

    backbone = RWKV7Backbone(
        hidden_size=128,
        num_layers=2,
        intermediate_size=256,
        head_dim=64,
    )
    model_cfg = _pretrained_config(checkpoint_path)

    try:
        load_pretrained_backbone(backbone, model_cfg)
    except ValueError as exc:
        assert "does not cover the complete local backbone" in str(exc)
    else:
        raise AssertionError("Missing required backbone key was accepted")


def test_checkpoint_shape_mismatch_fails(tmp_path):
    state, _ = _synthetic_stock_checkpoint()
    state["blocks.0.att.key.weight"] = torch.randn(127, 128)
    checkpoint_path = tmp_path / "bad_shape.pth"
    torch.save(state, checkpoint_path)

    backbone = RWKV7Backbone(
        hidden_size=128,
        num_layers=2,
        intermediate_size=256,
        head_dim=64,
    )
    model_cfg = _pretrained_config(checkpoint_path)

    try:
        load_pretrained_backbone(backbone, model_cfg)
    except ValueError as exc:
        assert "Checkpoint shape mismatch" in str(exc)
    else:
        raise AssertionError("Checkpoint shape mismatch was accepted")


def test_checkpoint_architecture_mismatch_fails_with_hint(tmp_path):
    state, _ = _synthetic_stock_checkpoint()
    checkpoint_path = tmp_path / "stock_rwkv7.pth"
    torch.save(state, checkpoint_path)

    backbone = RWKV7Backbone(
        hidden_size=64,
        num_layers=2,
        intermediate_size=128,
        head_dim=64,
    )
    model_cfg = _pretrained_config(
        checkpoint_path,
        hidden_size=64,
        num_attention_heads=1,
        intermediate_size=128,
    )

    try:
        load_pretrained_backbone(backbone, model_cfg)
    except ValueError as exc:
        message = str(exc)
        assert "checkpoint architecture does not match ModelConfig" in message
        assert "--hidden_size 128" in message
        assert "--intermediate_size 256" in message
    else:
        raise AssertionError("Incompatible checkpoint architecture was accepted")


def test_unexpected_checkpoint_key_fails(tmp_path):
    state, _ = _synthetic_stock_checkpoint()
    state["blocks.0.att.not_x070"] = torch.randn(1)
    checkpoint_path = tmp_path / "unexpected.pth"
    torch.save(state, checkpoint_path)

    backbone = RWKV7Backbone(
        hidden_size=128,
        num_layers=2,
        intermediate_size=256,
        head_dim=64,
    )
    model_cfg = _pretrained_config(checkpoint_path)

    try:
        load_pretrained_backbone(backbone, model_cfg)
    except ValueError as exc:
        assert "Unexpected RWKV-7 checkpoint key" in str(exc)
    else:
        raise AssertionError("Unexpected checkpoint key was accepted")
