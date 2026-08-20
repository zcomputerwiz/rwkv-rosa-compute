"""Lightweight integration sanity checks for the Experiment 0B RWKV path."""

import gc
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp0.config import ModelConfig, Task3SumConfig
from exp0.dataset import Task3SumDataset, build_default_vocab, pad_collate_fn
from exp0.models.rwkv import RWKV7Backbone
from exp0.rwkv_checkpoint import (
    infer_checkpoint_architecture,
    load_checkpoint_state_dict,
    sha256_file,
)
from exp0.task3sum import Instance3Sum
from exp0.train import create_model, initialize_model

pytestmark = [pytest.mark.exp0, pytest.mark.integration]


def _live_backbone() -> RWKV7Backbone:
    """Return a tiny RWKV with the initially-zero residual outputs activated."""
    torch.manual_seed(123)
    backbone = RWKV7Backbone(
        hidden_size=64,
        num_layers=1,
        intermediate_size=128,
        head_dim=64,
        rwkv_kernel="reference",
    )
    with torch.no_grad():
        for layer in backbone.layers:
            nn.init.normal_(layer.time_mix.output.weight, std=0.02)
            nn.init.normal_(layer.channel_mix.value.weight, std=0.02)
    backbone.eval()
    return backbone


@pytest.mark.parametrize("timesteps", [1, 15, 16, 17])
def test_rwkv_batch_elements_are_isolated(timesteps: int):
    """Batching independent sequences must not couple recurrent state."""
    backbone = _live_backbone()
    generator = torch.Generator().manual_seed(1000 + timesteps)
    seq_a = torch.randn(1, timesteps, 64, generator=generator)
    seq_b = torch.randn(1, timesteps, 64, generator=generator)

    with torch.no_grad():
        out_a = backbone(seq_a)
        out_b = backbone(seq_b)
        out_batched = backbone(torch.cat([seq_a, seq_b], dim=0))

    torch.testing.assert_close(out_batched[0:1], out_a, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_batched[1:2], out_b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("prefix_length", [1, 15, 16, 17])
def test_rwkv_prefix_is_unchanged_by_appended_suffix(prefix_length: int):
    """Later filler/transitions must never rewrite earlier recurrent outputs."""
    backbone = _live_backbone()
    generator = torch.Generator().manual_seed(2000 + prefix_length)
    prefix = torch.randn(2, prefix_length, 64, generator=generator)
    suffix = torch.randn(2, 5, 64, generator=generator)

    with torch.no_grad():
        prefix_only = backbone(prefix)
        with_suffix = backbone(torch.cat([prefix, suffix], dim=1))

    torch.testing.assert_close(
        with_suffix[:, :prefix_length],
        prefix_only,
        rtol=1e-5,
        atol=1e-6,
    )


def _tiny_match3_batch():
    task_cfg = Task3SumConfig(
        length=3,
        dimension=1,
        num_filler=0,
        num_samples=2,
    )
    vocab = build_default_vocab(length=3, dimension=1)
    instances = [
        Instance3Sum(
            tuples=[(1,), (2,), (7,)],
            has_3sum=True,
            matching_indices=(0, 1, 2),
        ),
        Instance3Sum(
            tuples=[(1,), (2,), (4,)],
            has_3sum=False,
            matching_indices=None,
        ),
    ]
    dataset = Task3SumDataset(
        instances,
        format_type="immediate",
        num_filler=0,
        vocab=vocab,
        seed=17,
    )
    batch = pad_collate_fn([dataset[0], dataset[1]])
    return task_cfg, vocab, batch


def _loss_from_batch(model, batch) -> torch.Tensor:
    logits = model.loss_logits(batch["input_tuples"], batch["targets"])
    shifted_targets = batch["loss_mask"][:, 1:].reshape(-1)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shifted_targets,
        ignore_index=-100,
    )


def _tiny_random_model(task_cfg, vocab):
    model_cfg = ModelConfig(
        architecture="rwkv",
        init_mode="random",
        rwkv_kernel="reference",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=128,
        head_dim=64,
        vocab_size=len(vocab),
        output_vocab_size=32000,
        device="cpu",
    )
    model = create_model(
        model_cfg,
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
    )
    initialize_model(model, model_cfg)
    return model


def _assert_finite_nonzero_grad(parameter: torch.nn.Parameter, name: str) -> None:
    assert parameter.grad is not None, f"{name} gradient is missing"
    assert torch.isfinite(parameter.grad).all(), f"{name} gradient is non-finite"
    assert parameter.grad.abs().max().item() > 0.0, f"{name} gradient is zero"


def test_rwkv_whole_match3_harness_one_optimizer_step():
    """Exercise dataset -> shared input seam -> RWKV -> LM head -> optimizer."""
    torch.manual_seed(321)
    task_cfg, vocab, batch = _tiny_match3_batch()
    model = _tiny_random_model(task_cfg, vocab)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    input_before = model.input_proj.weight.detach().clone()
    recurrent_before = model.backbone.layers[0].time_mix.output.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = _loss_from_batch(model, batch)
    assert torch.isfinite(loss)
    loss.backward()

    _assert_finite_nonzero_grad(model.input_proj.weight, "input_proj.weight")
    _assert_finite_nonzero_grad(model.head.weight, "head.weight")
    _assert_finite_nonzero_grad(
        model.backbone.layers[0].time_mix.output.weight,
        "backbone.layers.0.time_mix.output.weight",
    )

    optimizer.step()
    assert not torch.equal(model.input_proj.weight.detach(), input_before)
    assert not torch.equal(
        model.backbone.layers[0].time_mix.output.weight.detach(),
        recurrent_before,
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


def _write_stable_stock_checkpoint(tmp_path):
    """Write a stock-shaped x070 checkpoint using numerically stable local weights."""
    template = RWKV7Backbone(
        hidden_size=64,
        num_layers=1,
        intermediate_size=128,
        head_dim=64,
    )
    source = {}
    expected = {}
    optional_first_layer_v = {
        "layers.0.time_mix.v0",
        "layers.0.time_mix.v1",
        "layers.0.time_mix.v2",
    }
    for target_key, tensor in template.state_dict().items():
        if target_key in optional_first_layer_v:
            continue
        value = tensor.detach().clone()
        source[_target_to_source_key(target_key)] = value
        expected[target_key] = value

    # Stock LM interface is deliberately ignored by Experiment 0B.
    source["emb.weight"] = torch.zeros(8, 64)
    source["head.weight"] = torch.zeros(8, 64)

    checkpoint_path = tmp_path / "stock_rwkv7_integration.pth"
    torch.save(source, checkpoint_path)
    return checkpoint_path, expected


def test_synthetic_pretrained_checkpoint_runs_through_whole_wrapper(tmp_path):
    """Strict stock loading must integrate with the repaired Match-3 interface."""
    task_cfg, vocab, batch = _tiny_match3_batch()
    checkpoint_path, expected_backbone = _write_stable_stock_checkpoint(tmp_path)
    model_cfg = ModelConfig(
        architecture="rwkv",
        init_mode="pretrained",
        rwkv_checkpoint=str(checkpoint_path),
        rwkv_checkpoint_sha256=sha256_file(checkpoint_path),
        rwkv_kernel="reference",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=128,
        head_dim=64,
        vocab_size=len(vocab),
        output_vocab_size=32000,
        device="cpu",
    )
    model = create_model(
        model_cfg,
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
    )
    input_before = model.input_proj.weight.detach().clone()
    head_before = model.head.weight.detach().clone()

    provenance = initialize_model(model, model_cfg)

    assert provenance["strict_backbone_load"] is True
    assert "emb.weight" in provenance["ignored_source_keys"]
    assert "head.weight" in provenance["ignored_source_keys"]
    assert torch.equal(model.input_proj.weight.detach(), input_before)
    assert torch.equal(model.head.weight.detach(), head_before)
    assert torch.equal(
        model.backbone.layers[0].time_mix.key.weight.detach(),
        expected_backbone["layers.0.time_mix.key.weight"],
    )

    loss = _loss_from_batch(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    _assert_finite_nonzero_grad(model.input_proj.weight, "input_proj.weight")
    _assert_finite_nonzero_grad(
        model.backbone.layers[0].time_mix.output.weight,
        "backbone.layers.0.time_mix.output.weight",
    )


@pytest.mark.checkpoint
def test_real_stock_checkpoint_whole_wrapper_smoke():
    """Optional real-file acceptance gate enabled by EXP0_RWKV_CHECKPOINT."""
    checkpoint_path = os.environ.get("EXP0_RWKV_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("Set EXP0_RWKV_CHECKPOINT to run the real RWKV checkpoint smoke")

    state = load_checkpoint_state_dict(checkpoint_path)
    inferred = infer_checkpoint_architecture(state)
    del state
    gc.collect()

    task_cfg, vocab, batch = _tiny_match3_batch()
    model_cfg = ModelConfig(
        architecture="rwkv",
        init_mode="pretrained",
        rwkv_checkpoint=checkpoint_path,
        rwkv_checkpoint_sha256=sha256_file(checkpoint_path),
        rwkv_kernel="reference",
        hidden_size=int(inferred["hidden_size"]),
        num_hidden_layers=int(inferred["num_hidden_layers"]),
        num_attention_heads=int(inferred["num_attention_heads"]),
        intermediate_size=int(inferred["intermediate_size"]),
        head_dim=int(inferred["head_dim"]),
        vocab_size=len(vocab),
        output_vocab_size=32000,
        device="cpu",
    )
    model = create_model(
        model_cfg,
        d_input=task_cfg.mod * task_cfg.dimension + task_cfg.length,
        vocab=vocab,
        task_cfg=task_cfg,
    )
    provenance = initialize_model(model, model_cfg)
    assert provenance["strict_backbone_load"] is True
    assert provenance["checkpoint_sha256"] == model_cfg.rwkv_checkpoint_sha256

    loss = _loss_from_batch(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    _assert_finite_nonzero_grad(model.input_proj.weight, "input_proj.weight")
    _assert_finite_nonzero_grad(model.head.weight, "head.weight")
