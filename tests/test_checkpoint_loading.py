import os
import tempfile

import pytest
import torch

from rosa_compute import (
    ROSAConfig,
    ROSAModelSkeleton,
    inspect_checkpoint,
    load_rosa_checkpoint,
    validate_checkpoint_state_dict,
)


def test_checkpoint_validation_synthetic():
    config = ROSAConfig(n_layer=1, n_embd=16, vocab_size=100, rosa_groups=4)
    model = ROSAModelSkeleton(config)
    state_dict = model.state_dict()

    info = validate_checkpoint_state_dict(state_dict, config=config)
    assert info["is_valid"] is True
    assert len(info["missing_keys"]) == 0
    assert len(info["unexpected_keys"]) == 0
    assert len(info["mismatched_shapes"]) == 0


def test_checkpoint_validation_missing_keys():
    config = ROSAConfig(n_layer=1, n_embd=16, vocab_size=100, rosa_groups=4)
    model = ROSAModelSkeleton(config)
    state_dict = model.state_dict()
    del state_dict["emb.weight"]

    info = validate_checkpoint_state_dict(state_dict, config=config)
    assert info["is_valid"] is False
    assert "emb.weight" in info["missing_keys"]


def test_checkpoint_validation_unexpected_keys():
    config = ROSAConfig(n_layer=1, n_embd=16, vocab_size=100, rosa_groups=4)
    model = ROSAModelSkeleton(config)
    state_dict = model.state_dict()
    state_dict["extra.foo.weight"] = torch.randn(10)

    info = validate_checkpoint_state_dict(state_dict, config=config)
    assert info["is_valid"] is False
    assert "extra.foo.weight" in info["unexpected_keys"]


def test_checkpoint_validation_mismatched_shapes():
    config = ROSAConfig(n_layer=1, n_embd=16, vocab_size=100, rosa_groups=4)
    model = ROSAModelSkeleton(config)
    state_dict = model.state_dict()
    state_dict["head.weight"] = torch.randn(100, 32)  # Expected (100, 16)

    info = validate_checkpoint_state_dict(state_dict, config=config)
    assert info["is_valid"] is False
    assert "head.weight" in info["mismatched_shapes"]


def test_checkpoint_validation_non_tensor():
    config = ROSAConfig(n_layer=1, n_embd=16, vocab_size=100, rosa_groups=4)
    model = ROSAModelSkeleton(config)
    state_dict = model.state_dict()
    state_dict["head.weight"] = "not_a_tensor"

    info = validate_checkpoint_state_dict(state_dict, config=config)
    assert info["is_valid"] is False
    assert "head.weight" in info["mismatched_shapes"]


def test_checkpoint_roundtrip_and_inspect():
    config = ROSAConfig(
        n_layer=1, n_embd=16, vocab_size=100, rosa_groups=4, dtype=torch.float32
    )
    model = ROSAModelSkeleton(config)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name

    try:
        torch.save(model.state_dict(), tmp_path)

        inspect_info = inspect_checkpoint(tmp_path)
        assert inspect_info["num_tensors"] == len(model.state_dict())
        assert inspect_info["total_parameters"] > 0

        state_dict, load_info = load_rosa_checkpoint(
            tmp_path, config=config, compute_hash=True
        )
        assert load_info["is_valid"] is True
        assert "sha256" in load_info

        # Load back into new model instance
        new_model = ROSAModelSkeleton(config)
        new_model.load_state_dict(state_dict)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_checkpoint_loading_non_dict_rejected():
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name

    try:
        torch.save(["not", "a", "dict"], tmp_path)
        with pytest.raises(ValueError, match="Expected checkpoint file to contain a state dict"):
            load_rosa_checkpoint(tmp_path)
        with pytest.raises(ValueError, match="Expected checkpoint file to contain a state dict"):
            inspect_checkpoint(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.checkpoint
def test_checkpoint_loading_optional():
    model_path = os.environ.get("ROSA_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        pytest.skip("ROSA_MODEL_PATH environment variable not set or file does not exist")
    state_dict, info = load_rosa_checkpoint(model_path, compute_hash=True)
    assert len(state_dict) > 0
    assert "sha256" in info



def test_upstream_target_checkpoint_compatibility():
    """Verifies checkpoint compatibility against the actual pinned upstream target structure
    (BlinkDL RWKV-8 ROSA 0.1B target, 12 layers, 768 hidden dim, 65536 vocab).
    """
    config = ROSAConfig(
        n_layer=12,
        n_embd=768,
        vocab_size=65536,
        rosa_groups=192,
        rosa_bits=4,
        context_length=512,
        dtype=torch.float32,
    )

    # 1. Generate exact upstream state dict layout matching 260222_rosa4bitLM_L12.py
    upstream_state_dict = {}
    upstream_state_dict["emb.weight"] = torch.randn(config.vocab_size, config.n_embd)

    for i in range(config.n_layer):
        prefix = f"blocks.{i}."
        if i == 0:
            upstream_state_dict[f"{prefix}ln0.weight"] = torch.ones(config.n_embd)
            upstream_state_dict[f"{prefix}ln0.bias"] = torch.zeros(config.n_embd)

        upstream_state_dict[f"{prefix}ln2.weight"] = torch.ones(config.n_embd)
        upstream_state_dict[f"{prefix}ln2.bias"] = torch.zeros(config.n_embd)
        upstream_state_dict[f"{prefix}ln3.weight"] = torch.ones(config.n_embd)
        upstream_state_dict[f"{prefix}ln3.bias"] = torch.zeros(config.n_embd)

        # ROSA layer
        upstream_state_dict[f"{prefix}rosa.x_q"] = torch.zeros(1, 1, config.n_embd)
        upstream_state_dict[f"{prefix}rosa.x_k"] = torch.zeros(1, 1, config.n_embd)
        upstream_state_dict[f"{prefix}rosa.x_v"] = torch.zeros(1, 1, config.n_embd)
        upstream_state_dict[f"{prefix}rosa.q.weight"] = torch.randn(config.n_embd, config.n_embd)
        upstream_state_dict[f"{prefix}rosa.q.bias"] = torch.zeros(config.n_embd)
        upstream_state_dict[f"{prefix}rosa.k.weight"] = torch.randn(config.n_embd, config.n_embd)
        upstream_state_dict[f"{prefix}rosa.k.bias"] = torch.zeros(config.n_embd)
        upstream_state_dict[f"{prefix}rosa.v.weight"] = torch.randn(config.n_embd, config.n_embd)
        upstream_state_dict[f"{prefix}rosa.v.bias"] = torch.zeros(config.n_embd)
        upstream_state_dict[f"{prefix}rosa.rosa_qkv.emb"] = torch.ones(1, 1, config.n_embd)
        upstream_state_dict[f"{prefix}rosa.o.weight"] = torch.randn(config.n_embd, config.n_embd)
        upstream_state_dict[f"{prefix}rosa.o.bias"] = torch.zeros(config.n_embd)

        # FFN / ChannelMix
        upstream_state_dict[f"{prefix}ffn.x_k"] = torch.zeros(1, 1, config.n_embd)
        upstream_state_dict[f"{prefix}ffn.key.weight"] = torch.randn(config.n_embd * 4, config.n_embd)
        upstream_state_dict[f"{prefix}ffn.value.weight"] = torch.randn(config.n_embd, config.n_embd * 4)

    upstream_state_dict["ln_out.weight"] = torch.ones(config.n_embd)
    upstream_state_dict["ln_out.bias"] = torch.zeros(config.n_embd)
    upstream_state_dict["head.weight"] = torch.randn(config.vocab_size, config.n_embd)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name

    try:
        torch.save(upstream_state_dict, tmp_path)

        # 1. Verify safe loading with weights_only=True
        state_dict, validation_info = load_rosa_checkpoint(
            tmp_path, config=config, compute_hash=True
        )

        # 2. Verify state dict validation accepts it
        assert validation_info["is_valid"] is True
        assert len(validation_info["missing_keys"]) == 0
        assert len(validation_info["unexpected_keys"]) == 0
        assert len(validation_info["mismatched_shapes"]) == 0

        # 3. Verify ROSAModelSkeleton loads it
        model = ROSAModelSkeleton(config)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        assert len(missing_keys) == 0
        assert len(unexpected_keys) == 0

        # 4. Verify parameter names and layout match upstream exact expectation
        model_sd = model.state_dict()
        assert set(model_sd.keys()) == set(upstream_state_dict.keys())

        # 5. Verify no accidental local-only keys are required
        for k, v in model_sd.items():
            assert tuple(v.shape) == tuple(upstream_state_dict[k].shape)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
