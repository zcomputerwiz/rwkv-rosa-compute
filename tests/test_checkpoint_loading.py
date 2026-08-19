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


def _get_expected_0_1b_schema() -> dict[str, tuple[int, ...]]:
    schema: dict[str, tuple[int, ...]] = {
        "emb.weight": (65536, 768),
        "ln_out.weight": (768,),
        "ln_out.bias": (768,),
        "head.weight": (65536, 768),
    }
    for i in range(12):
        if i == 0:
            schema["blocks.0.ln0.weight"] = (768,)
            schema["blocks.0.ln0.bias"] = (768,)
        schema[f"blocks.{i}.ln2.weight"] = (768,)
        schema[f"blocks.{i}.ln2.bias"] = (768,)
        schema[f"blocks.{i}.ln3.weight"] = (768,)
        schema[f"blocks.{i}.ln3.bias"] = (768,)
        schema[f"blocks.{i}.rosa.x_q"] = (1, 1, 768)
        schema[f"blocks.{i}.rosa.x_k"] = (1, 1, 768)
        schema[f"blocks.{i}.rosa.x_v"] = (1, 1, 768)
        schema[f"blocks.{i}.rosa.q.weight"] = (768, 768)
        schema[f"blocks.{i}.rosa.q.bias"] = (768,)
        schema[f"blocks.{i}.rosa.k.weight"] = (768, 768)
        schema[f"blocks.{i}.rosa.k.bias"] = (768,)
        schema[f"blocks.{i}.rosa.v.weight"] = (768, 768)
        schema[f"blocks.{i}.rosa.v.bias"] = (768,)
        schema[f"blocks.{i}.rosa.rosa_qkv.emb"] = (1, 1, 768)
        schema[f"blocks.{i}.rosa.o.weight"] = (768, 768)
        schema[f"blocks.{i}.rosa.o.bias"] = (768,)
        schema[f"blocks.{i}.ffn.x_k"] = (1, 1, 768)
        schema[f"blocks.{i}.ffn.key.weight"] = (3072, 768)
        schema[f"blocks.{i}.ffn.value.weight"] = (768, 3072)
    return schema


def test_upstream_target_checkpoint_compatibility():
    """Verifies that ROSAModelSkeleton at 0.1B scale produces the exact expected state dict schema.

    Uses meta device for zero tensor allocation.
    """
    config = ROSAConfig(
        n_layer=12,
        n_embd=768,
        vocab_size=65536,
        rosa_groups=192,
        dtype=torch.float16,
    )
    with torch.device("meta"):
        model = ROSAModelSkeleton(config)

    actual_schema = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    expected_schema = _get_expected_0_1b_schema()

    missing_keys = set(expected_schema.keys()) - set(actual_schema.keys())
    unexpected_keys = set(actual_schema.keys()) - set(expected_schema.keys())

    assert len(missing_keys) == 0, f"Missing state dict keys: {missing_keys}"
    assert len(unexpected_keys) == 0, f"Unexpected state dict keys: {unexpected_keys}"

    for key, expected_shape in expected_schema.items():
        actual_shape = actual_schema[key]
        assert actual_shape == expected_shape, (
            f"Shape mismatch for key {key}: expected {expected_shape}, got {actual_shape}"
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
        res = new_model.load_state_dict(state_dict, strict=False)
        assert len(res.missing_keys) == 0
        assert len(res.unexpected_keys) == 0
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
