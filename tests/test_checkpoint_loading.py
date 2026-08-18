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
