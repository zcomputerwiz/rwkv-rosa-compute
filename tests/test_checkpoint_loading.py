import os

import pytest
import torch

from rosa_compute import (
    ROSAConfig,
    load_rosa_checkpoint,
    validate_checkpoint_state_dict,
)


def test_checkpoint_validation_synthetic():
    config = ROSAConfig(n_layer=1, n_embd=16, vocab_size=100, rosa_groups=4)
    # Create minimal state dict
    state_dict = {
        "emb.weight": torch.randn(100, 16),
        "ln_out.weight": torch.randn(16),
        "ln_out.bias": torch.randn(16),
        "head.weight": torch.randn(100, 16),
        "blocks.0.ln0.weight": torch.randn(16),
        "blocks.0.ln0.bias": torch.randn(16),
        "blocks.0.ln2.weight": torch.randn(16),
        "blocks.0.ln2.bias": torch.randn(16),
        "blocks.0.ln3.weight": torch.randn(16),
        "blocks.0.ln3.bias": torch.randn(16),
        "blocks.0.ffn.x_k": torch.randn(1, 1, 16),
        "blocks.0.ffn.key.weight": torch.randn(64, 16),
        "blocks.0.ffn.value.weight": torch.randn(16, 64),
        "blocks.0.rosa.x_q": torch.randn(1, 1, 16),
        "blocks.0.rosa.x_k": torch.randn(1, 1, 16),
        "blocks.0.rosa.x_v": torch.randn(1, 1, 16),
        "blocks.0.rosa.q.weight": torch.randn(16, 16),
        "blocks.0.rosa.k.weight": torch.randn(16, 16),
        "blocks.0.rosa.v.weight": torch.randn(16, 16),
        "blocks.0.rosa.o.weight": torch.randn(16, 16),
        "blocks.0.rosa.rosa_qkv.emb": torch.randn(1, 1, 16),
    }
    info = validate_checkpoint_state_dict(state_dict, config=config)
    assert info["is_valid"] is True
    assert len(info["missing_keys"]) == 0
    assert len(info["unexpected_keys"]) == 0

@pytest.mark.checkpoint
def test_checkpoint_loading_optional():
    model_path = os.environ.get("ROSA_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        pytest.skip("ROSA_MODEL_PATH environment variable not set or file does not exist")
    state_dict, info = load_rosa_checkpoint(model_path, compute_hash=True)
    assert len(state_dict) > 0
    assert "sha256" in info
