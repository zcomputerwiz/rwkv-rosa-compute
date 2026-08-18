import torch

from rosa_compute import ROSAConfig, ROSAModelSkeleton


def test_model_state_dict_keys():
    config = ROSAConfig(n_layer=2, n_embd=16, vocab_size=100, rosa_groups=4)
    model = ROSAModelSkeleton(config)
    state_keys = set(model.state_dict().keys())

    expected_keys = {
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
        "blocks.0.ln2.weight",
        "blocks.0.ln2.bias",
        "blocks.0.ln3.weight",
        "blocks.0.ln3.bias",
        "blocks.0.rosa.x_q",
        "blocks.0.rosa.x_k",
        "blocks.0.rosa.x_v",
        "blocks.0.rosa.q.weight",
        "blocks.0.rosa.q.bias",
        "blocks.0.rosa.k.weight",
        "blocks.0.rosa.k.bias",
        "blocks.0.rosa.v.weight",
        "blocks.0.rosa.v.bias",
        "blocks.0.rosa.rosa_qkv.emb",
        "blocks.0.rosa.o.weight",
        "blocks.0.rosa.o.bias",
        "blocks.0.ffn.x_k",
        "blocks.0.ffn.key.weight",
        "blocks.0.ffn.value.weight",
        "blocks.1.ln2.weight",
        "blocks.1.ln2.bias",
        "blocks.1.ln3.weight",
        "blocks.1.ln3.bias",
        "blocks.1.rosa.x_q",
        "blocks.1.rosa.x_k",
        "blocks.1.rosa.x_v",
        "blocks.1.rosa.q.weight",
        "blocks.1.rosa.q.bias",
        "blocks.1.rosa.k.weight",
        "blocks.1.rosa.k.bias",
        "blocks.1.rosa.v.weight",
        "blocks.1.rosa.v.bias",
        "blocks.1.rosa.rosa_qkv.emb",
        "blocks.1.rosa.o.weight",
        "blocks.1.rosa.o.bias",
        "blocks.1.ffn.x_k",
        "blocks.1.ffn.key.weight",
        "blocks.1.ffn.value.weight",
        "ln_out.weight",
        "ln_out.bias",
        "head.weight",
    }

    assert state_keys == expected_keys
    # Block 1 should NOT have ln0
    assert "blocks.1.ln0.weight" not in state_keys


def test_model_forward_shape_and_dtype():
    config = ROSAConfig(
        n_layer=1, n_embd=16, vocab_size=50, rosa_groups=4, dtype=torch.float32
    )
    model = ROSAModelSkeleton(config)

    # Check parameters dtype
    for param in model.parameters():
        assert param.dtype == torch.float32

    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model(idx)
    assert out.shape == (1, 4, 50)


def test_model_residual_and_ffn_execution():
    """Targeted test proving ln0, ln2, ln3, ROSA, and FFN execution contribute to the block output."""
    config = ROSAConfig(
        n_layer=1, n_embd=8, vocab_size=20, rosa_groups=2, dtype=torch.float32
    )
    model = ROSAModelSkeleton(config)
    model.eval()

    idx = torch.tensor([[1, 2]], dtype=torch.long)

    # Forward pass through block
    x_emb = model.emb(idx)
    block = model.blocks[0]

    # Step by step calculation
    x_ln0 = block.ln0(x_emb)
    x_rosa = block.rosa(block.ln3(x_ln0))
    x_after_rosa = x_ln0 + x_rosa
    x_ffn = block.ffn(block.ln2(x_after_rosa))
    expected_block_out = x_after_rosa + x_ffn

    actual_block_out = block(x_emb)

    assert torch.allclose(actual_block_out, expected_block_out, atol=1e-5)
    # Ensure FFN is non-zero so residual actually mattered
    assert not torch.allclose(x_ffn, torch.zeros_like(x_ffn))
