import torch

from rosa_compute import (
    ROSALayerCompat,
    apply_blinkdl_embedding,
    blinkdl_rosa_4bit_reference,
    rosa_4bit_forward,
)


def test_three_way_forward_equivalence_cpu():
    """Verifies that BlinkDL reference and rosa_soft reference produce identical
    signed ROSA outputs {-1.0, 0.0, +1.0} on CPU across various shapes up to C=768.
    """
    test_shapes = [
        (1, 4, 16),
        (1, 8, 16),
        (1, 16, 16),
        (1, 32, 16),
        (1, 4, 768),  # Full 0.1B target shape (192 groups * 4 bits)
    ]

    for B, T, C in test_shapes:
        torch.manual_seed(42)
        q = torch.randn(B, T, C)
        k = torch.randn(B, T, C)
        v = torch.randn(B, T, C)

        out_blinkdl = blinkdl_rosa_4bit_reference(q, k, v)
        out_rosa_soft_ref = rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=False)

        assert out_blinkdl.shape == out_rosa_soft_ref.shape
        # Exact equality for discrete {-1.0, 0.0, +1.0} signed ROSA symbols
        assert torch.equal(out_blinkdl, out_rosa_soft_ref)


def test_apply_blinkdl_embedding_semantics():
    """Direct deterministic tests for apply_blinkdl_embedding sign & zero rules."""
    # signed_rosa: [1, 3, 4] with +1, -1, 0, and NaN values
    signed_rosa = torch.tensor([
        [
            [1.0, -1.0, 0.0, float("nan")],
            [1.0, 1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ])
    emb = torch.tensor([[[2.0, -3.0, 4.0, 0.5]]])

    out = apply_blinkdl_embedding(signed_rosa, emb)

    expected = torch.tensor([
        [
            [2.0, 3.0, 0.0, 0.0],       # +1*2.0, -1*(-3.0), 0.0, NaN->0.0
            [2.0, -3.0, -4.0, -0.5],    # +1*2.0, +1*(-3.0), -1*4.0, -1*0.5
            [0.0, 0.0, 0.0, 0.0],       # all 0
        ]
    ])

    assert torch.equal(out, expected)


def test_rosa_layer_compat_reference_equivalence():
    """Verifies that ROSALayerCompat produces identical outputs whether using BlinkDL ref or rosa_soft ref."""
    torch.manual_seed(123)
    layer = ROSALayerCompat(n_embd=16, max_suffix_length=512)
    layer.eval()

    x = torch.randn(2, 8, 16)

    out_blinkdl = layer(x, use_cuda=False, use_blinkdl_ref=True)
    out_rosa_soft = layer(x, use_cuda=False, use_blinkdl_ref=False)

    assert torch.allclose(out_blinkdl, out_rosa_soft, atol=1e-5)


def test_embedding_double_application_regression():
    """Verifies that applying emb twice yields a different output, catching double-application bugs."""
    signed_rosa = torch.tensor([[[1.0, -1.0, 0.0, 1.0]]])
    emb = torch.tensor([[[2.0, 3.0, 4.0, 5.0]]])

    applied_once = apply_blinkdl_embedding(signed_rosa, emb)
    applied_twice = apply_blinkdl_embedding(applied_once, emb)

    assert not torch.equal(applied_once, applied_twice)
