import torch

from rosa_compute import blinkdl_rosa_4bit_reference, rosa_4bit_forward


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
