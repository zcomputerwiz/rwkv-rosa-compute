import torch

from rosa_compute import blinkdl_rosa_4bit_reference, rosa_4bit_forward


def test_three_way_forward_equivalence_cpu():
    # Compare BlinkDL reference against rosa_soft reference on CPU for symbol-level / hard routing logic
    B, T, C = 1, 4, 16
    torch.manual_seed(42)
    q = torch.randn(B, T, C)
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)

    out_blinkdl = blinkdl_rosa_4bit_reference(q, k, v)
    out_rosa_soft = rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=False)

    # For binary 0/1 symbols:
    assert out_blinkdl.shape == out_rosa_soft.shape
    assert torch.allclose(out_blinkdl, out_rosa_soft, atol=1e-5)
