import torch

from rosa_compute.blinkdl_reference import blinkdl_rosa_4bit_reference, rosa_slow_ref


def test_rosa_slow_ref_logic():
    # Correct test case matching sequence lengths
    q = [1, 1, 1]
    k = [1, 1, 1]
    v = [10, 20, 30]
    idx, ln = rosa_slow_ref(q, k, v)
    assert idx[2] == 30
    assert ln[2] == 2

def test_blinkdl_rosa_4bit_reference_basic():
    B, T, C = 1, 4, 8
    q = torch.randn(B, T, C)
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)
    out = blinkdl_rosa_4bit_reference(q, k, v)
    assert out.shape == (B, T, C)
