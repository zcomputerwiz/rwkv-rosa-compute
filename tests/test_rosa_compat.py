import torch

from rosa_compute import ROSALayerCompat, rosa_4bit_forward


def test_rosa_compat_shape():
    B, T, C = 2, 8, 16
    q = torch.randn(B, T, C)
    k = torch.randn(B, T, C)
    v = torch.randn(B, T, C)
    out = rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=False)
    assert out.shape == (B, T, C)

def test_rosa_layer_compat():
    layer = ROSALayerCompat(n_embd=16, max_suffix_length=512)
    x = torch.randn(2, 8, 16)
    out = layer(x, use_cuda=False)
    assert out.shape == (2, 8, 16)
