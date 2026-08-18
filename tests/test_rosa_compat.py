import pytest
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


def test_rosa_4bit_forward_validation():
    q = torch.randn(2, 8, 16)
    k = torch.randn(2, 8, 16)
    v = torch.randn(2, 8, 16)

    # Non-tensor
    with pytest.raises(TypeError, match="must be PyTorch Tensors"):
        rosa_4bit_forward("q", k, v)

    # Invalid rank
    with pytest.raises(ValueError, match="must be 3-D tensors"):
        rosa_4bit_forward(q.unsqueeze(0), k, v)

    # Shape mismatch
    with pytest.raises(ValueError, match="shapes must match"):
        rosa_4bit_forward(q, k[:, :4, :], v)

    # Dtype mismatch
    with pytest.raises(ValueError, match="dtypes must match"):
        rosa_4bit_forward(q, k.double(), v)

    # Non floating point
    with pytest.raises(ValueError, match="floating-point dtype"):
        rosa_4bit_forward(q.long(), k.long(), v.long())

    # Invalid dimensions
    with pytest.raises(ValueError, match="B>=1"):
        rosa_4bit_forward(torch.randn(0, 8, 16), torch.randn(0, 8, 16), torch.randn(0, 8, 16))

    # C not divisible by 4
    with pytest.raises(ValueError, match="divisible by 4"):
        rosa_4bit_forward(torch.randn(2, 8, 10), torch.randn(2, 8, 10), torch.randn(2, 8, 10))

    # max_suffix_length < 1
    with pytest.raises(ValueError, match="max_suffix_length must be >= 1"):
        rosa_4bit_forward(q, k, v, max_suffix_length=0)

    # use_cuda=True with CPU tensors
    with pytest.raises(ValueError, match="use_cuda=True requires CUDA tensors"):
        rosa_4bit_forward(q, k, v, use_cuda=True)


def test_rosa_4bit_forward_non_contiguous():
    q = torch.randn(2, 8, 16).transpose(0, 1).transpose(0, 1)  # Non-contiguous stride
    k = torch.randn(2, 8, 16)
    v = torch.randn(2, 8, 16)
    out = rosa_4bit_forward(q, k, v)
    assert out.shape == (2, 8, 16)
