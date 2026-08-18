import pytest
import torch


@pytest.mark.cuda
def test_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    import rosa_soft
    if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
        pytest.skip("rosa_soft CUDA extension is not compiled")

    from rosa_compute import rosa_4bit_forward
    B, H, D = 1, 192, 4
    for T in [4, 8, 16, 32]:
        q = torch.randn(B, T, H * D, device="cuda")
        k = torch.randn(B, T, H * D, device="cuda")
        v = torch.randn(B, T, H * D, device="cuda")
        out_cuda = rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=True)
        out_ref = rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=False)
        assert torch.allclose(out_cuda.cpu(), out_ref.cpu(), atol=1e-3)
