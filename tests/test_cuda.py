import pytest
import torch


@pytest.mark.cuda
def test_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    import rosa_soft
    if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
        pytest.skip("rosa_soft CUDA extension is not compiled")

    from rosa_compute import blinkdl_rosa_4bit_reference, rosa_4bit_forward

    B, H, D = 1, 192, 4
    C = H * D  # 768

    for T in [4, 8, 16, 32]:
        torch.manual_seed(42)
        q = torch.randn(B, T, C)
        k = torch.randn(B, T, C)
        v = torch.randn(B, T, C)

        out_blinkdl = blinkdl_rosa_4bit_reference(q, k, v)
        out_soft_ref = rosa_4bit_forward(q, k, v, max_suffix_length=512, use_cuda=False)

        q_cuda = q.cuda()
        k_cuda = k.cuda()
        v_cuda = v.cuda()
        out_cuda = rosa_4bit_forward(q_cuda, k_cuda, v_cuda, max_suffix_length=512, use_cuda=True).cpu()

        # Compare CUDA against both rosa_soft reference and BlinkDL reference
        assert torch.allclose(out_cuda, out_soft_ref, atol=1e-3)
        assert torch.allclose(out_cuda, out_blinkdl, atol=1e-3)
