#include <torch/extension.h>

void rwkv7_step_cuda_forward(
    const torch::Tensor& r,
    const torch::Tensor& raw_w,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& state,
    torch::Tensor& out);

void rwkv7_step_forward(
    const torch::Tensor& r,
    const torch::Tensor& raw_w,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& state,
    torch::Tensor& out) {
    TORCH_CHECK(r.is_cuda(), "RWKV-7 step tensors must be CUDA tensors");
    TORCH_CHECK(r.scalar_type() == torch::kBFloat16,
                "RWKV-7 step inputs must be BF16");
    TORCH_CHECK(state.scalar_type() == torch::kFloat32,
                "RWKV-7 step state must be FP32");
    TORCH_CHECK(r.dim() == 3 && r.size(0) == 1 && r.size(1) == 12 &&
                    r.size(2) == 64,
                "RWKV-7 step inputs must have shape [1, 12, 64]");
    TORCH_CHECK(state.dim() == 4 && state.size(0) == 1 &&
                    state.size(1) == 12 && state.size(2) == 64 &&
                    state.size(3) == 64,
                "RWKV-7 step state must have shape [1, 12, 64, 64]");
    rwkv7_step_cuda_forward(r, raw_w, k, v, a, b, state, out);
}

TORCH_LIBRARY(rwkv7_step_exp0, m) {
    m.def(
        "forward(Tensor r, Tensor raw_w, Tensor k, Tensor v, Tensor a, "
        "Tensor b, Tensor(a!) state, Tensor(b!) out) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_step_exp0, CUDA, m) {
    m.impl("forward", &rwkv7_step_forward);
}
