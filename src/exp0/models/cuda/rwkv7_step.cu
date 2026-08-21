#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

using bf16 = __nv_bfloat16;
using i64 = long long int;

constexpr float W_SCALE = -0.6065306597f;  // -exp(-0.5)

template <int N>
__launch_bounds__(N, 2) __global__ void rwkv7_step_kernel(
    const bf16* __restrict__ r,
    const bf16* __restrict__ raw_w,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ a,
    const bf16* __restrict__ b,
    float* __restrict__ state,
    bf16* __restrict__ out) {
    const int batch = blockIdx.y;
    const int head = blockIdx.x;
    const int row = threadIdx.x;
    const int vector_base = (batch * 12 + head) * N;
    const int index = vector_base + row;
    float* state_row =
        state + (i64(batch * 12 + head) * N + row) * N;

    __shared__ float shared_r[N];
    __shared__ float shared_w[N];
    __shared__ float shared_k[N];
    __shared__ float shared_a[N];
    __shared__ float shared_b[N];

    shared_r[row] = __bfloat162float(r[index]);
    const float w_sigmoid =
        1.0f / (1.0f + __expf(-__bfloat162float(raw_w[index])));
    shared_w[row] = __expf(W_SCALE * w_sigmoid);
    shared_k[row] = __bfloat162float(k[index]);
    shared_a[row] = __bfloat162float(a[index]);
    shared_b[row] = __bfloat162float(b[index]);
    __syncthreads();

    float state_a = 0.0f;
#pragma unroll
    for (int column = 0; column < N; ++column) {
        state_a += state_row[column] * shared_a[column];
    }

    const float value = __bfloat162float(v[index]);
    float output = 0.0f;
#pragma unroll
    for (int column = 0; column < N; ++column) {
        const float updated =
            state_row[column] * shared_w[column] +
            state_a * shared_b[column] + value * shared_k[column];
        state_row[column] = updated;
        output += updated * shared_r[column];
    }
    out[index] = __float2bfloat16_rn(output);
}

void rwkv7_step_cuda_forward(
    const torch::Tensor& r,
    const torch::Tensor& raw_w,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& state,
    torch::Tensor& out) {
    const auto stream = at::cuda::getCurrentCUDAStream();
    rwkv7_step_kernel<_N_><<<dim3(12, 1), dim3(_N_), 0, stream>>>(
        reinterpret_cast<const bf16*>(r.data_ptr()),
        reinterpret_cast<const bf16*>(raw_w.data_ptr()),
        reinterpret_cast<const bf16*>(k.data_ptr()),
        reinterpret_cast<const bf16*>(v.data_ptr()),
        reinterpret_cast<const bf16*>(a.data_ptr()),
        reinterpret_cast<const bf16*>(b.data_ptr()),
        state.data_ptr<float>(),
        reinterpret_cast<bf16*>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
