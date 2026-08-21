# RWKV-7 single-step prototype

This prototype targets forward-only, latency-sensitive `B=1`, `T=1` RWKV-7
transitions. It is separate from the chunked training recurrence and does not
change training or backward behavior.

## Reference state contract

`rwkv7_reference_step(r, raw_w, k, v, a, b, state)` consumes one logical
transition. Each recurrence input has shape `[B, C]`. `raw_w` is the
pre-softplus decay parameter used by the fused training kernel.

The recurrent state has the following contract:

- shape: `[B, H, N, N]`, where `H = C / N` and `N = head_dim`
- dtype: FP32
- layout: contiguous row-major; the final two axes are value-row then
  key/receptance-column
- device: the same device as all recurrence inputs
- ownership: caller-owned
- mutation: the reference API does not mutate its input state and returns a
  newly allocated next state

The reference output has shape `[B, C]` and dtype FP32. The initial state for a
new independent sequence is all zeros.

## CUDA prototype contract

`rwkv7_cuda_step` is specialized for B=1, hidden size 768, 12 heads, and head
dimension 64. Its six recurrence inputs are contiguous BF16 CUDA tensors with
shape `[1, 768]`. It accepts the same raw decay parameter as the reference API.

The CUDA API mutates its caller-owned contiguous `[1, 12, 64, 64]` FP32 state
in place and returns the same tensor as `new_state`. This avoids allocating or
copying the 192 KiB recurrent state on every transition. It allocates a BF16
`[1, 768]` output. The operator is forward-only and has no backward definition.

Unsupported batches, hidden sizes, head layouts, dtypes, or devices raise an
error. There is no fallback to the chunked recurrence. The operator is kept in
a separate extension and the existing training forward/backward kernel is
unchanged.

## RTX 3070 Laptop characterization

Measurements below use B=1, hidden size 768, 12 heads, head dimension 64, BF16
model/recurrence inputs, and FP32 recurrent state. CUDA-event results use fixed
inputs, 100 warmups, and 1,000 recurrence iterations or 50 warmups and 500
four-layer full-model iterations.

Environment:

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, compute capability 8.6, 8 GiB
- driver: 610.62
- PyTorch: 2.13.0+cu126; PyTorch CUDA runtime: 12.6
- CUDA toolkit/nvcc: 12.9 / 12.9.86
- Visual Studio 2022 Community 17.14.37531.7
- MSVC 19.44.35228; Ninja 1.13

The cold CUDA suite compiled both extensions from an empty isolated cache. It
finished with 23 passed and 2 expected platform skips. In particular, the
unchanged fused forward/backward oracle tests passed, the persistent kernel
matched 1/2/4/16/32/128/512-step chains, CUDA Graph replay matched the
reference, and full-model sequence/step execution matched at lengths
1/2/4/16/17/32. No tolerance was weakened.

### Recurrence boundary

| path | median ms | p10 / p90 ms | steps/s | kernels | D2D copies | peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| old padded eager | 0.3256 | 0.3184 / 0.3602 | 3,071 | 7 | 6 | 0.41 MiB |
| old padded graph | 0.0452 | 0.0410 / 0.0548 | 22,147 | profiler-visible 6 | 6 | graph pool |
| true step eager | 0.0947 | 0.0710 / 0.1046 | 10,561 | 1 | 0 | 0.20 MiB |
| true step graph | 0.0257 | 0.0208 / 0.0385 | 38,844 | 1 | 0 | 0.19 MiB |

The true step is 3.44x faster eager and 1.75x faster under a graph at the
recurrence boundary. The old T=1 call presents 16 physical timesteps per head:
192 physical head transitions for 12 logical head transitions. Nsight Systems
confirms that every call also launches six padding-fill kernels and six device
copies before the recurrence kernel. The true step launches one kernel and no
copies.

### Full four-layer model

| path | median ms | p10 / p90 ms | token steps/s | kernels | D2D copies | peak allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|
| old sequence eager | 8.2944 | 8.1592 / 8.7823 | 120.6 | 292 | 24 | 75.3 / 148.0 MiB |
| old sequence graph | 0.7101 | 0.7075 / 0.8989 | 1,408.2 | 288 | 24 | 90.7 / 190.0 MiB |
| persistent step eager | 6.5487 | 6.4185 / 6.9474 | 152.7 | about 259 | 8 | 81.5 / 168.0 MiB |
| persistent step graph | 0.6829 | 0.6813 / 0.6858 | 1,464.4 | about 259 | 8 | 97.8 / 210.0 MiB |

The persistent path improves eager latency by 21%, but only improves the
whole-model CUDA Graph median by 3.8%. The latter is the relevant upper bound
for a fixed-input silent-transition loop on this machine. Persistent state and
its graph pool add about 7 MiB peak allocated and 20 MiB peak reserved; the
four explicit FP32 recurrence states themselves total 0.75 MiB.

Standalone CUDA Graph component timings are intentionally non-additive because
each row pays one graph replay launch. They locate the surrounding work:

| component | median ms |
|---|---:|
| input embedding/projection | 0.0371 |
| one LayerNorm | 0.0365 |
| TimeMix input projections | 0.0680 |
| padded fused recurrence | 0.0426 |
| TimeMix post-recurrence | 0.0276 |
| ChannelMix/FFN | 0.0433 |
| one full layer | 0.1678 |
| four-layer backbone | 0.6995 |
| output head | 0.0236 |
| full model | 0.7095 |

### Nsight findings

Nsight Systems measured matched 1,000-step NVTX ranges. The old eager range was
370.6 ms and the new range was 79.2 ms. The old custom kernel itself took a
16.0 us median, versus 11.9 us for the true-step kernel; most of the eager delta
therefore comes from padding copies and host launch gaps, not arithmetic.

Nsight Compute used one matched replay of each custom kernel:

| metric | old padded T=16 kernel | true T=1 kernel |
|---|---:|---:|
| replay duration | 31.10 us | 20.67 us |
| executed instructions | 173,712 | 13,824 |
| registers/thread | 143 | 116 |
| theoretical occupancy | 25.0% | 33.3% |
| achieved occupancy | 4.16% | 4.02% |
| SM throughput | 7.72% | 2.04% |
| DRAM throughput | 1.58% | 3.29% |
| L2 hit rate | 69.05% | 88.38% |

Both launches contain only 12 blocks on a 40-SM GPU, so neither comes close to
device-wide saturation. The new kernel is not compute- or DRAM-bandwidth-bound.
Its main measured stalls are LG throttle and long scoreboard waits, and Nsight
reports uncoalesced persistent-state accesses. Those accesses are a real local
optimization opportunity, but the 3.8% full-graph result makes it low priority.

## Ranked bottlenecks and next experiments

1. **Surrounding per-layer PyTorch work and launch volume.** The four-layer
   backbone takes about 0.70 ms graphed, while replacing the recurrence changes
   full-model graph latency by only 0.027 ms. The step path still exposes about
   259 kernels. If more single-token latency is needed, prototype compilation
   or fusion around TimeMix projections, normalization, post-recurrence work,
   ChannelMix, and the two per-layer token-shift state copies.
2. **Eager host launch and chunk-padding overhead.** This is dominant when a
   graph cannot be used: seven kernels plus six copies become one kernel, and
   full-model eager latency falls 21%. Keep the true-step operator for H2 and
   require CUDA Graph replay for steady silent-compute loops.
3. **Small-grid utilization.** Twelve blocks cannot occupy 40 SMs, and achieved
   occupancy is about 4%. This is architecture-independent for B=1/H=12 but its
   absolute importance varies by GPU clocks and launch overhead. Test alternate
   row/block mappings only after surrounding-op fusion raises recurrence share.
4. **Uncoalesced FP32 state traffic.** Nsight identifies inefficient state
   loads/stores, LG throttle, and scoreboard stalls. A shared-memory tiled load
   or explicitly transposed persistent layout is a contained future experiment,
   but it must preserve the state oracle and beat the current whole-model graph.
5. **Persistent-state memory.** The state footprint is small relative to 8 GiB;
   graph-pool reservation, not the 0.75 MiB logical state, explains most of the
   measured peak increase. It is not a current capacity bottleneck.

## Decision

The dedicated incremental kernel is justified as a correctness-preserving H2
prototype and as a substantial eager-path improvement. It is not evidence for
a larger recurrence rewrite: the best matched four-layer CUDA Graph result is
only 3.8% faster end to end. Stop recurrence optimization here. The next
evidence-backed task, if H2 needs lower latency, is reducing the roughly 259
surrounding graph kernels and eight token-shift copies while keeping this state
and numerical protocol unchanged.
