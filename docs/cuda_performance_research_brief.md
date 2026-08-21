# Research brief: bespoke CUDA optimizations for the Experiment 0 0B path

A request for targeted CUDA/PyTorch performance research. Everything below is
measured on our hardware, not assumed. The point of the brief is to avoid
re-deriving what we already know and to get specific, testable suggestions for
the parts that are still open.

## What we are running

```text
GPU        NVIDIA RTX 4060 Ti 16 GiB, Ada, sm_89, 34 SMs, ~288 GB/s peak
torch      2.13.0+cu126, Triton (triton-windows), MSVC 14.44, Windows 11
model      RWKV-7 "0B": hidden 768, 12 layers, head_dim 64, output vocab 32000
           ~115 M parameters
precision  bf16 autocast + torch.compile (inductor), fused AdamW
recurrence vendored upstream BlinkDL rwkv7_clampw.cu, bf16, _N_=64,
           _CHUNK_LEN_=16
```

Workload: a synthetic Match-3 (3SUM) task. Each optimizer batch mixes two
sequence formats — parallel chain-of-thought at T=136 and a filler format whose
length is the experiment's independent variable N (T=4 at N=0, T=40 at N=36).

We run the batch as two length-homogeneous subgroups instead of one padded
rectangle. Subgroup batch sizes are binomial around half the batch, so a
100-batch run produces ~18 distinct subgroup shapes.

## Current step profile

At N=0, batch 48, grouped, bf16 + compile + fused AdamW. Kernel-name attribution
via `torch.profiler`; nothing unclassified.

```text
wall  213-225 ms/step        (run-to-run spread is real, ~5%)
GPU   ~168 ms/step
gap   ~24% of wall not on the GPU

matmul / gemm        69.2 ms   41.3%
triton fused         45.5 ms   27.1%
elementwise / copy   23.1 ms   13.8%
rwkv recurrence      15.3 ms    9.1%
optimizer            10.8 ms    6.4%
cross entropy         3.6 ms    2.1%
reduction / norm      0.2 ms    0.1%
```

Dominant GEMM shapes, per layer, with `tokens ≈ 3360` at batch 48:

```text
TimeMix    [tokens, 768] x [768, 768]    x4  (receptance, key, value, output)
ChannelMix [tokens, 768] x [768, 3072]
           [tokens, 3072] x [3072, 768]
head       [supervised_tokens, 768] x [768, 32000]
```

Inductor reports `Not enough SMs to use max_autotune_gemm mode` on this card.

## What we have already established — please do not re-suggest these

Each of these is measured, with the evidence summarized so you can judge whether
our conclusion is sound rather than take it on trust.

**Upstream fused TimeMix/ChannelMix kernels — rejected.** The upstream tree ships
six. Reading them: `rwkv7_cmix_bf16_v5` calls `at::matmul` twice, i.e. the same
cuBLAS path PyTorch takes, with two hand-written elementwise kernels around it.
The five TimeMix kernels contain zero matmuls and are 3-4 pure elementwise CUDA
kernels each. So they cannot touch the 41% in GEMM, and their reach is only the
elementwise block, which is already at roughly 50% of peak bandwidth
(~144 GB/s estimated, same class as our hand-tuned fused AdamW at a measured
156 GB/s).

**Upstream `clampw_v3` / `v3_alt` recurrence — rejected on measurement.** Both are
numerically correct against our PyTorch oracle (max abs deviation 2e-4 against a
8e-2 tolerance) but not faster on Ada: 0.951-1.000x for v3, 0.693-0.922x for
`_alt`. Their optimization is a shared-memory preload of the six recurrence
inputs; on Ada those reads were already cache-served, so it pays barriers and
occupancy for nothing.

**CUDA graphs (`mode="reduce-overhead"`) — rejected.** 3.6x *slower*: wall goes
224.73 -> 811.70 ms. Graphs capture per shape and we present ~18 subgroup shapes,
so the run captures rather than replays. Bucketing subgroup batch sizes to reduce
shape count was quantified: bucket-8 still leaves 8 distinct shapes while adding
13.5% GPU work; bucket-4 leaves 12 shapes. Neither clears the 24% prize.

**Fused AdamW — adopted, 2.714x.** 32.01 -> 11.80 ms, 57 -> 156 GB/s. Already in.

**A dynamo specialization bug — fixed, 1.303x.** `torch.compile` was specializing
the block frame on the integer `self.layer_id`; 12 layers exceeded the recompile
limit of 8 and layers 8-11 silently ran eager. Worth flagging as a general
hazard, not just ours.

**Input pipeline — not a bottleneck.** Host-side work in the critical path totals
0.14 ms against a ~213 ms step. Collate is 112.9 ms/batch but runs in workers.

## Open questions

Ordered by the size of the block they address.

### 1. GEMM, 41% of GPU time

These are skinny GEMMs — `[3360, 768] x [768, 768]` and friends — on a 34-SM
card, in bf16.

- Is cuBLAS via `at::matmul` leaving meaningful performance on these shapes on
  sm_89, or is it already near roofline? What would you expect achievable
  TFLOP/s to be here versus what we should measure?
- `max_autotune_gemm` is disabled by inductor for insufficient SM count. Is that
  heuristic worth overriding on a 34-SM part, and if so how — CUTLASS profiler,
  `torch._inductor.config` overrides, explicit Triton GEMM templates?
- Would splitting the four `[768, 768]` TimeMix projections into one fused
  `[768, 3072]` GEMM (they share the same input up to the six lerps) be
  worthwhile, given the inputs differ per projection? Is there a standard
  formulation for "same-ish input, several projections" that we are missing?
- Does the `tokens` dimension (~3360, not a nice multiple) matter enough to be
  worth padding to a tile-friendly value?

### 2. Elementwise / triton-fused, 41% of GPU time

RWKV-7 TimeMix is elementwise-heavy: a token shift, six lerps off one shifted
tensor, two sigmoids, a tanh, an L2 normalize over head-reshaped `kk`, a
GroupNorm, and a per-head `rkv` reduction. Per layer, times twelve.

- Our estimated ~50% of peak bandwidth — is that the practical ceiling for this
  access pattern on Ada, or should a well-written kernel reach 70-80%?
- Is there a way to verify achieved bandwidth per kernel properly? We have
  Nsight Compute installed but `ncu` is not on PATH and we have not used it;
  advice on the minimal `ncu` invocation for per-kernel DRAM throughput on
  Windows would help.
- Are there known inductor limitations for this shape of graph — e.g. failing to
  fuse across the six lerps, or materializing intermediates it could keep in
  registers — that a hand-written kernel would avoid? We measured 6 graph breaks
  on the padded path and 12 on the grouped path; are those likely to be costing
  real fusion opportunities?

### 3. Launch / dispatch overhead, ~24% of wall time

Hundreds of kernel launches per step; one elementwise kernel alone runs 467
times. CUDA graphs are closed to us by dynamic shapes (see above).

- Anything between "no graphs" and "full CUDA graphs" for a workload with a
  small but non-trivial set of dynamic shapes? Partial capture of the
  static-shape subgraphs? `cudagraph_skip_dynamic_graphs`? Manual capture at a
  padded max shape with masking?
- Is the per-launch cost on Windows/WDDM materially worse than Linux, and is
  that a reason to expect this 24% to shrink on a Linux host?
- Would reducing kernel *count* — rather than launch cost — be the more
  productive direction, and if so which of the RWKV-7 elementwise ops are the
  usual suspects for over-decomposition?

### 4. Recurrence, 9% of GPU time

Chunked scan, `_CHUNK_LEN_=16`, `_N_=64`, T padded to a multiple of 16. Note the
N=0 filler subgroup has T=4 padded to 16, so it does 4x the necessary work.

- Is there a fundamentally better formulation for these shapes than the vendored
  chunked scan? We are aware of the FLA/`flash-linear-attention` family; is any
  of it a genuine improvement at T=136 and head_dim 64, or is that only a win at
  much longer sequences?
- Is a smaller `_CHUNK_LEN_` worth testing for the short-sequence subgroup
  specifically, given the padding waste, or does the chunk length interact with
  numerics in a way that would break a same-run comparison?

### 5. Occupancy and batch size

We have ~10 GiB free after grouping halved peak memory (9.76 -> 6.13 GiB at
batch 48).

- On a 34-SM card, what is the right way to reason about the batch size that
  saturates the memory controller without thrashing capacity? We are sweeping
  empirically, but a principled target would help.
- Is there a reliable signal — occupancy, achieved bandwidth, SM utilization —
  we should be watching rather than just wall-clock throughput?

## Hard constraints on any suggestion

```text
must not change the training objective or numerics beyond float32 epsilon
bf16 only; head_dim is 64; output vocab is 32000 and will not shrink
no L2Wrap or any auxiliary loss term
existing CUDA test tolerances must not be loosened to make a kernel pass
any numerical protocol change (TF32, fused optimizer, precision) is allowed but
  must be explicit, recorded in the run_id, and set at the start of a sweep
```

We would rather have three suggestions with a stated expected magnitude and a
way to falsify them than a long list of general CUDA advice. If your view is
that a block is already near its ceiling, saying so is a useful answer.
