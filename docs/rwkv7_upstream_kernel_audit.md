# Upstream RWKV-7 kernel audit

Inventory of the optimized training kernels in the pinned upstream tree,
mapped against what Experiment 0 currently uses, with an ADOPT / ADAPT /
BENCHMARK / REJECT recommendation and the evidence for each.

```text
upstream pinned commit : ec56ea2b172c065a793d25723bc03e2af1f018dd
upstream path          : external/RWKV-LM/RWKV-v7/train_temp/cuda/
our kernel             : rwkv7_clampw.cu + rwkv7_clampw.cpp, from that same
                         upstream directory
loaded by              : src/exp0/models/rwkv_cuda.py:25-31, 70-72
build flags            : -D_N_=64 -D_CHUNK_LEN_=16 (rwkv_cuda.py:84-92)
dtype                  : bf16 only, enforced again at rwkv_cuda.py:234
```

Experiment 0 compiles the **oldest** recurrence kernel in that directory
directly from the pinned submodule — `rwkv_cuda.py` says it "intentionally loads
the pinned BlinkDL RWKV-7 CUDA source" — and implements everything else
(TimeMix mixing, projections, gates, normalization, ChannelMix, and the output
head) as ordinary PyTorch.

Note for anyone auditing this tree: `external/rosa_soft/contrib/rwkv7_legacy/`
also contains a `rwkv7_clampw` kernel. That one is dtype-generic
(`template<typename F, int HEAD_SIZE, int CHUNK_LEN>`) and looks like the
natural comparison target, but **it is not ours** and is referenced nowhere in
`src/`, `tests/`, or `scripts/`. Comparing against it produces a false
signature-mismatch finding; the second revision of this document made exactly
that error.

This document has been through two rounds of independent verification. Findings
are listed under [Verification history](#verification-history).

## Summary

| Operation | Current repo | Upstream candidate | Directly compatible? | Expected benefit | Verdict |
|---|---|---|---|---|---|
| Recurrence fwd/bwd | `rwkv7_clampw` | `rwkv7_clampw_v3_for_h100` | Yes — same `template<int N>`, same bf16 binding surface; backward adds a `TILE` parameter | **None measured on Ada**: 0.951-1.000x, i.e. a tie or slightly slower | **REJECT** (measured) |
| Recurrence fwd/bwd, variant | `rwkv7_clampw` | `rwkv7_clampw_v3_for_h100_alt` | Same port cost as v3 | **Slower on Ada**: 0.693-0.922x | **REJECT** (measured) |
| Recurrence, wide head | `rwkv7_clampw` | `rwkv7_clampw128_v2` | No — `static_assert(_N_ == 128)` | None at our head size | **REJECT** |
| Recurrence, plain | `rwkv7_clampw` | `wkv7_cuda` / `wkv7_cuda_fp32` | Different op surface | Older than what we vendor | **REJECT** |
| TimeMix 6-way mixing | PyTorch | `rwkv7_tmix_mix6_bf16_v5` | bf16; channel-generic; fwd+bwd present | Fuses six mixes we express separately | **BENCHMARK** |
| TimeMix kk pre-norm | PyTorch | `rwkv7_tmix_kk_pre_bf16_v5` | bf16; requires `head_size == 64` | Fuses normalization + k prep | **BENCHMARK** |
| TimeMix a-gate | PyTorch | `rwkv7_tmix_a_gate_bf16` | bf16; channel-generic | Small fused gate | **BENCHMARK** |
| TimeMix v-residual gate | PyTorch | `rwkv7_tmix_vres_gate_bf16_v3` | bf16; channel-generic | Fuses v_first residual + gate | **BENCHMARK** |
| TimeMix ln/rkv/xg | PyTorch | `rwkv7_tmix_lnx_rkvres_xg_bf16_v1` | bf16; encodes `kHeadSize = 64` | Fuses GroupNorm + output composition | **BENCHMARK** |
| ChannelMix | PyTorch | `rwkv7_cmix_bf16_v5` | bf16; channel-generic | Fuses the whole ChannelMix | **BENCHMARK** |
| Head + CE | `nn.Linear` + `F.cross_entropy` | `rwkv7_head_l2wrap_ce_bf16_v4` | **No** — see below | Would fuse a 32k projection | **REJECT as-is, ADAPT the idea** |
| CE only | `F.cross_entropy` | `rwkv7_l2wrap_ce_bf16_v2` | **No** — L2Wrap | — | **REJECT** |

## Measured outcome: both v3 variants rejected

The audit below reasoned that v3's shared-memory preload should win because it
targets the memory traffic profiling had flagged. **It does not.** Measured on
Ada (RTX 4060 Ti, sm_89) against the PyTorch oracle at the tolerances in
`tests/test_exp0_cuda.py`, forward, at real Experiment 0 subgroup shapes:

```text
                        current       v3      v3_alt
CoT group  B24 T144      0.387    1.000x      0.875x
filler     B24 T16       0.066    0.955x      0.693x
padded     B48 T144      0.665    0.951x      0.922x
```

All three are numerically correct — max absolute deviation 0.0002 against a
0.08 tolerance, so the rejection is on speed alone, and no tolerance was
loosened to reach it. Reproduce with
`scripts/benchmark_rwkv7_recurrence_variants.py`.

### Why the prediction failed, and the more useful finding

Shared-memory preloading trades global reads for `__syncthreads` barriers and a
lower block-per-SM ceiling. On Ada the reads it eliminates were already being
served by cache, so it pays the barriers and the occupancy cost for nothing.
`_alt` is worse still: it drops the `v` preload to save 4 KiB and then pays a
global load per timestep, which is the trade going the wrong way.

The larger point is that the recurrence is **not where Experiment 0 spends its
time**:

```text
recurrence forward, 12 layers      7.98 ms
full padded training step        292.68 ms
forward share                       2.7%
forward + backward share            8.2%   (backward estimated at 2x forward,
                                            not measured)
```

Even a hypothetical kernel twice as fast as the one we have would return about
4% of step time. That caps Track E regardless of which variant wins, and it
means the remaining ~92% is in the surrounding TimeMix / ChannelMix / head work
that Track F and Track I cover. Profiling that surface is worth more than any
further recurrence-kernel comparison.

## The recurrence: `clampw_v3` is the headline candidate

The filename says `_for_h100`, which reads as a hard Hopper requirement. It is
not. Inspecting the source finds **no Hopper-specific machinery at all**:

```text
wgmma                   absent
thread-block clusters   absent
cp.async / TMA          absent
sm_90 guards            absent
__CUDA_ARCH__ gating    absent
```

The only conditional compilation in the file is `#ifdef _FP32_`
(`rwkv7_clampw_v3_for_h100.cu:4-13`), and the kernels are declared as ordinary
`__global__` functions (`:23-24`, `:96-97`). Its CUDA mechanisms are shared
memory, barriers, `float4` loads, and standard math intrinsics. Nothing at
source level restricts it to Hopper or prevents compiling for Ada `sm_89`.

What it actually does is preload the recurrence inputs into shared memory once
per chunk instead of re-reading them from global memory per timestep:

```cuda
__shared__ float r[_CHUNK_LEN_][N];   // and w, k, v, a, b
```

That is a plain shared-memory tiling optimization, portable to any architecture
with enough shared memory.

### Shared-memory footprint: forward and backward differ

An earlier revision of this document quoted 24 KiB per block for the whole v3
path. That figure is the **forward only**. The backward allocates nine
`[TILE][N]` arrays plus `dSb_shared[N]` (`:104-113`), launched with `TILE=16`
(`:223`):

```text
forward   6 x 16 x 64 x 4                =  24,576 B  = 24.00 KiB
backward  9 x 16 x 64 x 4  +  64 x 4     =  37,120 B  = 36.25 KiB
```

The occupancy consequence is asymmetric and matters for planning: on a nominal
100 KiB/SM part, shared memory alone permits four concurrent forward blocks but
only two backward blocks. Training is backward-dominated, so the backward figure
is the one that governs.

It does, however, **launch without special configuration**. The per-block
default cap on Ada `sm_89` is 48 KiB, and these are static `__shared__`
declarations with no dynamic shared-memory argument at the launch site:

```text
backward static shared   37,120 B
per-block default cap    49,152 B
headroom                 12,032 B  = 11.75 KiB
```

So no `cudaFuncSetAttribute` opt-in is needed and `TILE` need not drop to 8.
That matters because a `TILE` change would alter accumulation grouping, and the
numerical equivalence below is the property that makes v3 adoptable at all.

Both kernels additionally use per-thread `float` arrays of length `N` — `state`
in forward (`:28`), three of them in backward (`:102`). Those are compiler-managed
register or local storage rather than static shared memory, so the complete
occupancy picture needs compiled register and spill counts, not source alone.

### Signatures match ours

Against the kernel we actually compile — the upstream flat-directory
`rwkv7_clampw` — the templating and binding surface line up:

```text
ours  rwkv7_clampw.cu:20   template<int N> __launch_bounds__(N,2)  forward_kernel
      rwkv7_clampw.cu:83   template<int N>                         backward_kernel
      chunk length from the compile-time _CHUNK_LEN_ macro
      bf16 throughout; binding takes (B, T, H, bf* ...)

v3    ..._v3_for_h100.cu:23   template<int N> __launch_bounds__(N,2)  forward_kernel_preload
      ..._v3_for_h100.cu:96   template<int N, int TILE>               backward_kernel_preload
      same _CHUNK_LEN_ macro, same bf16 binding parameter lists
```

The only structural difference is the extra `TILE` template parameter on v3's
backward, which is a launch-site change. `rwkv7_clampw_v3.cpp` declares the same
forward and backward parameter lists as `rwkv7_clampw.cpp`; only the symbol
names (`cuda_forward_v3`) and the `TORCH_LIBRARY` namespace differ.

Because both are bf16-only, adopting v3 costs no dtype coverage. `rwkv_cuda.py`
already converts to BF16 and enforces it in the autograd function, so there is
no FP16 path to lose.

The practical adoption route is therefore to add the v3 sources as an alternate,
flag-selected pair in the extension build and A/B them — not to restructure our
kernel. The `TORCH_LIBRARY` namespace differs, so both can register side by side
for a direct comparison.

One contract to preserve: v3's forward loop always processes a full chunk and so
assumes `T` is chunk-divisible. That already holds — the kernel we load asserts
`T % _CHUNK_LEN_ == 0` (`rwkv7_clampw.cu:174`), the binding guards it, and
`rwkv_cuda.py` pads public inputs before dispatch — but it must stay true.

The verdict is BENCHMARK rather than ADOPT because nothing has been measured,
not because integration is hard.

### The `_alt` variant

`rwkv7_clampw_v3_for_h100_alt.cu` differs from the non-alt version in exactly
one functional respect: it does not preload `v`, reading it from global memory
during each timestep's compute phase instead (`:63` against `:36,49,65` in the
non-alt file). Forward shared memory drops from 24 KiB to 20 KiB; the backward
implementations are identical.

Lower footprint could allow five concurrent forward blocks instead of four,
subject to registers and allocation granularity, but it is paid for with a
per-timestep global load. Source inspection cannot say which wins, so `_alt` is
a **second benchmark arm, not a better candidate**.

Both `.cu` files export the same `cuda_forward_v3` / `cuda_backward_v3` symbols,
so the chosen build source — not the binding — selects the implementation.

### Numerics are equivalent

This is the load-bearing property for adoption, and it holds. The recurrence
equations, accumulation precision, and accumulation order match ours:

```text
same W_SCALE = -0.6065306597f
state, saved chunk state, sa, and accumulators all remain FP32
same increasing-j accumulation order in forward and backward
neither implementation contains clamp / min / max bounds that differ
BF16 conversion points unchanged for the BF16 specialization
```

v3 stages inputs by chunk and tile; it does not reorder the per-`j` reductions
or the recurrence timesteps. Adopting it would not move the training objective.

### `clampw128_v2`

Rejected on a concrete ground: `static_assert(_N_ == 128, ...)`
(`rwkv7_clampw128_v2.cu:16`), with a second `static_assert(N == 128)` on its
backward split path (`:95`) and a 128x2 thread launch (`:238`). Experiment 0
uses `head_dim=64`. Inapplicable without redesign.

## Objective-changing features: rejected

Both head/CE kernels carry **L2Wrap**, an auxiliary term appropriate to RWKV
pretraining and inappropriate to Experiment 0. It is added to the argmax-logit
gradient:

```text
rwkv7_head_l2wrap_ce_bf16_v4.cu:145-153
rwkv7_l2wrap_ce_bf16_v2.cu:193-205
```

Adopting either would change the objective, which Track G excludes without
separate approval. A second, independent blocker in the fused head kernel:

```cuda
constexpr int64_t HEAD_L2WRAP_CE_VOCAB = 65536;   // .cu:11
```

enforced again in its binding (`.cpp:34`, *"currently expects vocab=65536"*).
Experiment 0 uses `output_vocab_size = 32000` deliberately — it reproduces the
authors' loss geometry — so a kernel hardcoding 65536 is not applicable even
setting L2Wrap aside.

The *architectural* idea is still worth taking: fusing the projection with the
loss avoids materializing a full logits tensor. That is what Track B and Track H
pursue with plain `linear + cross_entropy` semantics and no auxiliary term.

## The fused TimeMix and ChannelMix kernels

Six kernels cover work Experiment 0 currently runs as ordinary PyTorch. All six
require bf16 and all six have **both** forward and backward kernels, so none is
excluded from training use:

```text
rwkv7_tmix_mix6_bf16_v5            fwd .cu:27   bwd .cu:74    channel-generic
rwkv7_tmix_kk_pre_bf16_v5          fwd .cu:43   bwd .cu:98    requires head_size == 64
rwkv7_tmix_a_gate_bf16             fwd .cu:32   bwd .cu:67    channel-generic
rwkv7_tmix_vres_gate_bf16_v3       fwd .cu:48   bwd .cu:72    channel-generic
rwkv7_tmix_lnx_rkvres_xg_bf16_v1   fwd .cu:43   bwd .cu:107   kHeadSize = 64
rwkv7_cmix_bf16_v5                 fwd .cu:28   bwd .cu:59    channel-generic
```

An earlier revision stated that all six encode `kHeadSize = 64`. **Only two
do** — `tmix_kk_pre` and `tmix_lnx_rkvres_xg`. The other four operate on generic
`[B,T,C]` with even `C`. They remain usable in a model whose recurrence head
dimension is 64, so the BENCHMARK verdict is unchanged, but the stated reason
was wrong.

They are BENCHMARK rather than ADOPT for one reason: since #32,
`torch.compile` already fuses much of this surrounding work and delivered 1.40x
on the 0B path. The relevant question is no longer "is a fused kernel faster
than eager PyTorch" but "is it faster than compiled PyTorch", and that has not
been measured. Track I re-profiles the compiled path first, so these are
prioritized by measured share of step time rather than by expectation.

## What this does not yet establish

Nothing here has been run. This is a source-level audit: it establishes
compatibility, assumptions, and exclusions, and it rules out three candidates on
concrete grounds. Every remaining verdict is contingent on:

```text
correctness A/B against the existing PyTorch recurrence oracle
existing CUDA tolerances, not loosened to make a kernel pass
timing at Experiment 0 shapes, after length grouping
comparison against compiled PyTorch, not eager
```

Source inspection specifically cannot settle whether v3's tiling pays off on
Ada, whether `_alt`'s lower footprint beats its extra global load, or whether
the backward's 36.25 KiB constrains occupancy in practice. Those need
measurement.

Any adopted upstream source must be pinned by exact commit SHA, and the commit
inspected here is recorded at the top of this document.

## Verification history

Two rounds of independent source re-verification.

**Round 1** confirmed the absence of Hopper machinery, the head-128 rejection,
both L2Wrap rejections including the 65536 vocab constant, the presence of
backward kernels in all six fused candidates, and numerical equivalence between
our kernel and v3. It corrected four claims:

```text
1  signature match         claimed identical; said to differ      -> WRONG, see round 2
2  shared-memory footprint 24 KiB is forward only; backward 36.25 KiB
3  fused kernel head size  only 2 of 6 encode head 64, not all 6
4  omitted files           _alt variant and rwkv7_clampw_v3.cpp binding
```

**Round 2** found that correction 1 was itself wrong, and why. The comparison had
been made against `external/rosa_soft/contrib/rwkv7_legacy/`, which is not the
kernel Experiment 0 builds. Reading the loader settles it:

```text
rwkv_cuda.py:25-31   _source_dir() -> external/RWKV-LM/RWKV-v7/train_temp/cuda
rwkv_cuda.py:70-72   loads rwkv7_clampw.cpp + rwkv7_clampw.cu from there
grep rwkv7_legacy across src/ tests/ scripts/   no hits
```

Against the file we do build, the original signature claim holds: both are
`template<int N>` driven by the `_CHUNK_LEN_` macro with identical bf16 binding
parameter lists, differing only in v3's extra `TILE` template parameter on the
backward. The supposed FP16 loss was also an artifact of the wrong comparison —
our path is bf16-only by construction.

Corrections 2, 3, and 4 concern upstream files and are unaffected. Round 2 also
established the 48 KiB launchability headroom recorded above.

The recurrence verdict is BENCHMARK in both revisions, but for a different
reason: not because integration is costly, but because nothing has been measured.

Method note: round 1 was given the wrong path as a premise and reported a
mismatch consistent with it. Verifying a claim about *our* build requires reading
the loader, not just the kernel sources.
