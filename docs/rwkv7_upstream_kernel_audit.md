# Upstream RWKV-7 kernel audit

Inventory of the optimized training kernels in the pinned upstream tree,
mapped against what Experiment 0 currently uses, with an ADOPT / ADAPT /
BENCHMARK / REJECT recommendation and the evidence for each.

```text
upstream pinned commit : ec56ea2b172c065a793d25723bc03e2af1f018dd
upstream path          : external/RWKV-LM/RWKV-v7/train_temp/cuda/
our vendored kernel    : external/rosa_soft/contrib/rwkv7_legacy/csrc/cuda/
                         rwkv7_clampw.cu + rwkv7_clampw.cpp
```

Experiment 0 vendors the **oldest** recurrence kernel in that directory and
implements everything else — TimeMix mixing, projections, gates, normalization,
ChannelMix, and the output head — as ordinary PyTorch.

The first revision of this document was independently re-verified against the
sources; four claims were corrected as a result. Corrections are marked inline
and listed under [Verification history](#verification-history).

## Summary

| Operation | Current repo | Upstream candidate | Directly compatible? | Expected benefit | Verdict |
|---|---|---|---|---|---|
| Recurrence fwd/bwd | `rwkv7_clampw` | `rwkv7_clampw_v3_for_h100` | **No** — different templating and dtype surface; port required | Shared-memory preload removes repeated global reads; targets the traffic Nsight already flagged | **BENCHMARK** |
| Recurrence fwd/bwd, variant | `rwkv7_clampw` | `rwkv7_clampw_v3_for_h100_alt` | Same port cost as v3 | Same, minus the `v` preload: 20 KiB forward instead of 24 KiB | **BENCHMARK** (second arm) |
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

Both kernels additionally use per-thread `float` arrays of length `N` — `state`
in forward (`:28`), three of them in backward (`:102`). Those are compiler-managed
register or local storage rather than static shared memory, so the complete
occupancy picture needs compiled register and spill counts, not source alone.

### Signatures do not match ours

An earlier revision claimed the kernel signatures were identical to ours and
that the Python binding would need little or no change. **That is wrong.** The
logical tensor-pointer order does match, but the templating and dtype surface do
not:

```text
ours  rwkv7_clampw.cu:40,116
      template<typename F, int HEAD_SIZE, int CHUNK_LEN>
      host entry points take runtime head_size and chunk_len
      dispatches FP32, FP16, BF16
      binding expects [B,T,H,N], reads head_size = r.size(3)

v3    rwkv7_clampw_v3_for_h100.cu:23,96-97
      template<int N>  and  template<int N, int TILE>
      chunk length from the compile-time _CHUNK_LEN_ macro
      FP32 or BF16 only
      binding infers B,T,H from dims 0-2 only
```

Adopting v3 therefore means either porting its tiling into our generic
templates, or deliberately narrowing the supported surface — which would
**drop FP16**. That is real integration work, not a drop-in swap, and it is why
the verdict is now plain BENCHMARK rather than "BENCHMARK then likely ADOPT".

One porting contract to preserve: v3's forward loop always processes a full
chunk and so assumes `T` is chunk-divisible. Our binding asserts
`seq_len % 16 == 0`; that assertion must survive the port.

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

The first revision was re-verified independently against the sources. Confirmed:
the absence of Hopper machinery, the head-128 rejection, both L2Wrap rejections
including the 65536 vocab constant, the presence of backward kernels in all six
fused candidates, and numerical equivalence between our kernel and v3.

Corrected:

```text
1  signature match         claimed identical; templating and dtype surface differ
2  shared-memory footprint 24 KiB is forward only; backward is 36.25 KiB
3  fused kernel head size  only 2 of 6 encode head 64, not all 6
4  omitted files           _alt variant and rwkv7_clampw_v3.cpp binding
```

Correction 1 removed the basis for the earlier "then likely ADOPT" wording, so
the recurrence verdict is now plain BENCHMARK.
