# Upstream RWKV-7 kernel audit

Inventory of the optimized training kernels in the pinned upstream tree,
mapped against what Experiment 0 currently uses, with an ADOPT / ADAPT /
BENCHMARK / REJECT recommendation and the evidence for each.

```text
upstream pinned commit : ec56ea2b172c065a793d25723bc03e2af1f018dd
upstream path          : RWKV-v7/train_temp/cuda/
our vendored kernel    : rwkv7_clampw.cu + rwkv7_clampw.cpp
```

Experiment 0 vendors the **oldest** recurrence kernel in that directory and
implements everything else — TimeMix mixing, projections, gates, normalization,
ChannelMix, and the output head — as ordinary PyTorch.

## Summary

| Operation | Current repo | Upstream candidate | Directly compatible? | Expected benefit | Verdict |
|---|---|---|---|---|---|
| Recurrence fwd/bwd | `rwkv7_clampw` | `rwkv7_clampw_v3_for_h100` | Signatures match exactly | Shared-memory preload removes repeated global reads; targets the traffic Nsight already flagged | **BENCHMARK** then likely ADOPT |
| Recurrence, wide head | `rwkv7_clampw` | `rwkv7_clampw128_v2` | Assumes head 128; we use 64 | None at our head size | **REJECT** |
| Recurrence, plain | `rwkv7_clampw` | `wkv7_cuda` / `wkv7_cuda_fp32` | Different op surface | Older than what we vendor | **REJECT** |
| TimeMix 6-way mixing | PyTorch | `rwkv7_tmix_mix6_bf16_v5` | bf16, fwd+bwd present | Fuses six mixes we express separately | **BENCHMARK** |
| TimeMix kk pre-norm | PyTorch | `rwkv7_tmix_kk_pre_bf16_v5` | `kHeadSize = 64` matches ours | Fuses normalization + k prep | **BENCHMARK** |
| TimeMix a-gate | PyTorch | `rwkv7_tmix_a_gate_bf16` | bf16 | Small fused gate | **BENCHMARK** |
| TimeMix v-residual gate | PyTorch | `rwkv7_tmix_vres_gate_bf16_v3` | bf16 | Fuses v_first residual + gate | **BENCHMARK** |
| TimeMix ln/rkv/xg | PyTorch | `rwkv7_tmix_lnx_rkvres_xg_bf16_v1` | bf16 | Fuses GroupNorm + output composition | **BENCHMARK** |
| ChannelMix | PyTorch | `rwkv7_cmix_bf16_v5` | bf16, fwd+bwd present | Fuses the whole ChannelMix | **BENCHMARK** |
| Head + CE | `nn.Linear` + `F.cross_entropy` | `rwkv7_head_l2wrap_ce_bf16_v4` | **No** — see below | Would fuse a 32k projection | **REJECT as-is, ADAPT the idea** |
| CE only | `F.cross_entropy` | `rwkv7_l2wrap_ce_bf16_v2` | **No** — L2Wrap | — | **REJECT** |

## The recurrence: `clampw_v3` is the headline candidate

The filename says `_for_h100`, which reads as a hard Hopper requirement. It is
not. Inspecting the source finds **no Hopper-specific machinery at all**:

```text
wgmma          absent
thread-block clusters   absent
cp.async / TMA absent
sm_90 guards   absent
__CUDA_ARCH__ gating    absent
```

What it actually does is preload the six recurrence inputs into shared memory
once per chunk instead of re-reading them from global memory per timestep:

```cuda
__shared__ float r[_CHUNK_LEN_][N];   // and w, k, v, a, b
```

That is a plain shared-memory tiling optimization, portable to any architecture
with enough shared memory. At `_CHUNK_LEN_=16`, `N=64` the footprint is
6 x 16 x 64 x 4 = 24 KiB per block, which fits Ada comfortably.

Two further points make it a strong candidate rather than a speculative one:

- **Its kernel signatures are identical to ours.** Both expose
  `forward_kernel*(int T, int H, r, w, k, v, a, b, y, s, sa)` and
  `backward_kernel*(..., dy, s, sa, dr, dw, dk, dv, da, db)`, and both are
  templated on the same `_N_` and `_CHUNK_LEN_`. The Python binding would need
  little or no change.
- **It targets the bottleneck already identified.** Codex's Nsight work reported
  uncoalesced FP32 state traffic, LG throttle, and scoreboard stalls. Preloading
  into shared memory is the direct answer to that class of stall.

It is marked BENCHMARK rather than ADOPT only because nothing has been measured
yet: the correctness A/B against the PyTorch oracle and the timing comparison at
Experiment 0 shapes both remain to be run. The name is not evidence either way.

`clampw128_v2` is rejected on a concrete ground: it is built for head size 128
and Experiment 0 uses `head_dim=64`, which the fused path already requires.

## Objective-changing features: rejected

Both head/CE kernels carry **L2Wrap**, an auxiliary term appropriate to RWKV
pretraining and inappropriate to Experiment 0:

```text
rwkv7_head_l2wrap_ce_bf16_v4.cu   7 references
rwkv7_l2wrap_ce_bf16_v2.cu       12 references
```

Adopting either would change the objective, which Track G excludes without
separate approval. A second, independent blocker in the same file:

```cuda
constexpr int64_t HEAD_L2WRAP_CE_VOCAB = 65536;
```

Experiment 0 uses `output_vocab_size = 32000` deliberately — it reproduces the
authors' loss geometry — so a kernel hardcoding 65536 is not applicable even
setting L2Wrap aside.

The *architectural* idea is still worth taking: fusing the projection with the
loss avoids materializing a full logits tensor. That is what Track B and Track H
pursue with plain `linear + cross_entropy` semantics and no auxiliary term.

## The fused TimeMix and ChannelMix kernels

Six kernels cover work Experiment 0 currently runs as ordinary PyTorch. They
share assumptions that match ours — bf16, `kHeadSize = 64`, forward and backward
kernels both present — so none is excluded on compatibility grounds.

They are all BENCHMARK rather than ADOPT for one reason: since #32,
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

Any adopted upstream source must be pinned by exact commit SHA, and the commit
inspected here is recorded at the top of this document.
