# Length-grouped execution: measured results

Track A2. The mixed 50/50 parallel-CoT / filler batch is padded to its longest
member, and parallel CoT is a fixed length regardless of N, so at low N every
filler example is carried through a CoT-sized rectangle. This runs the same
optimizer batch as length-homogeneous subgroups instead.

What is preserved exactly: one AdamW update, one scheduler step, one global
gradient clip over the complete accumulated gradient, and a token-weighted loss.
The objective does not change. This is implementation efficiency, **not compute
matching** — the scientific budget is still the requested filler transition
count N.

```text
hardware : RTX 4060 Ti 16 GiB (Ada, sm_89)
torch    : 2.13.0+cu126
model    : 0B RWKV-7, hidden 768, 12 layers, head_dim 64, fused CUDA recurrence
precision: bf16 + torch.compile
```

## Headline

At N=0, batch 48:

```text
                 padded    grouped   speedup
steady-state     292.68     219.23    1.335x
peak memory      9.76 GiB   5.03 GiB  1.94x less
loss             10.708069  10.708069 identical
```

Compile time is reported separately and never folded into the speedup. Across
millions of steps a one-off compile is irrelevant; what would matter is
recompilation *during* training, and there is none — see below.

## Across N

The padded path costs the same at every N — the rectangle is CoT-sized
regardless — so the speedup tracks how much filler padding grouping removes.

```text
  N   padded ms   grouped ms   speedup   supervised, padded -> grouped
  0     292.68      219.23      1.335x        51.1%  ->  100.0%
 16     292.44      214.68      1.362x        57.0%  ->  100.0%
 36     292.35      222.31      1.315x        64.4%  ->  100.0%
```

The three speedups span 1.315x to 1.362x, a range comparable to the run-to-run
spread (grouped stdev is 3-4 ms). Treat grouping as worth about **1.3x at batch
48 across the whole N range**, not as something that varies systematically with
N. Peak memory is 9.76 GiB padded and 5.03 GiB grouped at every N.

That flatness is worth noting because it is not what the padding arithmetic
alone predicts: the *fraction* of wasted head projections falls steadily with N
(51.1% to 64.4% supervised), yet the speedup does not. The recurrence over the
CoT group dominates, and that work is identical at every N.

## The speedup is batch-size dependent

Grouping trades avoided work against fixed per-group launch overhead. The
avoided work scales with batch size; the overhead does not.

```text
Ada (RTX 4060 Ti)
batch 24   0.958x   grouping is SLOWER
batch 48   1.335x
batch 64   padded path does not fit in 16 GiB
```

At batch 24 on Ada, splitting one launch into two costs more than the skipped
padding saves.

**That crossover is hardware-dependent.** On Ampere (RTX 3070 Laptop, 8 GiB) the
same batch 24 gives **1.202x** — grouping wins where it lost on Ada. The trade is
fixed launch overhead against avoided compute, so a slower GPU makes the avoided
compute worth relatively more. Do not assume either the batch-48 figure or the
batch-24 crossover transfers between cards.

An earlier measurement of this at batch 64 produced "11.7x". That number was an
artifact: peak allocation reached 16.41 GiB on a 16 GiB card, so the padded path
was thrashing the allocator rather than computing. It is recorded here because
the failure mode is easy to repeat — a padded path that barely fits will make
any alternative look spectacular.

## torch.compile behaviour

Variable subgroup batch sizes were the specific risk: dynamo could recompile
per step, or exceed its recompile limit and fall back to eager silently.

Neither happens. The benchmark sets `error_on_recompile` *after* warmup and runs
further steps, so continued recompilation raises rather than being inferred from
timings:

```text
recompiles after warmup : False   (both paths)
frames compiled         : 9       (both paths)
```

This was initially verified against batches that were **not** representative.
The benchmark built each batch as its own dataset with exact 50/50 ratios, so
every batch split 24/24 and produced one constant pair of subgroup shapes — the
one condition under which a recompilation test cannot fail. Real training draws
from a single shuffled dataset, where splits are binomial:

```text
per-batch subgroup sizes, shuffled DataLoader
  CoT group   16 to 30
  filler      18 to 32
  18 distinct subgroup shape-sets per 100 batches
```

Re-run against that distribution, the conclusion holds: dynamo marks the batch
dimension dynamic during warmup, after which varying subgroup sizes cost
nothing. Both generators are now fixed to draw from one shuffled dataset.

```text
                     padded    grouped
steady-state ms      292.42     213.55    1.369x
recompiles           False      False
frames compiled          9          9
peak memory        9.76 GiB   6.13 GiB
loss             10.691746  10.691745
```

The speedup is 1.369x on realistic batches against 1.335x on the uniform ones,
so the artifact was if anything understating it. Peak grouped memory is higher
(6.13 GiB, not 5.03) because some batches split as far as 30/18.

## A pre-existing compile defect, found while benchmarking

This is the most consequential finding and it has nothing to do with grouping —
it reproduces identically on the padded path.

```text
torch._dynamo hit config.recompile_limit (8)
  function: 'forward' (src/exp0/models/rwkv.py)
  last reason: self.layer_id == 7
```

`torch.compile` treats integer `nn.Module` attributes as static, so branching on
`self.layer_id` specializes the frame once per layer value. With 12 layers that
exceeds dynamo's default limit of 8, and **the remaining layers fall back to
eager permanently** — silently, with no error and nothing in the run report.
Every compiled 0B measurement taken before this fix, including the 1.40x
recorded in #32, ran with part of the stack uncompiled.

Two semantics-preserving fixes:

- `RWKV7Block` guarded on `self.layer_id == 0 and self.ln0 is not None`. `ln0`
  is constructed if and only if `layer_id == 0`, so the comparison is redundant
  and guarding on `ln0` alone is exactly equivalent.
- `RWKV7TimeMix` guarded on `self.layer_id == 0 or v_first is None`. Here the
  comparison is **not** redundant — a caller passing `v_first` to layer 0 would
  behave differently — so instead of dropping it, `is_first_layer` is stored as
  a bool at construction. A bool has two values, so the guard collapses to two
  specializations rather than twelve, with identical behaviour for every caller.

Measured effect at N=0, batch 48:

```text
                    padded    grouped   speedup   graph breaks (padded/grouped)
baseline            381.33     299.15    1.275x      18 / 20
+ ln0 fix           355.00     284.07    1.250x      12 / 16
+ bool fix          292.68     219.23    1.335x       6 / 12
```

The dynamo fixes alone are worth **1.303x on the padded path** with no
algorithmic change at all. Loss is 10.708069 in every one of these runs.

## Track B (masked head projection) is unnecessary

Grouping does not merely reduce head-projection waste, it eliminates it. Within
a length group every sequence has the same length and no internal padding, so
every projected position is supervised. Measured on real batches:

```text
  N   supervised, padded   supervised, grouped
  0        49.6%                100.0%
  1        50.0%                100.0%
  2        50.3%                100.0%
  4        51.1%                100.0%
  8        52.6%                100.0%
 16        55.7%                100.0%
 32        61.8%                100.0%
 36        63.3%                100.0%
```

The 100% column is structural, not a property of the sampled batches. A masked
head projection would have nothing left to recover, so Track B should not be
built.

## Constraints preserved

```text
one optimizer update, one scheduler step, one global gradient clip per batch
token-weighted loss (reduction="sum" per group, single global divide)
identity-neutral when off, so existing run_ids and checkpoints stay valid
precision="fp16" refused explicitly: GradScaler must wrap every backward, and
  grouping runs one per subgroup, so the combination would silently underflow
```

Enabling grouping changes the `run_id`, because summation order differs and
moves logits by float32 epsilon. It must not be switched on partway through a
sweep.

## Where the time actually goes

Profiled with `scripts/profile_exp0_step.py` at N=0, batch 48, bf16 + compile.
Attribution is by CUDA kernel name, because `torch.compile` erases module
boundaries. Nothing landed in the `unclassified` bucket.

```text
                     padded 316.12 ms      grouped 191.19 ms
matmul / gemm            37.5%                 36.2%
triton fused             26.1%                 24.8%
elementwise / copy       16.1%                 12.5%
optimizer                 9.7%                 16.0%
rwkv recurrence           8.1%                  8.6%
cross entropy             2.2%                  1.9%
reduction / norm          0.3%                  0.1%
```

Three things follow, and they reorder the remaining work.

**Pointwise work is the largest addressable block.** `triton fused` plus
`elementwise / copy` is 37.3% of the grouped step. That is exactly the surface
the upstream fused TimeMix/ChannelMix kernels target, so Track F is where the
headroom is — and its bar is now compiled PyTorch at 1.76x, not eager.

**The optimizer is a fixed per-parameter cost.** It measures 30.53 ms padded and
30.52 ms grouped: identical, because it depends on parameter count rather than
tokens. Grouping cannot reduce it, so shrinking everything else pushed it from
9.7% to 16.0%. See the next section — it is also the cheapest win available.

**The recurrence is 8.5%, as the A/B predicted.** Both upstream v3 variants were
already rejected on measurement; this confirms the ceiling was low regardless.

### What this profile cannot tell you

Splitting the GEMM bucket into output head versus backbone linears by operand
shape **does not work**, and an earlier version of the script reported the head
at `0.00 ms` because of it: kernel-level profiler events carry no
`input_shapes`, only `aten` op events do, so nothing matched. That reads as a
finding rather than as a measurement failure, and the code was removed.

By parameter count the head is 24.58M of 109.5M linear parameters over a nearly
identical token count, so roughly 22% of the GEMM bucket, about 8% of the step.
That is an estimate. Measuring it needs an ablation on `output_vocab_size`,
not shape matching.

## Fused AdamW: 2.7x, and a protocol change

The optimizer's 30.5 ms looked far above its bandwidth bound - AdamW touches
param, grad, `exp_avg` and `exp_avg_sq`, about 1.84 GB per step at 115.04M
parameters, which is 6-9 ms at this card's achievable bandwidth. Measured with
`scripts/benchmark_optimizer_variants.py`:

```text
variant             median ms   speedup     GB/s
foreach (default)       32.01    1.000x       57
fused                   11.80    2.714x      156
single-tensor           65.69    0.487x       28
```

That is 20.2 ms off a 191.19 ms step — **10.6% of total step time** — from an
existing flag, `--fused_adamw`, which is already correctly wired to
`fused=True` and already part of the `run_id`.

It supersedes the note in `docs/experiment0_precision_and_compile.md` that fused
AdamW "measured ~2-4% and sits inside the run-to-run variability of a real run".

**It is not numerically free.** Twenty steps with identical gradients:

```text
bitwise identical : False
max abs diff      : 1.812e-05
max rel diff      : 2.495e+00   (parameters near zero inflate the ratio)
fp32 eps          : 1.192e-07
```

So it belongs with TF32 and bf16: a deliberate protocol choice, recorded in the
`run_id`, adopted at the start of a sweep and never switched on partway through.

## Priority order after profiling

```text
1  --fused_adamw                    10.6% of step, one flag, protocol change
2  Track F fused TimeMix/ChannelMix  37.3% block, must beat COMPILED PyTorch
3  larger batch                      grouping freed ~4.7 GiB; amortizes the
                                     fixed optimizer cost over more samples
4  recurrence kernels                CLOSED - 8.5%, both v3 variants rejected
5  masked head projection            CLOSED - grouping already reaches 100%
```

## CUDA graphs: rejected, and structurally so

The wall-vs-GPU gap showed 25.2% of the grouped step is not on the GPU — launch
and Python overhead, which grouping itself creates by running two subgroups per
step where the padded path runs one. The padded path's gap is only 1.8%.

`torch.compile(mode="reduce-overhead")` targets exactly that. It does not work
here.

```text
                  GPU ms    wall ms    gap
default           168.09     224.73    25.2%
reduce-overhead   234.11     811.70    71.2%
```

**3.6x slower on wall clock.** CUDA graphs capture per shape, and grouping
produces ~18 distinct subgroup shape-sets per 100 batches, so the run spends
most of its time capturing and re-capturing rather than replaying. GPU time
rises too, because capture adds copies into graph-owned static buffers.

Measured under the uniform-shape generator this looked like a 1.089x win. That
was an artifact of the same unrepresentative batches described above — one
shape, one capture, pure replay. The reversal is total, and it is the clearest
argument for fixing that generator.

There is also a correctness constraint on the pairing, independent of speed.
Grouping accumulates gradients across subgroups, and CUDA graphs own the tensors
they produce, so `zero_grad(set_to_none=True)` lets the second subgroup's
backward overwrite a graph-owned tensor from the first:

```text
RuntimeError: accessing gradient tensor output of CUDAGraphs that has been
overwritten by a subsequent run
```

Preallocated, zeroed-not-freed `.grad` buffers are required. If the two are ever
enabled together, that is a correctness requirement rather than a tuning choice.

PyTorch's own warning names the same two remedies — pad inputs to a few fixed
shapes, or set `cudagraph_skip_dynamic_graphs=True`. Bucketing subgroup batch
sizes was quantified rather than guessed:

```text
  bucket  shapes  padded work  overhead
       1      36       1.000x     0.0%
       4      12       1.064x     6.4%
       8       8       1.135x    13.5%
      16       6       1.346x    34.6%
```

It does not clear. Bucket-8 still leaves 8 distinct shapes — at dynamo's
recompile limit — while adding 13.5% GPU work. The entire prize is the 25.2%
launch gap, and GPU work is roughly 75% of wall time, so bucket-8 spends about
10% of wall to chase 25%, before graph capture overhead is counted. Bucket-4
keeps the cost low but leaves 12 shapes, which is worse than the 9 that already
triggered the collapse.

**Conclusion.** Grouping and CUDA graphs are structurally incompatible:
grouping produces variable shapes by construction and graphs require static
ones. The launch-overhead half of the 25% gap is not reachable while grouping is
on, and padding shapes to recover it costs more than it returns.

The remaining addressable share is the GPU-time half: the 27% `triton fused`
plus 14% `elementwise / copy` that Track F's fused TimeMix/ChannelMix kernels
target.

## Cumulative result

```text
padded  + foreach AdamW + compile    322.53 ms wall    1.000x
grouped + fused AdamW   + compile    224.73 ms wall    1.435x
```

Both at N=0, batch 48, bf16, on realistic shuffled batches. The padded figure is
shape-invariant — it always pads to B=48 T=136 — so the generator fix does not
move it.

## Batch size: 96 is the operating point

Grouping halved peak memory, which frees room to raise the batch. Swept at N=0,
grouped, bf16 + compile + fused AdamW:

```text
batch   throughput      peak memory       off-GPU gap
   48   210.6 samp/s    6.63 GiB (41%)       24.1%
   96   257.8 samp/s   11.18 GiB (70%)       11.0%
  144    26.9 samp/s   15.64 GiB (98%)        1.5%
  192    11.6 samp/s   19.61 GiB (123%)         --
```

**Batch 96 gives 1.224x throughput and halves the off-GPU gap**, 24.1% to 11.0%.
That is the same launch and dispatch overhead CUDA graphs failed to recover:
more tokens per launch amortizes the fixed per-launch cost and the fixed 10.8 ms
optimizer, so a larger batch reaches part of it for free.

**The capacity cliff is not gradual.** 96 to 144 is a 9.6x throughput collapse,
not a taper — once allocation approaches the card there is no graceful
degradation. Hence 70% as the operating point rather than 90%: allocator
behaviour varies batch to batch, and a configuration averaging 90% will
occasionally land past the cliff mid-run.

Peak memory is **not linear in batch size**, which matters when predicting
whether a configuration fits:

```text
                       padded    grouped
batch 24 (both cards)   6.56       3.87-4.38
batch 32 (Ampere)       8.54       6.12
batch 48 (Ada)          9.76       6.13
```

Scaling the batch-48 figure linearly predicts 6.51 GiB for batch 32; the
measured value is 8.54 GiB, a 24% underestimate that put a run into host-memory
spill on an 8 GiB card. There is a large batch-independent component, so
size configurations from a measurement at the target batch, not from a ratio.

The batch-192 row is instrumentation failure, not data: it reports GPU time
22790 ms against wall 16573 ms, a negative gap, which is impossible. Under
host-memory spill the profiler's kernel accounting stops corresponding to wall
time. The throughput number is directionally right; its components are not.

Two framings worth separating, because they point opposite ways:

```text
capacity  (VRAM occupancy)      stay well below   - the cliff above
bandwidth (memory controller)   push it up        - idle controller is waste
```

The elementwise block runs at roughly 50% of the card's 288 GB/s at batch 48,
and a larger batch moves that up. Capacity is the binding constraint here,
bandwidth is not.

**Batch size is a protocol change, not a throughput knob.** It alters gradient
noise and the optimization trajectory, so it changes the `run_id` and must not
be varied within a sweep.

## Answers to the CUDA research brief

External analysis proposed three optimizations. Two are refuted by measurement,
one targets a gap that batch sizing already closed.

### Elementwise: already at 83% of DRAM peak

The proposal was a hand-fused 6-lerp + activation + normalization kernel,
projected at -15 to -22 ms/step. Measured with Nsight Compute on the isolated
TimeMix elementwise block:

```text
dram__throughput   82-85% of peak sustained
lts__throughput    41-44%
sm__throughput     58-64%
```

The block is DRAM-bound at 83% of peak, and inductor already fuses the six
lerps together with the L2 normalize into one kernel
(`triton_per_fused_add_clamp_min_div_expand_linalg_vector_norm_mul_sub_view_0`).
Total remaining headroom is ~17%, about 8 ms — less than half the projected
saving, and only achievable at 100% of peak.

This also corrects an earlier estimate in this document. Hand-counting tensor
passes gave ~50% of peak; the measured figure is 83%. The conclusion (do not
port the upstream fused kernels) was right, but the arithmetic behind it was
not — and it nearly went the other way when the 32 MiB L2 raised the question of
whether that traffic was DRAM-served at all. It is.

### GEMM: cuBLAS wins even with autotuning forced

Inductor gates GEMM autotuning on `is_big_gpu()`, which hard-codes
`min_sms = 68`. This card has 34, so setting `max_autotune_gemm = True` is
silently ignored — the only symptom is a "Not enough SMs" warning. A first
attempt to test this was therefore void. With the gate monkeypatched so
templates are genuinely considered:

```text
                         wall ms   matmul bucket   throughput
default GEMM              371.25      133.13 ms   258.6 samp/s
autotune (gate bypassed)  364.05      133.24 ms   263.7 samp/s
```

The matmul bucket does not move. cuBLAS beats every Triton template inductor
generates for `[6720, 768] x [768, 768]` and friends.

For calibration: we achieve **32.8 TFLOP/s**. Dense BF16 peak on this card is
~88 TFLOP/s (4352 cores at 2.54 GHz, 4x FP32 with FP32 accumulate) — the
commonly quoted 176 TFLOP/s is the 2:4 sparsity figure and does not apply. So
we are at 37% of dense peak, which is unremarkable for skinny `K=768` GEMMs but
not obviously improvable with available tooling.

### Launch overhead: batch sizing got there first

A shape-keyed CUDA graph cache — capturing each of the ~18 subgroup shapes once
into a shared memory pool, rather than `mode="reduce-overhead"` re-capturing —
is a real design and correctly identifies that our rejection tested only the
automatic path. The Windows WDDM launch cost (12-25 us against 3-5 us on Linux)
is a plausible mechanism for the size of the gap.

But it was sized against batch 48, where the gap was 24%. At batch 96 the gap is
**9.8%**, because more tokens per launch amortizes the same fixed cost. The
remaining prize is roughly 36 ms, not 45-55 ms, against a substantial
implementation carrying the stable-`.grad` constraint documented above.

### Current best configuration

```text
padded  + foreach + compile, batch 48    148.8 samp/s   1.00x
grouped + fused   + compile, batch 96    263.7 samp/s   1.77x
```

## Incompatible with torch.compile on the Llama path

`--grouped_execution` and `--compile` together fail during inductor codegen for
the 0A Llama model:

```text
torch._inductor.exc.InductorError: CantSplit:
  73728*s50 + 442368 not divisible by 192*s50 + 1152
```

`s50` is a symbolic dimension. Grouping makes the subgroup batch size dynamic,
and inductor cannot split the resulting iteration ranges for this model's
shapes. Each option works alone:

```text
--compile --fused_adamw                       OK
--grouped_execution --fused_adamw             OK
--compile --grouped_execution --fused_adamw   fails in inductor codegen
```

The failure is slow — it hangs in codegen rather than erroring promptly — so a
long run started with both will appear to be warming up for minutes before
dying. Smoke-test the combination on a small `--num_samples` before committing
GPU hours to it.

This did not appear in the RWKV 0B benchmarks, where the same two options
coexist without recompiles after warmup. The difference is shape structure, not
a defect in either option: RWKV pads time to CHUNK_LEN multiples, so its
subgroup shapes are coarser, while Llama passes the raw lengths through.

Until this is resolved, treat the two as mutually exclusive on the Llama path
and pick by measurement: `torch.compile` is worth 1.31x on 0A shapes (#32),
while grouped execution's benefit on Llama has not been measured on GPU at all
- every grouped benchmark in this document is RWKV 0B.
