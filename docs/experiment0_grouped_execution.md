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
batch 24   0.958x   grouping is SLOWER
batch 48   1.335x
batch 64   padded path does not fit in 16 GiB
```

At batch 24 splitting one launch into two costs more than the skipped padding
saves. Do not assume the N=0 batch-48 figure transfers to a smaller batch.

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

Real batches produce few distinct shapes — at fixed N there are exactly two
formats, so exactly two subgroups — which is why recompilation converges:

```text
  B     T   count
 24   136       6
 24     4       6
  padded rectangle: B=48 T=136
```

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
