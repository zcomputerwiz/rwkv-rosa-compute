# Experiment 0 to Experiment 1 transition plan

> **Status, 2026-08-27: historical record. Both forward-looking items below are
> closed.** This document is accurate as of the day it was written and is kept
> as a handoff record, not as current direction.
>
> - *"Finish the planned seed replication"* — **done.** The 0B seed study
>   completed at 5 (N=0) versus 4 (N=36) with complete separation; the
>   pre-registered one-sided exact Mann-Whitney rejects at `p = 0.0079`, the
>   design's minimum achievable value. The seed-42 table below is therefore now
>   one row of a completed study rather than the `n=1` Bernoulli draw it
>   correctly called itself at the time.
> - *"H2 remains the core question"* — **not answerable on the apparatus this
>   document assumes.** The generator it points at is solvable by a strictly
>   streaming interpreter, so silent tokens arrive with no computation left to
>   do; and a layer-scaling probe showed the learnability wall between `D=3` and
>   `D=4` does not move when layers are doubled, so the failure is not
>   representational depth. H2 continues on the query-deferred pointer chase
>   instead.

Reviewer handoff, 2026-08-23, with three additions from `claude-ada` marked
inline. Experiment 0 stops being the main research target: 0A validated the
apparatus, 0B produced H1 and optimization findings, and H2 remains the core
question.

## What Experiment 0 established

**0A / Llama.** Filler tokens give a large sample-efficiency benefit on Match-3
but are not necessary for asymptotic capability - N=0 nearly solves the task
given enough training (96.40% on hard negatives at 10M x 5). The benefit is
strongest on structurally hard negatives with many 2-of-3 near-matches. At
matched high budget, N=36 reached ceiling after ~10M presentations where N=0
needed ~50M.

**0B / RWKV.** The seed-42 pair shows no filler advantage. The substantive
finding is a discrete acquisition event:

```text
near_3plus AUC    ep1     ep2     ep3     ep4     ep5
N=0             0.5354  0.5523  0.5201  0.7120  0.7104
N=36            0.5389  0.5650  0.5660  0.5661  0.5662
```

N=0 crossed a hard-negative discrimination transition at epoch 4; N=36 never
did. With n=1 per arm this is one transition/no-transition Bernoulli draw.
Finish the planned seed replication; do not gate H2 on a full H1 N-sweep.

**Acquisition localization.** Reverting block 11 removes 90.5% of the acquired
signal; inserting epoch-4 block 11 into the epoch-3 body scores 20.0% *below*
baseline. Reading: a distributed representation change across the stack with a
late readout in block 11. One transition only - do not over-generalize, but keep
frequent checkpoints in H2 so future transitions can be localized.

## Before Experiment 1 production

### 1. Freeze execution identity

The current run-identity hole: a graph-visible `torch.library` replacement for
`_RWKV7ClampW` would change numerics and fusion while leaving the high-level
`run_id` **identical**. Add execution-variant fields:

```text
rwkv_kernel / backend revision
compile execution revision
grouped execution revision
```

DataLoader worker and prefetch settings stay identity-neutral. Batch size,
precision, compile, grouped execution and optimizer backend stay
identity-bearing.

> **Addition (claude-ada).** This ordering is forced, not merely preferred.
> Adding fields to the hash changes every `run_id`, including runs in flight,
> so it cannot be done during the 0B seed study. "Finish 0B seeds, then fix
> execution identity" is the only valid sequence.

### 2. Freeze the Experiment-0 stack

Run one small regression sentinel on Llama 0A and RWKV 0B, verifying loss
trajectory, accuracy/AUC, checkpoint/resume, run identity, and
throughput/VRAM. Then stop using Experiment 0 as the default systems benchmark.

### 3. Preserve acquisition-sensitive instrumentation

H2 production runs must save epoch checkpoints, periodic step checkpoints
(~2k-5k steps), optimizer/scheduler state, per-instance predictions and margins,
and eval-set fingerprints. The 0B transition was localized only to a 62,500-step
interval; that should not recur.

### 4. Keep calibration-aware metrics

Raw accuracy misled us in 0B - accuracy and AUC moved in opposite directions
three separate times. Report accuracy, balanced accuracy, prediction bias, ROC
AUC, and prediction margin. For multiclass, target log-probability and
strongest-wrong-class margin.

## Experiment 1 design

H2 needs a genuinely sequential task, not Match-3. Required: explicit dependency
depth D; the same surface length able to carry different D; state_k dependent on
state_(k-1); no easy local shortcut; an exact final answer; difficulty
independent of prompt length.

Candidate grids: `D = {1,2,4,8,16,32}`, `N = {0,1,2,4,8,16,32}`. The measurement
is accuracy/margin against required sequential depth D and available recurrent
transitions N.

> **Addition (claude-ada): the full grid is not affordable, and breadth is the
> wrong thing to buy.** Measured aggregate capacity across the three nodes is
> **4.8 runs/day** at Experiment-0 per-run budget (13-18 h per run):
>
> ```text
> full D x N grid, 3 seeds     126 runs    26.2 days
> smoke (D=3 x N=4), 1 seed     12 runs     2.5 days
> smoke, 3 seeds                36 runs     7.5 days
> ```
>
> More importantly, if H2 learning is transition-like - which is what 0B
> showed - then 3 seeds per cell is close to worthless. For the 0B study,
> perfect separation at 3 seeds per arm gives Fisher exact p = 0.10, with a
> ~30% chance of missing a Llama-strength effect. Forty-two underpowered cells
> are worth less than eight well-powered ones. Prefer
> `D = {2,8,32} x N = {0,4,32}` at 5+ seeds, and size the per-run budget during
> the smoke rather than inheriting the Experiment-0 budget.

### Mandatory pre-training gates

**Shortcut/leakage audit.** Before any GPU training, test whether labels are
predictable from sequence length, token counts, construction branch,
special-token positions, padding, first/last operation, simple metadata, or
bag-of-token features. Baselines: majority, logistic regression on token counts,
bag-of-token MLP, metadata-only classifier. If any beats chance materially, fix
the generator first.

**Tiny overfit gate.** Make RWKV overfit 256 / 1024 / 4096 examples at several
depths, with both insufficient and sufficient N. If it cannot memorize the
mapping, do not launch production.

**Phase-diagram smoke.** `D = {2,8,16} x N = {0,4,16,32}`, one seed, reduced
budget. Target a regime where difficulty varies across the grid. If everything
solves at N=0 the task is too easy; if nothing solves at N=32 it is too hard.

> **Addition (claude-ada).** The audit, the baselines and the generator are
> **CPU-only** and can run now, in parallel with the GPU-bound 0B seed study.
> They are also the gates most likely to send us back to the drawing board, so
> running them first is free de-risking.

### Controls to design in now

```text
A. N=0
B. learned scratchpad token x N
C. neutral token x N
```

`B > A` means extra silent positions help; `C > A` means extra recurrent
transitions themselves help; `B > C` means scratchpad-specific representation
contributes. Scratchpad may run first under compute pressure, but the neutral
control must not require a rewrite.

### Production protocol

Freeze generator version/hash, depth distribution, N values, budget, batch size,
optimizer/LR, precision/backend, checkpoint frequency, eval sets and seeds. At
least 3 seeds; more seeds beat longer runs if learning stays bimodal.
Pre-register success criteria before seeing results.

Strong H2 support: performance improves reproducibly with N, the benefit grows
with required depth, the effect repeats across seeds, and controls show it
tracks additional recurrent computation. Strongest form: the minimum N needed
for reliable solving increases with D.

### Keep 1A and 1B separate

**1A** trains separate models at different N - does extra recurrent compute
improve learning? **1B** trains at `N_train` and evaluates at different
`N_test` - can one fixed model exploit additional unseen recurrent transitions?
Do not conflate them.

## Systems work parked

Full RWKV H1 N-sweep, further CUDA Graph work, recurrence microoptimization,
Windows multi-process tuning, ROSA integration, adaptive compute, continuous
hidden feedback, DiffusionBlocks/local objectives, block-wise training.

One exception: after the 0B seed study is frozen, removing the RWKV graph break
via proper `torch.library` registration is worth benchmarking, because dispatch
and not recurrence is now the major performance lead. Give it a new execution
identity first.

## Order

```text
finish predeclared 0B seeds
-> write Experiment-0 conclusion record
-> fix execution-backend identity
-> freeze Experiment-0 stack
-> run small regression sentinel
-> build sequential H2 generator
-> shortcut audit
-> tiny overfit tests
-> D x N phase-diagram smoke
-> tune task once if necessary
-> freeze H2 protocol
-> 3+ seed production
-> neutral-token control
-> fixed-model N_test extrapolation
```

## The lesson

Do not analyze H2 only through final accuracy. Track when capabilities appear,
at which dependency depths, whether N changes transition probability or timing,
and whether the effect survives held-out structures. Experiment 0 showed that a
smooth-looking endpoint comparison can hide a discrete learning transition plus
calibration effects.
