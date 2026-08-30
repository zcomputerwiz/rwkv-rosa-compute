# Experiment 1: gate 0a, and the recall-capacity audit

Gate 0a asks the simplest question on the Experiment 1 ladder: at depth `D=1`,
given a start node and one selector, can the model emit that node's image under
that map? It ran, and it failed. This document records what it returned, the
tool built to diagnose it, and — as importantly — the claims that tool does
**not** support.

## 1. What gate 0a returned

Registered regime: `d_model 128`, `layers 2`, batch 64, 10 epochs, fixed;
`V = 16` nodes, `K_maps = 4`, fp32 reference kernel, no workspace; one fixed
train bank of 5,000 memories and one fixed held-out bank of 500, shared across
the three model seeds. Outcome is the **median over seeds of final-epoch
held-out accuracy**, threshold `0.95`, chance `1/V = 0.0625`.

```text
seed   final held-out
1001          0.0625
1002          0.0585
1003          0.0625
median        0.0625
```

Chance, on all three seeds. Training loss moved 2.800 to 2.653 against
`ln 16 = 2.7726`.

## 2. Two failures, and they are not the same failure

The number above is produced by **under-training**, not by memorisation. Gate
0a has no `--lr` flag and inherits `TrainConfig.learning_rate = 1e-4` from
Experiment 0, where it was tuned for a 768-wide model. At that rate the loss
barely leaves `ln 16`.

Raise the learning rate into the band the published literature uses for this
task family — 2e-4 and above — and optimisation works. What then happens is
different and is the subject of this audit: the model **memorises the training
bank instead of learning in-context retrieval**, reaching high or perfect
training accuracy while held-out accuracy stays at chance.

Both are real, and only the second is what the recall audit measures. No report
may use the second to explain the first.

## 3. The tool

[`scripts/audit_recall_capacity.py`](../scripts/audit_recall_capacity.py) sweeps
learning rate, `d_model`, layers, bank size, and the task's `M x K` shape, and
reports held-out accuracy alongside training accuracy on a probe of the training
bank. Held-out accuracy alone cannot separate "has not learned" from "has
memorised"; the pair can.

It is an analysis tool. It is not a gate, it decides nothing, and its artifacts
are estimates.

Three properties worth knowing before reading its output:

- **Both `held_out_final` and `held_out_best` are recorded, and `gap` and `z`
  derive from the final epoch**, because that is the statistic the gates use.
  Deriving them from the maximum builds a selection over 32–128 epochs into a
  number that then reads as an outcome.
- **`z` is clustered at the memory level.** `queries_per_memory` queries share
  one memory, so treating instances as independent understates the standard
  error by `sqrt(queries_per_memory)`.
- **The artifact carries its environment and repo commit.** These runs disagree
  between nodes, and a cross-node comparison that cannot see the compute
  capability or the torch and CUDA versions cannot separate architecture from
  environment.

Audit regime, distinct from the gate's in every one of these fields: batch 256,
`lr 1e-3`, AdamW `betas=(0.9, 0.95)`, weight decay 0.01, gradient clipping 1.0,
`CosineAnnealingLR(T_max=epochs)` stepped per epoch, fp32 reference kernel.

## 4. Bank size, at matched optimiser steps

Data volume and optimiser steps are confounded if epochs are held fixed, since a
larger bank then also gets more steps. Run as a 2×2 at matched steps, on
`sm_89`, `d_model 128`, `layers 2`, final-epoch held-out:

```text
                        4,992 memories            19,968 memories
2,496 steps      held 0.0647  train 0.6189   held 0.0631  train 0.0787
                 loss 1.0598                 loss 2.7651
9,984 steps      held 0.0776  train 1.0000   held 1.0000  train 1.0000
                 loss 3.4e-08                loss 2.2e-08
```

Chance is 0.0625. The bottom-left cell is the point: at 4,992 memories with
matched steps the model reaches perfect training accuracy and a loss of
`3.4e-08` while held-out sits at 0.0776, a gap of 0.92. The top-right cell shows
data alone is not sufficient either — 2,496 steps has simply not trained yet.
Both are required.

Leakage was checked before believing the 1.0000, since exact 1.0 with a
vanishing loss is the shape a leak takes: zero memories and zero queries are
shared between the banks, the most common held-out answer occurs at 0.0798
against a uniform 0.0625, and the memory space at `M=16, K=4` is `16!^4 ≈
1.9e53`. The 4,992 bank is a strict prefix of the 19,968 bank — same generator,
same seed — so these are two points on one curve differing only in size.

A third point at 79,872 memories also reaches 1.0000.

## 5. The bank a cell needs scales with its association count

Swept on `sm_75`, 32 epochs, `lr 1e-3`, final-epoch held-out:

```text
  M   K  assoc   at 4,992   at 19,968
  4   1      4     1.0000      1.0000
  4   2      8     1.0000      1.0000
  4   4     16     1.0000      1.0000
  8   1      8     1.0000      1.0000
  8   2     16     1.0000      1.0000
  8   4     32     0.2433      1.0000
 16   1     16     1.0000      1.0000
 16   2     32     0.1406      1.0000
 16   4     64     0.0603      0.9459
```

At or below 16 associations both bank sizes generalise. At 32 the small bank
fails and the large one is perfect. At 64 — the gate's own cell — the large bank
is still short of 1.0.

Note that `M` is not a clean axis: changing it also changes chance, the class
count, the input width, and so the parameter count. `params` is recorded per
cell for that reason.

## 6. What this does not establish

**It does not show the gate cannot pass at any budget.** `CosineAnnealingLR`
with `T_max=epochs` anneals the learning rate to approximately zero by the final
epoch, and held-out accuracy in the 128-epoch cell flatlines at 0.077567 over
its last eight epochs. The supportable claim is that *a 128-epoch cosine run at
`lr 1e-3` did not generalise*. A long constant-learning-rate budget is untested,
and delayed generalisation after memorisation — which requires exactly the
sustained training the anneal prevents — is not excluded. Weight decay is on at
0.01.

**It does not show the gate is device-dependent.** The same cell reaches 0.95 at
different epochs on `sm_89`, `sm_86` and `sm_75`, and on `sm_75` does not reach
it within 32 epochs. But that comparison is one model seed per node, at the
audit's regime rather than the gate's, and the nodes differ in environment as
well as architecture. Node, environment, nondeterminism and architecture remain
confounded. It is a portability flag, not a result — and it does not bear on the
gate's pass condition, which is a median over three seeds at the final epoch.

**It is not the memorisation instrument.** Gate 0c is: 512 fixed `D=4`
instances, train and evaluation identical, outcome on training-set accuracy.
Its stated licence is that a later held-out failure is then not explained by an
inability to optimise or to store. Ad hoc "can it fit a small set" controls run
here duplicate a registered gate and should not be reported as findings about
the apparatus.

**It does not import Zoology's bound.** Zoology's claim is about model
*dimension* scaling with MQAR input sequence length. This tool measures a
*data* bound. What is observed here runs mildly against a pure width reading —
`d` is held at 128 while associations go from 4 to 64, and the failure is
rescued by more memories rather than more width. Treat the parallel as an open
question.

## 7. Reproducing

```bash
python scripts/audit_recall_capacity.py --lrs 1e-3 --memories 4992 19968 --epochs 32 --out results/recall_capacity/sweep.json --label my-node
```

Memory counts must be multiples of `batch_size / gcd(batch_size, queries_per_memory)`
— 64 at the default batch 256 and 4 queries per memory. Under
`--compile --compile-backend cudagraphs` a ragged bank is refused up front
rather than raised from inside a graph replay partway through the first
evaluation.

`--fresh-memories-per-epoch` draws a new training bank every epoch, so no memory
is seen twice and memorising is impossible by construction. Generation costs
about 46 µs per instance, so the effective bank is unbounded for roughly 1% more
wall clock.
