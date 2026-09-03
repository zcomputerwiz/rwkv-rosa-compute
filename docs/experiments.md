# Experiments

This document defines the research sequence for the ROSA Compute project. The project has four related but distinct hypotheses; experiments must keep them separate so that a negative result in one does not incorrectly invalidate the others.

## Research hypotheses

### H1 — Transformer-style filler computation

Can a recurrent RWKV model reproduce the test-time filler-token effect previously demonstrated in transformers on parallelizable algorithmic tasks?

This is primarily a cross-architecture replication question.

### H2 — Recurrent silent computation

Can additional recurrent state transitions improve performance on problems requiring genuinely sequential computation?

This is the project's central hypothesis.

The intended mechanism is:

```text
problem
    ↓
state transition
    ↓
<scratchpad>
    ↓
state transition
    ↓
<scratchpad>
    ↓
state transition
    ↓
answer
```

The important distinction from transformer filler-token work is that each additional token provides another **sequential recurrent state update**, rather than another parallel transformer position.

### H3 — ROSA-specific advantage

Does ROSA's discrete suffix-based memory provide an advantage for recurrent test-time computation over an otherwise comparable vanilla RWKV model?

H3 should only be evaluated after H2 has established that the recurrent mechanism itself is capable of producing a measurable compute-scaling effect.

### H4 — Adaptive compute allocation

Can the model learn to determine when additional computation is useful, using little or no scratchpad computation for easy problems and more computation for difficult problems?

This is the eventual adaptive-compute objective and should follow the fixed-budget experiments.

---

# Experiment 0 — Validate the measurement apparatus

Before interpreting a negative result from a new mechanism, establish that the training and evaluation system can reproduce a known positive result.

## 0A — Transformer positive control

Use a small transformer and reproduce the qualitative filler-token experiment from:

**Pfau, Merrill & Bowman, "Let's Think Dot by Dot: Hidden Computation in Transformer Language Models" (2024).**

Use a parallelizable synthetic task such as 3SUM.

Measure:

```text
N ∈ {0, 1, 2, 4, 8, 16, 32}
```

where `N` is the number of filler/scratchpad positions before the answer.

The purpose is not to conduct a new study of transformers. It is to verify that:

* the dataset construction is sound;
* the training signal can teach filler-token use;
* scratchpad insertion is functioning correctly;
* the evaluation harness can detect an accuracy-vs-compute relationship.

A flat result here should be treated as a problem with the experimental apparatus or training protocol before drawing conclusions about RWKV.

## 0B — RWKV H1 replication

Run the same conceptual 3SUM experiment on a stock, pretrained RWKV-7 checkpoint.

Interpretation:

```text
rising curve → RWKV reproduces the transformer-style phenomenon
flat curve   → informative H1 negative result
```

Neither result should be treated as a verdict on H2.

H1 is specifically about whether recurrent inference reproduces the **transformer filler-token effect**.

---

# Experiment 1 — H2: Sequential recurrent computation

This is the primary experiment.

> **Qwen4-Exp micro pilot status, 2026-09-03:** the borrowed four-layer
> Qwen4-Exp harness did not clear its prerequisite D=1 held-out lookup gate.
> Two independently identified apparatuses both failed at chance-level
> accuracy, so D=2 and latent-workspace cells are closed under that protocol.
> This is an apparatus result, not a negative H2 result: neither population
> varied silent recurrent transitions. See
> [`qwen4_micro_pilot.md`](qwen4_micro_pilot.md).

Use a stock, already-working RWKV-7 checkpoint before introducing ROSA.

The task should require genuinely sequential dependent computation.

Do not begin with complicated natural-language reasoning tasks. Start with synthetic problems where computational depth is explicit and controllable.

For example:

```text
initial state
→ operation 1
→ operation 2
→ operation 3
→ ...
→ answer
```

Construct families with controlled dependency depth:

```text
depth = 1
depth = 2
depth = 4
depth = 8
depth = 16
...
```

Then independently vary the silent-compute budget:

```text
N ∈ {0, 1, 2, 4, 8, 16, 32}
```

The key measurement is:

```text
accuracy
    vs.
number of silent recurrent transitions
```

and, separately:

```text
accuracy
    vs.
required sequential depth
```

The desired result is evidence that additional recurrent transitions allow the model to solve problems that exceed the computation available from a single answer-generation transition.

---

# Experiment 2 — Neutral-token control

Every H2 experiment should include a control that provides the same number of recurrent transitions without using the scratchpad protocol.

For example:

```text
Arm A:
problem
<scratchpad> × N
answer

Arm B:
problem
<neutral-token> × N
answer
```

Everything else should be matched.

This control is necessary because an additional token inherently provides another RWKV state transition.

The experiment must distinguish:

```text
additional recurrent computation
```

from:

```text
scratchpad-specific learned behavior
```

Do not interpret improvement from extra tokens alone as proof that the scratchpad mechanism is responsible.

---

# Experiment 3 — H2 generalization

Once a sequential synthetic task shows a measurable effect, test whether it generalizes beyond the exact generating procedure used for training.

Use held-out problem structures rather than merely new numerical substitutions.

Vary:

* operation types;
* sequence lengths;
* computational depth;
* irrelevant/confounding information;
* order of operations;
* numerical ranges.

The evaluation must prevent memorized templates from appearing as reasoning improvements.

---

# Experiment 4 — ROSA checkpoint baseline

After H2 is established on vanilla RWKV, return to the ROSA Compute repository and establish the pretrained ROSA-4bit model as an experimental baseline.

Before any scratchpad fine-tuning:

1. Load the pretrained ROSA-4bit checkpoint.
2. Verify checkpoint structure.
3. Verify BlinkDL reference compatibility.
4. Verify `rosa_soft` reference compatibility.
5. Verify CUDA compatibility.
6. Measure baseline task performance.

The compatibility chain must remain:

```text
BlinkDL reference
        ==
rosa_soft reference
        ==
rosa_soft CUDA
```

No scratchpad conclusions should be drawn until the implementation is known to reproduce the original ROSA semantics.

---

# Experiment 5 — H3: ROSA versus vanilla RWKV

Compare:

```text
vanilla RWKV-7
vs.
ROSA-4bit 0.1B
```

using the same sequential-computation task family and the same scratchpad training/evaluation protocol.

Keep the following controlled wherever practical:

* training examples;
* scratchpad budgets;
* evaluation problems;
* optimizer/training schedule;
* tokenization;
* reporting methodology.

The question is:

> Does ROSA's discrete suffix-based memory provide additional useful computational capability beyond the recurrent backbone alone?

Do not interpret absolute differences caused by model size, pretraining quality, or tokenizer differences as a ROSA-specific architectural result without accounting for them.

---

# Experiment 6 — Scratchpad training signal

Investigate how the model learns to use silent computation.

Do not assume ordinary supervised fine-tuning is sufficient.

At minimum compare:

### Protocol A — ordinary answer supervision

The loss is applied normally to the visible answer sequence.

### Protocol B — denser intermediate supervision

Provide explicit supervision designed to encourage useful state evolution at intermediate positions.

### Protocol C — alternative structured supervision

Explore another training objective only if the preceding experiments show that ordinary supervision fails to produce useful scratchpad behavior.

The goal is to determine whether scratchpad use is actually learned rather than merely appearing as an output-token pattern.

Record:

```text
training protocol
scratchpad budget
training loss
validation loss
accuracy
```

separately.

---

# Experiment 7 — Compute-budget scaling

For a fixed trained model, evaluate the same unseen problems with:

```text
N = 0
N = 1
N = 2
N = 4
N = 8
N = 16
N = 32
```

and later larger budgets where useful.

Produce an accuracy-vs-budget curve.

Do this separately for:

* easy problems;
* moderate problems;
* hard problems;
* sequential-depth variants.

The main question is whether additional recurrent computation provides useful scaling rather than merely a fixed improvement.

---

# Experiment 8 — H4: Adaptive compute

Once fixed-budget scaling is demonstrated, investigate whether the model can learn to allocate computation adaptively.

Desired behavior:

```text
easy problem
    → little or no scratchpad

moderate problem
    → modest scratchpad

difficult problem
    → more scratchpad
```

Do not assume that adaptive allocation will emerge automatically.

Measure both:

```text
accuracy
```

and:

```text
average compute used per problem
```

The ideal result is improved accuracy at lower average computation than a fixed maximum budget.

---

# Experiment 9 — Epistemic behavior

Evaluate the model's ability to determine whether a problem should be solved at all.

Include:

```text
fully specified problem
missing required information
contradictory premises
underdetermined problem
ambiguous terminology
false asserted conclusion
irrelevant/confounding information
```

Expected behaviors include:

```text
solve
ask for clarification
identify contradiction
state insufficiency
reject false premise
ignore irrelevant information
```

This evaluation is valuable independently of scratchpad scaling.

A model that reliably recognizes:

> "The information provided is insufficient to determine the answer"

should receive credit even when no numerical answer exists.

---

# Experiment 10 — Scratchpad usefulness versus problem difficulty

Measure whether compute demand correlates with actual task difficulty.

Construct problems where the superficial prompt length is similar but the required computation differs.

For example:

```text
short prompt / easy computation
short prompt / difficult computation
long prompt / easy computation
long prompt / difficult computation
```

The purpose is to prevent the model from simply associating prompt length with required compute.

---

# Experiment 11 — Error recovery

Include problems where an initial line of reasoning can lead to an incorrect intermediate result.

The desired behavior is:

```text
attempt
→ checkpoint
→ recognize inconsistency
→ revise
→ continue
→ answer
```

This is particularly relevant to the planned `<checkpoint>` token.

Measure:

* first-attempt error rate;
* recovery rate;
* final accuracy;
* effect of additional scratchpad budget on recovery.

---

# Experiment 12 — ROSA and recurrent state behavior

Only after the basic H2/H3 results are available, investigate whether ROSA's internal state representation changes the nature of test-time compute.

Questions include:

* Does ROSA benefit from more scratchpad transitions than vanilla RWKV?
* Does ROSA show different scaling curves for sequential versus parallelizable tasks?
* Does ROSA recover from intermediate errors differently?
* Does ROSA require fewer transitions for the same accuracy?
* Does ROSA's discrete suffix memory interact with the scratchpad token in a measurable way?

These are architecture-level questions and should not be conflated with basic proof that recurrent silent computation works.

---

# Experiment 13 — Performance and incremental runtime

Once the computational behavior is established, optimize inference.

The desired eventual runtime is:

```text
prompt prefill
      ↓
persistent recurrent state
      ↓
one incremental state update per scratchpad token
      ↓
answer
```

rather than repeatedly recomputing the full context.

Performance experiments should measure separately:

* prefill cost;
* per-scratchpad-token cost;
* answer-generation cost;
* GPU utilization;
* memory usage;
* total latency.

The existing `rosa_soft` CUDA implementation should be reused where semantically compatible.

Any new optimization must retain the reference implementation and semantic tests.

---

# Experimental Controls and Reporting

Every major experiment should report:

```text
model
checkpoint
training dataset/version
training protocol
scratchpad token definition
neutral-token control
compute budgets
evaluation dataset/version
random seeds
training steps
optimizer
learning rate
batch size
validation methodology
```

Do not report a single accuracy number without the corresponding compute budget.

Whenever practical, report:

```text
accuracy vs compute
```

rather than only final accuracy.

---

# Interpretation Rules

A negative result must be attributed only to the hypothesis actually tested.

Examples:

### Flat H1 on RWKV

Conclusion:

> RWKV did not reproduce the transformer-style filler-token effect on the tested parallelizable task.

Do **not** conclude that recurrent silent computation is ineffective.

### Flat H2 on sequential tasks

First check:

* training signal;
* neutral-token control;
* model capacity;
* task difficulty;
* scratchpad insertion;
* evaluation methodology.

Only after those controls are sound should a negative H2 result substantially lower confidence in the mechanism.

### Positive H2 on vanilla RWKV

Proceed to H3.

The question then becomes whether ROSA provides an additional advantage.

### Positive H3

Investigate whether the improvement comes from:

* ROSA's discrete memory;
* model capacity/pretraining differences;
* training differences;
* or interaction between ROSA and scratchpad computation.

### Positive H4

Investigate adaptive computation as the primary research direction.

---

# Experimental Priorities

The intended order is:

```text
1. H1 positive-control apparatus
        ↓
2. H1 on RWKV
        ↓
3. H2 on vanilla RWKV
        ↓
4. H2 generalization
        ↓
5. ROSA baseline / compatibility
        ↓
6. H3 ROSA vs vanilla RWKV
        ↓
7. training-signal studies
        ↓
8. compute-budget scaling
        ↓
9. H4 adaptive allocation
        ↓
10. incremental/performance optimization
```

The ROSA compatibility repository work may proceed in parallel as engineering infrastructure, but **ROSА-specific optimization and extensive checkpoint work should not be allowed to obscure the H2 experiment**.

---

# Core Success Criterion

The project's most important result is not simply that a model can emit:

```text
<think>
<scratchpad>
<scratchpad>
...
```

The meaningful result is:

> **Additional silent recurrent state transitions produce a reproducible improvement in performance on previously unseen problems, with the improvement scaling with the available test-time compute and exceeding the neutral-token control.**

Only once that is demonstrated does it make sense to investigate whether ROSA makes the effect stronger, cheaper, or qualitatively different.
