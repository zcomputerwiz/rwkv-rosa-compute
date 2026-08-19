# Complementary Research Roadmap

This document defines research directions that extend, but do not replace, the staged hypothesis tests in [`docs/experiments.md`](experiments.md).

The existing experiment sequence remains the canonical path for establishing H1-H4. In particular, the project should first establish whether extra RWKV token-time state transitions provide useful test-time computation on controlled sequential tasks before introducing mechanisms that change the recurrence, training objective, stopping policy, or runtime.

The extensions here are organized into three lanes:

1. **Scientific mechanism** — how silent recurrent computation is represented, trained, and allocated.
2. **Architecture compatibility** — optional upstream RWKV/ROSA features that should be studied without contaminating the core H2 result.
3. **Systems/runtime** — persistent recurrent state, fast ROSA retrieval, and speculative decoding.

These lanes may proceed partly in parallel as engineering work, but scientific claims must respect the dependency gates below.

---

# 1. Preserve the causal distinction between recurrence mechanisms

Several related research areas use the word *recurrence* for materially different operations. This project must keep them separate.

## 1.1 RWKV token-time recurrence

This is the mechanism tested by H2.

```text
problem token(s)
      ↓
RWKV state transition
      ↓
<scratchpad> / neutral / think input
      ↓
RWKV state transition
      ↓
<scratchpad> / neutral / think input
      ↓
RWKV state transition
      ↓
answer
```

Each additional input position advances the RWKV recurrent state once. The model depth is unchanged.

This is the project's primary definition of **recurrent silent computation**.

## 1.2 Recurrent depth

A different family of models repeatedly applies the same neural block in depth before emitting the next token:

```text
z_0
 ↓ R
z_1
 ↓ R
z_2
 ↓ R
z_3
```

Examples include recurrent-depth / looped transformer work such as *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*.

This is relevant comparative research, but it is not the same intervention as adding RWKV token-time recurrent transitions. A positive result from one mechanism should not be attributed to the other.

## 1.3 Continuous hidden-state feedback

A third mechanism feeds a model-generated continuous hidden representation back as the next input instead of quantizing it through the vocabulary:

```text
prompt
  ↓
h_0
  ↓ project / feed back
RWKV transition
  ↓
h_1
  ↓ project / feed back
RWKV transition
  ↓
h_2
  ↓
answer
```

This is conceptually related to Coconut-style continuous thought. It still causes token-time recurrent state evolution in RWKV, but the input driving each transition is data-dependent continuous state rather than a fixed vocabulary embedding.

This distinction motivates the first extension study below.

---

# 2. Scientific mechanism lane

## Gate S0 — Establish the existing baseline first

Before interpreting any complementary mechanism, complete the relevant stages of [`docs/experiments.md`](experiments.md):

1. Experiment 0A validates the transformer filler-token apparatus.
2. Experiment 0B tests the H1 filler-token effect on a pretrained RWKV-7 checkpoint.
3. Experiment 1 tests H2 on a genuinely sequential synthetic task with controlled dependency depth.
4. Experiment 2 provides the matched neutral-token control.

The decisive H2 result is an **accuracy-vs-recurrent-transition-budget curve on a sequential task**.

The extensions below may be prototyped before H2 is complete, but they should not replace or reinterpret the baseline.

---

## Study S1 — Continuous silent-compute ablation

### Research question

If extra RWKV state transitions are useful, how much does performance depend on the representation that drives those transitions?

### Arms

Use the same trained/evaluated task instances and the same number of recurrent transitions wherever possible.

#### Arm A — Discrete learned scratchpad token

```text
problem
<scratchpad> × K
answer
```

This is the primary H2 mechanism.

#### Arm B — Neutral discrete token

```text
problem
<neutral> × K
answer
```

This is the existing control for the benefit of extra recurrence independent of a scratchpad-specific embedding.

#### Arm C — Fixed continuous pause input

Bypass vocabulary lookup for the silent positions and drive each transition with a fixed continuous vector:

```text
z_pause = 0
```

or a learned parameter:

```text
z_pause ∈ R^d
```

A learned fixed vector should be interpreted as a **soft pause/scratchpad embedding**, not as evidence of data-dependent latent reasoning.

#### Arm D — Hidden-state feedback

Feed a model-generated hidden representation back as the next silent-step input:

```text
z_{k+1} = P(h_k)
```

where `P` is identity when dimensions/interfaces permit, or a deliberately small learned projection.

This is the strongest test of continuous latent recurrence because each silent transition is driven by the model's evolving internal computation rather than the same repeated input vector.

### Experimental design

Evaluate:

```text
K ∈ {0, 1, 2, 4, 8, 16, 32}
```

First use 3SUM only as a continuity/diagnostic comparison with Experiment 0. The primary result should come from the controlled **sequential H2 task**, because 3SUM is the parallelizable H1 apparatus.

For each arm, report:

- task accuracy vs. K;
- training loss and validation loss;
- train-time silent-compute budget distribution;
- test-time silent-compute budget;
- whether the test budget is in-distribution or extrapolated;
- latency and incremental-state transitions;
- hidden-state norm/stability diagnostics;
- parameter-count differences introduced by the mechanism.

### Required controls

- Match task examples across arms.
- Match the number of RWKV token-time state transitions.
- Do not call a fixed learned `z_pause` "latent reasoning" without the hidden-state-feedback control.
- Distinguish training at fixed K from evaluating a model at unseen K.
- Test whether improvements survive held-out problem structures, not merely held-out values.

### Implementation direction

The silent budget should be a model/runtime parameter rather than a fake sequence-formatting operation. The sequence formatter should continue to represent actual discrete-token protocols only.

Potential modules:

```text
src/exp0/models/rwkv.py
src/exp0/train.py
src/exp0/evaluate.py
scripts/run_experiment.py
```

Prefer a stateful step API that can later be reused by the systems lane.

### Decision rule

If Arm D substantially improves compute scaling over Arms A-C under matched transition budgets, that is evidence that **data-dependent continuous feedback** contributes beyond merely providing more recurrent transitions.

If all arms behave similarly, the simpler interpretation is that the useful resource is primarily the additional RWKV state evolution itself.

---

## Study S2 — Budget extrapolation and fixed-budget generalization

This study extends Experiments 3 and 7.

### Research questions

1. Does a model trained at one silent-compute budget benefit from additional transitions at test time?
2. Does it degrade, saturate, or continue improving beyond the training distribution?
3. Does the answer differ between discrete filler, neutral input, fixed continuous pause, and hidden-state feedback?

### Recommended matrix

Train separate models or controlled curricula with budgets such as:

```text
K_train ∈ {0, 4, 8, variable}
```

Evaluate each at:

```text
K_test ∈ {0, 1, 2, 4, 8, 16, 32, 64}
```

where computationally practical.

Report the full train-budget × test-budget matrix, not only the best point.

### Interpretation

A model that improves only at its trained K has learned a fixed protocol. A model that continues improving at larger unseen K provides stronger evidence for scalable recurrent test-time computation.

---

## Study S3 — Adaptive compute allocation (H4)

This study implements the intent of Experiment 8 only **after fixed-budget scaling is established**.

### Research question

Can the model learn to spend more RWKV token-time recurrent transitions on harder problems and fewer on easier problems while preserving or improving accuracy?

### Mechanism discipline

The first adaptive-compute implementation should halt the **same recurrent thought loop already validated by H2**.

Do not begin by recursively reapplying `ROSABlock` or another full neural block in depth. That would change the experiment into recurrent-depth computation.

### Candidate halting methods

Use a clean, named formulation rather than an undocumented hybrid.

Two suitable starting points are:

- **Adaptive Computation Time (ACT)** — deterministic/differentiable adaptive computation.
- **PonderNet** — probabilistic halting with an explicit computation/accuracy tradeoff.

Whichever mechanism is selected, document the exact halting distribution, training loss, regularization term, and inference stopping rule.

### Difficulty controls

Do not use input length alone as the definition of problem difficulty.

Construct matched-length examples with different required dependency depths, for example:

```text
same prompt length / depth 1
same prompt length / depth 4
same prompt length / depth 8
same prompt length / depth 16
```

This prevents the halting policy from succeeding by learning a trivial prompt-length heuristic.

### Metrics

Report jointly:

```text
accuracy
average transitions used
transition-count distribution
accuracy vs. compute
compute vs. known dependency depth
```

Also compare against fixed-budget baselines with the same average and maximum compute.

### Success criterion

The strongest H4 result is not merely early stopping. It is a Pareto improvement in which adaptive allocation matches or improves accuracy while using less average recurrent computation than an appropriate fixed-budget policy.

---

## Study S4 — Training objectives for silent recurrent computation

This study belongs under Experiment 6, especially Protocol C.

### Baseline order

Always retain:

1. ordinary answer supervision;
2. denser intermediate supervision;
3. only then alternative structured objectives.

The goal is to avoid attributing a failure of supervision to a failure of recurrent computation.

### DiffusionBlocks candidate

*DiffusionBlocks* is a particularly relevant candidate because it trains residual-network blocks through diffusion-derived local denoising objectives and includes experiments on recurrent-depth language models.

However, its published recurrent-depth formulation repeatedly applies the same network **in depth**. RWKV's central mechanism here is **token-time recurrence with matrix state evolution**.

Therefore:

> Do not assume that DiffusionBlocks automatically removes BPTT for RWKV token-time recurrence.

Treat that transfer as a research problem.

Before implementation, derive and document:

- what the clean target variable `y` represents for the RWKV silent-compute problem;
- what variable receives Gaussian/noise corruption;
- whether the denoising variable is token representation, recurrent state, answer representation, or another latent;
- how noise level conditioning enters the RWKV computation;
- what is held fixed while one block/objective is optimized;
- what gradients are intentionally stopped;
- what inference procedure corresponds to the training objective.

A valid DiffusionBlocks experiment should compare against the same task/model under conventional end-to-end or truncated recurrent training and report:

```text
peak VRAM
training FLOPs / wall time
convergence
final accuracy
accuracy-vs-compute scaling
```

### Scientific gate

Do not replace the ordinary H2 training path with DiffusionBlocks before the simpler path has been measured. DiffusionBlocks should initially answer:

> Can the same useful recurrent-compute behavior be trained more efficiently or more robustly with a structured local objective?

rather than:

> Does recurrent silent computation work at all?

---

# 3. Architecture compatibility lane

## Study A1 — DeepEmbed-compatible RWKV variants

DeepEmbed-related upstream RWKV work is relevant to model capacity and deployment, but it is not required to establish H2.

### Scope rule

Do not describe a single inferred DeepEmbed implementation as "the full RWKV-8 specification."

RWKV-8 is evolving, and upstream experimental variants differ. Any compatibility work must be pinned to:

```text
upstream repository
exact commit/tag
checkpoint identifier
checkpoint hash
expected state-dict schema
```

### Implementation rule

Implement the exact DeepEmbed behavior required by a selected upstream checkpoint rather than a generic approximation such as simply multiplying FFN output by `nn.Embedding(token_ids)`.

If the motivation includes keeping the embedding table in system RAM or on SSD, implement and benchmark actual host-resident storage/prefetch behavior. Merely declaring `nn.Embedding` does not constitute CPU offloading if normal model device transfers move it to the GPU.

### Measurements

Report:

- checkpoint fidelity and strict-load behavior;
- GPU VRAM;
- host RAM;
- PCIe transfer volume;
- prefetch hit/miss behavior if applicable;
- tokens/sec;
- task accuracy before and after the architectural change.

### Relationship to H2/H3

DeepEmbed work may proceed as an engineering branch, but do not use a DeepEmbed-enabled model as the sole baseline for H2 or H3 unless there is a matched control. Otherwise model-capacity and architecture changes become confounds.

---

# 4. Systems/runtime lane

This lane can proceed substantially in parallel because it improves infrastructure rather than changing the basic scientific mechanism.

## Study R1 — Persistent incremental RWKV state API

Before speculative decoding or efficient silent-compute loops, expose an explicit incremental inference interface.

Required conceptual operations:

```text
prefill(prompt) -> state S_0
step(input, S_t) -> logits/output, S_{t+1}
clone/checkpoint state
restore/commit state
```

For batched speculative work, the API should eventually support retaining intermediate states for a candidate sequence:

```text
S_0 -> S_1 -> S_2 -> ... -> S_K
```

without recomputing the entire prompt.

### Validation

For identical inputs, incremental execution must match full-sequence execution within the expected numerical tolerance.

Record:

- prefill latency;
- per-step latency;
- state memory footprint;
- full-sequence vs. incremental equivalence.

This API also benefits Studies S1-S3.

---

## Study R2 — Optimized ROSA retrieval/index

The current `rosa_slow_ref` is a semantic oracle. It performs brute-force suffix search and should remain available for correctness testing.

Do **not** use it as the performance implementation for speculative drafting.

Build or integrate an incremental suffix index/automaton whose outputs are checked against the slow reference on randomized and adversarial sequences.

Preserve the compatibility invariant where applicable:

```text
BlinkDL semantic reference
        ==
optimized reference/index
        ==
rosa_soft reference
        ==
rosa_soft CUDA
```

Performance claims must describe the complexity and measured behavior of the actual optimized implementation rather than attributing the oracle's semantics to an assumed O(1) implementation.

---

## Study R3 — ROSA retrieval-based speculative decoding

### Research question

Can ROSA's discrete suffix/retrieval structure act as a cheap draft generator for speculative decoding of an RWKV/ROSA target model?

This is related to retrieval-based speculative decoding such as REST, but ROSA provides a model-native discrete retrieval mechanism worth testing independently.

### Required inference semantics

For a proposed draft:

```text
d_1, d_2, ... d_K
```

the target recurrent model must advance causally through those tokens. A single batched/kernel invocation may make this efficient, but the recurrent dependencies do not become mathematically parallel token computations.

The runtime therefore needs transactional state handling:

```text
save S_0
  ↓
advance target through d_1 ... d_K
  ↓
find accepted prefix m
  ↓
commit S_m
  ↓
discard states after m
```

### Baselines

Compare at minimum against:

1. ordinary autoregressive RWKV decoding;
2. a simple n-gram/retrieval drafter;
3. a REST-style retrieval baseline where practical;
4. ROSA-derived drafting.

### Metrics

Report:

```text
target tokens/sec
end-to-end latency
accepted tokens per target invocation
acceptance rate by draft position
CPU utilization
GPU utilization
PCIe traffic
state-copy overhead
retrieval/index overhead
```

Break results out by data type:

- repetitive synthetic data;
- structured algorithmic data;
- code/text where appropriate;
- low-repetition controls.

A ROSA drafter should not be considered successful solely because it performs well on artificially repetitive data.

---

# 5. Dependency graph and recommended order

The recommended scientific order is:

```text
Existing Experiment 0 apparatus
        ↓
Existing H2 fixed-budget sequential task
        ↓
Existing neutral-token control
        ↓
S1 continuous silent-compute ablation
        ↓
S2 budget extrapolation/generalization
        ↓
S3 adaptive compute / H4
        ↓
S4 alternative training objectives
```

The architecture lane is mostly independent:

```text
pin exact upstream checkpoint/spec
        ↓
A1 DeepEmbed compatibility
        ↓
matched architecture experiments if scientifically useful
```

The systems lane can proceed in parallel:

```text
R1 persistent incremental state
        ↓
R2 optimized ROSA retrieval/index
        ↓
R3 speculative ROSA drafting
```

The most useful cross-lane dependency is R1: a correct incremental state API directly improves the scientific silent-compute experiments and is also required for efficient speculative decoding.

---

# 6. Project-wide interpretation rules for complementary work

## 6.1 Do not change two compute mechanisms at once

For example, do not compare:

```text
baseline discrete RWKV
vs.
ROSA + hidden-state feedback + adaptive halting + DiffusionBlocks
```

and attribute the difference to any one mechanism.

Introduce one intervention at a time with matched controls.

## 6.2 Report transition budget separately from token count

For discrete filler these may coincide. For continuous or optimized execution they may not.

Record both:

```text
visible/generated tokens
RWKV recurrent state transitions
```

## 6.3 Distinguish fixed continuous input from continuous reasoning

A repeated learned vector is a useful mechanism, but it is functionally close to a learned soft pause token. Hidden-state feedback or another data-dependent latent input is required before making a stronger continuous-reasoning claim.

## 6.4 Keep recurrent depth separate from token-time recurrence

Recurrent-depth results are valuable comparisons and may inspire training methods, but they answer a different architectural question.

## 6.5 Treat runtime speedups as end-to-end claims

Report host work, GPU work, transfers, state-copy cost, and synchronization. A faster retrieval primitive does not imply faster generation if state management or target verification dominates.

## 6.6 Preserve negative results

A flat result for one extension should narrow only that mechanism.

Examples:

- fixed soft pause fails, hidden-state feedback succeeds → transition input matters;
- hidden-state feedback fails while discrete filler succeeds → continuous feedback is not automatically beneficial;
- adaptive halting fails after fixed-budget scaling succeeds → H4/training-policy problem, not an H2 failure;
- DiffusionBlocks underperforms conventional training → local-objective transfer failed, not recurrent computation itself;
- ROSA drafting has low acceptance → systems/retrieval result, not evidence against H3.

---

# 7. Reproducibility requirements

Every complementary experiment should record the existing fields required by [`docs/experiments.md`](experiments.md) plus mechanism-specific provenance.

At minimum:

```text
repository commit
model architecture
checkpoint identifier and SHA-256
initialization mode
training dataset/version
training protocol
silent-compute mechanism
training compute budget(s)
evaluation compute budget(s)
number of recurrent state transitions
visible token count
random seeds
evaluation seed
optimizer and learning rate
batch size
validation methodology
```

For continuous mechanisms additionally record:

```text
pause-vector initialization
whether pause vector is trainable
feedback projection definition
feedback detach/gradient behavior
hidden-state normalization, if any
```

For adaptive compute additionally record:

```text
halting algorithm
halting regularizer/prior
maximum budget
stopping threshold or sampling rule
per-example transition counts
```

For DiffusionBlocks-style work additionally record:

```text
noise distribution
noise range
block partitioning
noise conditioning
clean target definition
gradient-stop boundaries
training and inference equations
```

For speculative decoding additionally record:

```text
draft source
maximum draft length
verification algorithm
state checkpoint/rollback method
acceptance criterion
```

---

# 8. Near-term milestones

## Milestone M0 — Complete apparatus repair and Experiment 0

No complementary mechanism should delay a clean 0A/0B result.

## Milestone M1 — Establish sequential H2

Produce the first trustworthy accuracy-vs-RWKV-transition-budget curve on a controlled sequential task with a neutral-token control.

## Milestone M2 — Add stateful incremental RWKV execution

Implement R1 and prove equivalence with full-sequence execution.

This is the preferred first engineering extension because it supports both science and performance work.

## Milestone M3 — Run the four-arm silent-compute study

Compare:

```text
discrete scratchpad
neutral token
fixed/learned continuous pause
hidden-state feedback
```

under matched transition budgets.

## Milestone M4 — Test budget extrapolation

Determine whether additional test-time recurrent transitions continue to help outside the trained budget.

## Milestone M5 — Adaptive compute

Only after M3-M4 show a useful fixed-budget curve, implement ACT or PonderNet-style halting on the same recurrent-thought loop.

## Milestone M6 — Structured-training study

Formulate and test a DiffusionBlocks-inspired objective as an alternative training protocol, not as a replacement for the baseline.

## Milestone M7 — ROSA speculative-decoding prototype

After R1 and R2, compare ROSA retrieval drafting against ordinary autoregressive decoding and simple retrieval baselines.

## Milestone M8 — DeepEmbed compatibility as needed

Implement only when a concrete upstream checkpoint/model target makes it experimentally useful.

---

# 9. References and adjacent work

These references motivate the extensions but do not by themselves establish that the mechanisms transfer to RWKV token-time recurrence.

- Pfau, Merrill & Bowman (2024), **Let's Think Dot by Dot: Hidden Computation in Transformer Language Models**. arXiv:2404.15758. https://arxiv.org/abs/2404.15758
- Hao et al. (2024), **Training Large Language Models to Reason in a Continuous Latent Space (Coconut)**. arXiv:2412.06769. https://arxiv.org/abs/2412.06769
- Graves (2016), **Adaptive Computation Time for Recurrent Neural Networks**. arXiv:1603.08983. https://arxiv.org/abs/1603.08983
- Banino, Balaguer & Blundell (2021), **PonderNet: Learning to Ponder**. arXiv:2107.05407. https://arxiv.org/abs/2107.05407
- Geiping et al. (2025), **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**. arXiv:2502.05171. https://arxiv.org/abs/2502.05171
- Shing et al., **DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation**. arXiv:2506.14202. https://arxiv.org/abs/2506.14202
- He et al. (2024), **REST: Retrieval-Based Speculative Decoding**. NAACL 2024. https://aclanthology.org/2024.naacl-long.88/
- BlinkDL RWKV-LM upstream research and RWKV-8 notes. https://github.com/BlinkDL/RWKV-LM

---

# Core objective

The central question remains deliberately simple:

> Can useful computation be bought at test time by advancing a recurrent model's internal state through additional transitions?

The complementary roadmap should deepen that result in stages:

```text
Does extra recurrence help?
        ↓
What input should drive those transitions?
        ↓
Does useful compute extrapolate to larger budgets?
        ↓
Can the model decide how much compute to use?
        ↓
Can that behavior be trained more efficiently?
        ↓
Can ROSA and recurrent-state systems make it faster in practice?
```

Keeping those questions separate is more important than implementing all of the mechanisms quickly.