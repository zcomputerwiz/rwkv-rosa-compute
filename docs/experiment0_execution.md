# Experiment 0 Execution Protocol

This document is the operational companion to [`experiments.md`](experiments.md).
It defines how to run Experiment 0A/0B without silently changing initialization,
checkpoint identity, or report provenance.

## 0A — Transformer positive control

Experiment 0A uses the small Llama-style transformer with random initialization.
Random initialization is the default for `--architecture llama`; `--init random`
may also be supplied explicitly.

The repaired positive-control implementation uses:

```text
4-layer Llama-style causal transformer by default
standard Hugging Face Llama split-half RoPE (theta = 10000)
Llama backbone/head initializer range = 0.02
hard-coded tuple digit + position features
shared tuple/CoT Match-3 input projection
supervised ':' continuation-boundary token
n^2 filler positions unless N is explicitly overridden
AdamW beta = (0.9, 0.95)
5% linear warmup followed by linear LR decay
```

The separator and shared Match-3 input-feature seam are required protocol
invariants. Pre-repair runs used a positionless transformer, dropped the
supervised continuation boundary, and projected continuation tokens through an
unrelated embedding table; those runs are diagnostic artifacts rather than valid
negative Experiment 0A results.

See [`experiment0_positive_control_repair.md`](experiment0_positive_control_repair.md)
for the diagnosis, source-code cross-check, implementation plan, CoT metric
semantics, and progressive rerun procedure.

Example smoke configuration:

```bash
python scripts/run_experiment.py \
  --architecture llama \
  --init random \
  --length 12 \
  --dimension 3 \
  --num_samples 20000 \
  --val_samples 2000 \
  --epochs 5 \
  --seeds 42 43 44 \
  --batch_size 384 \
  --device cuda \
  --out_dir results/exp0
```

Before scaling length-6 or length-12 substantially, run the easy immediate
control described in the repair plan. The CPU test suite enforces a smaller
fixed-set overfit gate, but held-out generalization remains an experimental
measurement rather than a unit-test assertion.

### Training protocol provenance

The default positive-control optimizer now mirrors the authors' Match-3 runner:

```text
learning_rate    1e-4
AdamW betas      (0.9, 0.95)
warmup_fraction  0.05
lr_schedule      linear_warmup_decay
```

Immediate-answer runs retain this repository's existing immediate-control
weight-decay/gradient-clipping/epoch-multiplier behavior. Precision remains an
explicit independent setting (`fp32`, `bf16`, or `fp16`) and is always recorded.

Every optimizer/schedule field is part of `TrainConfig` and therefore the
canonical run identity. N-sweep identity also records the fixed positive-control
protocol so a future protocol change cannot silently overwrite an older sweep.

### CoT diagnostics

Do not interpret high final answer accuracy with a supplied ground-truth CoT as
evidence that the model independently learned 3SUM. Parallel CoT explicitly
contains match information.

The old ambiguous `cot_accuracy` field has been removed. Reports now distinguish:

```text
cot_answer_given_cot_accuracy
    final answer accuracy with the ground-truth CoT teacher-forced
    (leakage-aware diagnostic; not independent-computation evidence)

cot_pair_position_token_accuracy
cot_pair_position_semantic_accuracy
cot_sum_token_accuracy
cot_sum_semantic_accuracy
cot_match_index_accuracy
cot_result_semantic_accuracy
cot_result_nll
    intermediate next-token generation diagnostics
```

With vocabulary reduction enabled, exact pair/sum targets contain deliberate
randomness. The semantic variants accept any computationally equivalent reduced
output, while the exact metrics retain visibility into the sampled training
objective.

Reports also include:

```text
epoch_online_train_answer_accuracies
best_online_train_answer_accuracy
```

per seed, plus aggregate `best_online_training_answer_accuracy`. These values
come from each training batch's existing forward pass before that batch's
optimizer update. They are therefore an online fit diagnostic, not a frozen
end-of-epoch training-set evaluation.

The documented full-scale 0A protocol may use a larger training set, but the
scientific identity of every run is recorded in the generated report. Do not
rename an old report to make it stand in for a different configuration.

## 0B — Stock pretrained RWKV-7 H1 replication

A valid 0B run uses a **stock pretrained RWKV-7 x070 backbone checkpoint**.
The runner does not silently substitute a randomly initialized RWKV model.

`--architecture rwkv` therefore requires one of the following:

1. an explicit `--rwkv_checkpoint PATH`, which resolves to pretrained mode; or
2. `--init random`, which is an explicit engineering/debug run and **is not 0B**.

No environment variable or hard-coded checkpoint path is consulted. Checkpoint
identity is part of experimental provenance.

The upstream 0.1B x070 reference used by this repository has:

```text
num_hidden_layers = 12
hidden_size       = 768
intermediate_size = 3072
head_dim          = 64
num_heads         = 12
```

A corresponding run is therefore configured explicitly, for example:

```bash
python scripts/run_experiment.py \
  --architecture rwkv \
  --init pretrained \
  --rwkv_checkpoint /path/to/RWKV-x070-World-0.1B.pth \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --intermediate_size 3072 \
  --head_dim 64 \
  --length 12 \
  --dimension 3 \
  --num_samples 20000 \
  --val_samples 2000 \
  --epochs 5 \
  --seeds 42 43 44 \
  --device cuda \
  --out_dir results/exp0
```

The loader infers the checkpoint's x070 dimensions and refuses to train if they
do not match the requested model configuration. The error reports the required
checkpoint-compatible dimensions.

### What is pretrained

Experiment 0 replaces the stock language model's vocabulary interface with the
synthetic-task `InputEmbedWrapper`. Consequently:

```text
stock RWKV-7 recurrent backbone       pretrained
original LM embedding table           not used
original LM vocabulary head           not used
Experiment 0 shared input projection  randomly initialized
Experiment 0 classifier head          randomly initialized
```

The same repaired shared Match-3 feature mapping is used for Llama and RWKV so
0A/0B compare the backbone mechanisms rather than different synthetic-task
interfaces.

Reports therefore label a valid 0B initialization as:

```text
mode             = pretrained
pretrained_scope = backbone_only
task_interface   = random
```

This distinction is required when interpreting "pretrained RWKV" results.

## Checkpoint adaptation contract

The stock x070 checkpoint uses upstream names such as:

```text
blocks.0.att.*
blocks.0.ffn.*
blocks.0.ln0.*
blocks.0.ln1.*
blocks.0.ln2.*
ln_out.*
emb.weight
head.weight
```

The Experiment 0 backbone uses:

```text
layers.0.time_mix.*
layers.0.channel_mix.*
layers.0.ln0.*
layers.0.ln1.*
layers.0.ln2.*
ln_out.*
```

The adapter performs an explicit key mapping and shape validation. Only
documented source-only parameters are excluded:

- `emb.weight` and `head.weight`, because Experiment 0 supplies its own task
  interface;
- `blocks.N.ln0.*` for `N > 0`, because upstream instantiates these parameters
  but only block 0 uses `ln0`;
- first-layer value-residual parameters may be absent from stock checkpoints;
  the local first-layer copies are retained only because layer 0 never consumes
  them.

After those documented adaptations, the mapped dictionary must exactly equal
the local backbone state-dict key set and is loaded with `strict=True`.
Unknown keys, missing active backbone keys, shape mismatches, and incompatible
model dimensions are hard errors.

Checkpoint deserialization uses:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

## Checkpoint provenance

Before training, the runner resolves the checkpoint path and computes its
SHA-256. The report records:

```text
initialization mode
pretrained scope
resolved checkpoint path
checkpoint SHA-256
source checkpoint architecture
requested/target architecture
strict-load status
ignored source-only keys
retained unused target defaults, if any
```

The checkpoint SHA-256, rather than its machine-specific path, participates in
the deterministic run identity. The loader verifies that the file still has the
same SHA-256 when it is loaded for training.

## Report identity and overwrite protection

Each report includes a canonical `run_config` containing the complete model,
training, task, evaluation, and seed configuration. The report filename contains
a 16-hex-character SHA-256-derived run ID.

If the expected report path already exists, the runner skips work **only when the
stored full `run_config` exactly equals the requested configuration**. A hash
match alone is not sufficient for overwrite/skip decisions.

This prevents changes such as the following from silently reusing an older
report:

```text
hidden size
layer count
FFN/intermediate size
RWKV head dimension
Llama RoPE configuration
Llama initializer range
shared Match-3 input-feature protocol
continuation-boundary protocol
Adam betas
LR schedule / warmup
initialization mode
checkpoint SHA-256
training sample count
validation sample count
epoch count
batch size
learning rate
mixture ratios
vocabulary-reduction mode
training seeds
evaluation seed
```

## Sweeps

`scripts/sweep_n.py` forwards the initialization/checkpoint and scientific CLI
configuration into each N run. Its default learning rate matches the 0A runner
(`1e-4`), and its deterministic sweep identity also records fixed protocol
settings such as RoPE, shared feature input, Adam betas, and LR scheduling.

A sweep is stored under:

```text
<out_dir>/<architecture>_<sweep_id>/
    n0/
    n1/
    n2/
    n4/
    n8/
    n16/
    n32/
    sweep.json
```

The sweep summary records filler accuracy, online training-answer accuracy, the
leakage-aware answer-given-CoT metric, intermediate CoT semantic metrics, and
CoT result NLL. It does not emit the removed ambiguous `cot_accuracy` field.

The sweep fails if a child run completes without producing the exact report path
predicted by `run_experiment.py`; it does not select an arbitrary JSON file.

## Random RWKV debug runs

A randomly initialized RWKV run remains useful for engineering tests, but it
must be requested explicitly:

```bash
python scripts/run_experiment.py \
  --architecture rwkv \
  --init random \
  --device cpu \
  --num_samples 16 \
  --val_samples 16 \
  --epochs 1
```

Do not use such a run as the documented Experiment 0B result.

## Validation gate before a new 0A/0B run

Before starting a long GPU run, `main` should pass:

```bash
ruff check .
pytest -m "not cuda and not checkpoint and not slow and not exp0" -v
pytest -m "exp0 and not cuda and not slow" -v
```

The Experiment 0 suite includes a real runner-to-training API-seam test with
zero training epochs, exact answer-scoring regressions, deterministic data
contracts, Llama RoPE/position tests, shared-feature input equivalence,
initializer/scheduler checks, separator supervision, CoT diagnostic leakage
separation, a tiny fixed-set overfit gate, RWKV recurrence equivalence tests,
and synthetic stock-checkpoint mapping tests.
