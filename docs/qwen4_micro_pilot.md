# Qwen4-Exp micro pilot: D=1 closure and next diagnostic

## Status

**Closed at the D=1 prerequisite on 2026-09-03.** Do not run D=2, depth
selection, or a latent-workspace cell under this protocol.

This pilot tested whether an official Qwen4-Exp-style micro backbone could be
reused as a cheap host for later latent-workspace experiments. It did not test
whether extra silent recurrent transitions improve accuracy: every population
run used `num_silent=0` and `workspace=false`. Its negative result therefore
does not answer H2.

## Fixed cell

```text
architecture       qwen4_exp
variant            hybrid: 3 Gated DeltaNet layers + 1 QSA layer
hidden size        128
layers             4
parameters         2,653,816 trainable
task               pointer chase, D=1, 16 nodes, 4 maps
train bank         5,120 memories x 4 queries = 20,480 instances
held-out bank        512 memories x 4 queries =  2,048 instances
model seeds        3011, 3012, 3013
train data seed    3014, fixed across model seeds
held-out seed      3015, fixed across model seeds
budget             10 epochs, 3,200 optimizer steps
batch/precision    64 / fp32
evaluation         final-epoch held-out accuracy
workspace          disabled
```

The fixed data bank is deliberate: the three runs vary initialization and
training RNG, not the sampled task. A later task incorrectly restated the data
seeds as functions of each model seed. The execution agent checked the original
artifacts and retained the governing fixed `3014/3015` mapping; that correction
is accepted.

## Results

### Original apparatus

```text
qsa_implementation   causal-exact
gdn_chunk_policy     fixed-64

seed    final held-out accuracy
3011          0.06396484375
3012          0.07519531250
3013          0.06884765625
verdict       FAIL: all three below 0.95
```

### Optimized apparatus

```text
qsa_implementation   batched-stable-v1
gdn_chunk_policy     min-sequence-64-v1

seed    final held-out accuracy
3011          0.06689453125
3012          0.06982421875
3013          0.06982421875
verdict       FAIL: all three below 0.95
```

The two apparatuses have separate model/checkpoint identities. Their metrics
must not be pooled, and differences between their accuracies are not an
optimization effect. Each independently answers the same gate and each fails.

The original checkpoints scored only about `0.269-0.279` on their own training
bank while remaining near chance on held-out memories. The large-bank runs
therefore did not establish a fitted solution that merely failed to generalize.
By contrast, a separate 64-memory, 200-epoch learnability cell reached training
accuracy `1.0` with held-out accuracy at chance. Together these results show
that the model can memorize a small fixed bank, but they do not identify why
the registered large-bank D=1 cell fails.

## Accepted performance apparatus

PR [#98](https://github.com/zcomputerwiz/rwkv-rosa-compute/pull/98), merged as
`f989a2789690648423214797e3a38789adf1a185`, adds two explicit choices:

- `batched-stable-v1` batches QSA selection and defines stable lower-block-index
  tie behavior. It is not claimed to reproduce upstream near-tie choices.
- `min-sequence-64-v1` uses `min(64, sequence_length)` for the pinned Torch GDN
  fallback. Below length 64 it is numerically distinct and therefore versioned.

The original defaults remain omitted from resolved identity so old checkpoints
remain loadable. Cross-apparatus resume is rejected.

Measured complete-run wall times fell from `72.80-76.67` minutes to
`5.73-5.78` minutes per seed, a `12.59-13.37x` end-to-end scheduling gain. This
speedup is accepted for future Qwen4 micro work under the explicit optimized
identity. It does not alter the D=1 verdict or transfer automatically to other
shapes, kernels, devices, or architectures.

## Interpretation boundary

The supported conclusion is:

> This Qwen4-Exp micro configuration, data regime, and training protocol did
> not learn the held-out D=1 prerequisite under either registered apparatus.

Do not conclude that:

- latent workspaces or silent test-time computation fail;
- Gated DeltaNet recurrence cannot perform pointer chase;
- QSA is the cause of the failure;
- more epochs, a larger model, or another data volume would pass;
- the optimized and original accuracy differences are meaningful.

None of those interventions was tested by the population gate.

## Next step

Do not spend another GPU budget until the D=1 read path is understood. The
next task is a bounded source/CPU audit of the existing apparatus:

1. Trace `generate_dataset` through `PointerChaseDataset`, `encode_batch`, the
   Qwen input projection, final-position readout, and target construction.
2. Prove from encoded examples that the requested D=1 mapping and selector are
   present at the positions available to the answer head.
3. Check train/held-out structural parity and confirm the fixed banks differ
   only in sampled memories.
4. Establish an oracle decoder over the encoded tensors. If an oracle cannot
   recover every D=1 label, repair the apparatus before training anything.
5. Inspect QSA visibility and the all-GDN path only after the encoding proof.

If that audit passes, preregister one small calibration study rather than
tuning the failed gate in place. The calibration should distinguish failure to
fit from failure to generalize by reporting both training-bank and held-out
curves. Vary only one axis at a time (data volume, training exposure, or the
final QSA layer), retain the optimized apparatus identity, and use new artifact
paths and development seeds. An all-GDN control is warranted only if the audit
leaves QSA as a live causal explanation.

No calibration run is authorized by this document. Its design must be reviewed
before GPU execution, and a passing D=1 baseline remains mandatory before any
latent-workspace comparison.

## Evidence record

The shared immutable review record contains:

```text
RESULT_QWEN4_D1_POPULATION_GATE_COMPLETION_ADA.md
  sha256 a4366aa0... / 5343ceab... / 7fb4f5d9... for seeds 3011-3013

RESULT_QWEN4_BATCHED_QSA_GDN_CHUNK_GATE_ADA.md
  sha256 ca3a4e36083781207b6baeb6aa5b11758a2057e25bb69963fe79d34857a937f8

RESULT_QWEN4_OPTIMIZED_D1_POPULATION_ADA.md
  sha256 7e26a7622eeb1bc3eacfc4843f9dd160123073921b3f11415874f003801df051

optimized run artifacts
  seed 3011  5a477645f18a3337149107e5b8008dc0614a11b659ab5eefbb142a1be343d8d0
  seed 3012  c391c802b1038e7f986b63c5305493dba55b4c83fe6bd467c29d1c7f60feb644
  seed 3013  2f953915dd6e4a7905705cd47d5f2154c772ae50b1dd2e00e515e6145b7afc06
```
