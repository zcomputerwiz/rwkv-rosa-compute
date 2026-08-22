# Experiment 0B: RWKV-7 N=0 on the structural challenge set

The 0B RWKV-7 N=0 arm evaluated on the same frozen 6-stratum challenge set used
for the 0A Llama arms. This is the control half of a pair; the matching N=36 arm
runs on the Ampere laptop and is not yet available, so nothing here speaks to
the filler-token hypothesis on its own.

```text
challenge set   results/exp0_structural/structural_challenge_20260821.json
challenge_id    e06f92897411fe2e
content_sha256  bef50bba1c80600de6885bf60ef5f9fdfed1b37135715dce0b938cee6a1cb21b
evaluation      results/exp0_0b_structural/eval_rwkv_n0_seed_42.json
run_id          c968fce9af66aa32, seed 42
protocol        RWKV-7 768h/12L head_dim 64, bf16, compile + grouped + fused
                AdamW, batch 32, 2M x 5 epochs, no early stop triggered
```

The `content_sha256` matches the value recorded by all six Llama evaluations,
so this is the identical instance set and the comparison below is like-for-like.

## Results

```text
arm                  canon   chall    bias      pos corr_pos  near_0  near_1  near_2 near_3+
Llama N=0  s43      84.35%  78.03%  +14.1%   88.30%  88.00%  98.60%  86.30%  64.70%  42.30%
Llama N=0  s44      86.90%  83.12%   +9.8%   90.50%  88.20%  99.70%  90.40%  76.60%  53.30%
Llama N=0  s45      70.55%  65.00%  +25.4%   84.90%  86.20%  93.70%  64.90%  39.00%  21.30%
Llama N=36 s43      83.45%  80.80%  +11.8%   89.90%  87.90%  99.40%  89.70%  70.30%  47.60%
Llama N=36 s44     100.00% 100.00%   +0.0%  100.00% 100.00% 100.00% 100.00% 100.00% 100.00%
Llama N=36 s45      78.75%  73.72%  +17.2%   86.30%  86.60%  98.30%  80.60%  56.90%  33.60%
Llama N=0 10M       99.30%  98.85%   +0.2%   98.80%  98.40% 100.00%  99.90%  99.60%  96.40%
RWKV  N=0  s42      66.10%  60.60%  +27.0%   81.60%  81.30%  64.90%  48.20%  43.10%  44.50%
```

`bias` is the challenge-set True-prediction bias: predicted-true minus
actual-true, as a percentage of the 6000 instances.

## The accuracy profile looks flat, but the discrimination is graded

At epoch 5 the negative-stratum accuracies (64.9 / 48.2 / 43.1 / 44.5) have far
less spread than any Llama arm, whose gradients span 46-72 points. It is
tempting to read that as "the model is not doing graded rejection". That
reading is wrong, and the per-instance logits show why.

Accuracy confounds discrimination with calibration. AUC does not - it is
threshold-free, so a miscalibrated model with real signal still scores above
0.5. Computed against the positive instances, per epoch:

```text
          bias   bal.acc  AUC all    near_0  near_1  near_2  near_3+
epoch 1  +49.2%  55.84%   0.5924    0.6677  0.5943  0.5723   0.5354
epoch 2  +49.2%  58.39%   0.6255    0.7549  0.6114  0.5837   0.5523
epoch 3  +47.3%  60.21%   0.6497    0.8216  0.6507  0.6065   0.5201
epoch 4  +34.2%  62.22%   0.7265    0.7776  0.7220  0.6944   0.7120
epoch 5  +27.0%  65.81%   0.7500    0.8314  0.7496  0.7088   0.7104
```

At epoch 5 the strata run 0.8314 / 0.7496 / 0.7088 / 0.7104 - well above chance
and ordered by structural difficulty. The model **is** discriminating in a
graded way. The flat accuracy profile is a symptom of the +27.0% True-bias
compressing those differences, not of absent signal.

## The training trajectory, and an acquisition event at epoch 4

All five epoch checkpoints were evaluated on the same frozen set:

```text
epoch  canon   chall    pos  corr_pos  near_0  near_1  near_2  near_3+
  1    56.85%  44.33%  91.30   89.40   33.20   20.20   17.80    14.10
  2    58.35%  46.62%  93.70   93.70   42.90   20.90   14.70    13.80
  3    60.35%  48.87%  94.00   94.50   54.60   21.60   15.70    12.80
  4    63.05%  55.02%  84.00   83.70   49.90   38.80   34.70    39.00
  5    66.10%  60.60%  81.60   81.30   64.90   48.20   43.10    44.50
```

Through epochs 1-3 the model learns to reject only the easiest negatives:
`near_0` accuracy climbs 33.2 -> 54.6 while the three harder strata stay flat or
decline (`near_3plus` goes 14.1 -> 13.8 -> 12.8). This is the Llama shape
beginning to form - easy negatives first.

Epoch 3 to 4 is a regime change. Positive accuracy **drops** (94.0 -> 84.0,
`corrupted_arm_surviving_positive` 94.5 -> 83.7) while every hard negative jumps
(`near_3plus` 12.8 -> 39.0). Falling positives with rising negatives is the
signature of a decision-threshold shift, and the training-side True-bias moves
in the same epoch (34.9% -> 22.1%).

But it is **not only** a threshold shift, and AUC proves it, because moving a
threshold cannot change AUC. `near_3plus` AUC sits at chance through three
epochs (0.5354, 0.5523, 0.5201) and then jumps to 0.7120 at epoch 4. The model
acquired discriminative signal on the hardest negatives and recalibrated in the
same epoch. Both AUC and balanced accuracy rise monotonically across all five
epochs, so nothing here is stuck or saturating.

## Prefer AUC for the N=0 vs N=36 comparison

The N=0 arm ends at +27.0% True-bias. If the N=36 arm is better calibrated -
which the 0A Llama results suggest, where bias fell alongside accuracy - then
comparing raw stratum accuracies partly measures calibration rather than
capability. AUC is invariant to that, so report both: AUC for whether filler
tokens buy discriminative signal, accuracy for what the deployed decision
actually does.

## What this cannot support

- **n=1.** One seed against two or three per Llama arm. The Llama N=36 seeds
  spanned 0.788 to 1.000 on filler accuracy, so a single seed cannot separate
  "architecture is behind" from "this seed landed low".
- **The optimization settings are not matched to the Llama arms.** Both use
  `lr 1e-4` with the same schedule, but RWKV runs at batch 32 against Llama's
  384. With no gradient accumulation that is 12x as many updates at the same
  step size, so the effective per-sample learning rate differs by roughly an
  order of magnitude. "RWKV learns this task more slowly than Llama" is
  therefore **not** supported by this comparison; the batch and LR would have to
  be matched, or scaled by the usual linear/sqrt heuristics, before that claim
  could be made.
- **The Llama arms have no AUC figures here.** The trajectory and AUC analysis
  cover only the RWKV checkpoints; the Llama per-epoch checkpoints were not
  retained, so their rows in the results table stay accuracy-only and the AUC
  comparison above is within-RWKV.
- **Undertrained, by design.** The 0A work established that N=0 at 5x this
  budget reaches 96.4% on the hard negatives, so a weak N=0 result at 2M x 5 is
  the expected control behaviour and not a capability ceiling.

## What the paired N=36 arm should show

The discriminating measurement is **AUC on the hard negatives**, not overall
accuracy. The N=0 arm acquired `near_3plus` signal only at epoch 4 and ended at
0.7104. An N=36 arm reaching comparable AUC in fewer epochs demonstrates the
sample-efficiency effect the 0A work established; one ending near 0.71 after
five epochs does not, however much its raw accuracy differs.

Accuracy should still be reported, but read second and with the bias beside it:
a uniform accuracy lift with AUC unchanged would be a better-calibrated
guesser, not a model that has started computing the task.

## Reproducing

All five epoch checkpoints were evaluated identically, varying only
`--checkpoint` and `--out`; the command below is the final epoch.

The evaluation needs the RWKV CUDA kernel, which is JIT-compiled and therefore
needs `ninja` and MSVC `cl.exe` on PATH. Invoking `.venv\Scripts\python.exe`
directly does not provide either: `ninja.exe` lives in `.venv\Scripts`, which is
only on PATH when the venv is activated, and `cl.exe` needs `vcvars64.bat`.
`scripts/run_cuda_tests.ps1` contains the supported setup (vswhere, newest
toolset, environment imported into the session); reuse it rather than
hand-rolling, or the kernel load fails with a bare
"Failed to compile/load the RWKV-7 CUDA kernel".

```bash
python scripts/evaluate_structural_challenge.py \
  --checkpoint results/exp0_0b_pilot/n0/checkpoints/c968fce9af66aa32/seed_42/epoch_005.pt \
  --challenge_set results/exp0_structural/structural_challenge_20260821.json \
  --precision bf16 --device cuda --batch_size 64 \
  --out results/exp0_0b_structural/eval_rwkv_n0_seed_42.json
```
