# Experiment 0B: RWKV-7 N=0 vs N=36 on the structural challenge set

Paired evaluation of the 0B RWKV-7 arms on the frozen 6-stratum structural
challenge set used for the 0A Llama arms. Both arms share the identical task and
architecture configurations, including the same 50/50 format mixture; the only
difference is the filler-token budget.

```text
challenge set   results/exp0_structural/structural_challenge_20260821.json
challenge_id    e06f92897411fe2e
content_sha256  bef50bba1c80600de6885bf60ef5f9fdfed1b37135715dce0b938cee6a1cb21b
N=0 run_id      c968fce9af66aa32, seed 42 (50/50 CoT/filler, 0 filler tokens)
N=36 run_id     c923f49572cadb88, seed 42 (50/50 CoT/filler, 36 filler tokens)
protocol        RWKV-7 768h/12L head_dim 64, bf16, compile + grouped + fused
                AdamW, batch 32, 2M x 5 epochs, no early stop triggered
```

The `content_sha256` matches the value recorded by all six Llama evaluations and both RWKV arms.

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
RWKV  N=36 s42      61.20%  52.10%  +40.6%   89.10%  89.00%  61.80%  29.50%  22.90%  20.30%
```

`bias` is the challenge-set True-prediction bias: predicted-true minus
actual-true, as a percentage of the 6000 instances.

## RWKV N=0 training trajectory, and the acquisition event at epoch 4

All five epoch checkpoints of each arm were evaluated on the frozen set.
Accuracy confounds discrimination with calibration; AUC does not, being
threshold-free, so both are reported. AUC is computed against the positive
instances.

```text
epoch  canon   chall    pos  corr_pos  near_0  near_1  near_2  near_3+
  1    56.85%  44.33%  91.30   89.40   33.20   20.20   17.80    14.10
  2    58.35%  46.62%  93.70   93.70   42.90   20.90   14.70    13.80
  3    60.35%  48.87%  94.00   94.50   54.60   21.60   15.70    12.80
  4    63.05%  55.02%  84.00   83.70   49.90   38.80   34.70    39.00
  5    66.10%  60.60%  81.60   81.30   64.90   48.20   43.10    44.50
```

```text
          bias   bal.acc  AUC all    near_0  near_1  near_2  near_3+
epoch 1  +49.2%  55.84%   0.5924    0.6677  0.5943  0.5723   0.5354
epoch 2  +49.2%  58.39%   0.6255    0.7549  0.6114  0.5837   0.5523
epoch 3  +47.3%  60.21%   0.6497    0.8216  0.6507  0.6065   0.5201
epoch 4  +34.2%  62.22%   0.7265    0.7776  0.7220  0.6944   0.7120
epoch 5  +27.0%  65.81%   0.7500    0.8314  0.7496  0.7088   0.7104
```

Through epochs 1-3 only the easiest negatives improve: `near_0` accuracy climbs
33.2 -> 54.6 while `near_3plus` *declines* (14.1 -> 13.8 -> 12.8). Epoch 3 to 4
is a regime change - positive accuracy drops 94.0 -> 84.0 while every hard
negative jumps, which is the signature of a threshold shift, and the
training-side True-bias moves in the same epoch (34.9% -> 22.1%).

But it is not only a threshold shift, and AUC proves it, because moving a
threshold cannot change AUC. `near_3plus` AUC sits at chance for three epochs
(0.5354, 0.5523, 0.5201) then jumps to 0.7120. The model acquired discriminative
signal on the hardest negatives and recalibrated in the same epoch. This
distinction is what the N=0 vs N=36 comparison below turns on.

The flat *accuracy* profile at epoch 5 (64.9 / 48.2 / 43.1 / 44.5, far less
spread than any Llama arm) is therefore not absent discrimination. It is the
+27.0% True-bias compressing genuinely graded signal - per-stratum AUC runs
0.8314 / 0.7496 / 0.7088 / 0.7104, ordered by difficulty.

## RWKV N=36 Training Trajectory

All five epoch checkpoints for the N=36 arm were evaluated against the challenge set.

### Accuracy by stratum:
```text
epoch  canon   chall    bias      pos  corr_pos  near_0  near_1  near_2  near_3+
  1    57.20%  45.85%  +46.7%   90.20   87.30   37.00   23.00   21.20    16.40
  2    56.85%  46.27%  +47.7%   91.40   90.40   42.50   21.70   15.90    15.70
  3    58.20%  45.48%  +50.8%   95.20   93.80   43.30   15.90   11.80    12.90
  4    59.90%  50.68%  +42.4%   89.70   89.70   58.10   26.10   20.70    19.80
  5    61.20%  52.10%  +40.6%   89.10   89.00   61.80   29.50   22.90    20.30
```

### Threshold-free AUC and Balanced Accuracy:
```text
          bias   bal.acc  AUC all    near_0  near_1  near_2  near_3+
epoch 1  +46.7%  56.57%   0.5992    0.6709  0.6030  0.5842   0.5389
epoch 2  +47.7%  57.43%   0.6192    0.7415  0.5965  0.5740   0.5650
epoch 3  +50.8%  57.74%   0.6375    0.7878  0.6209  0.5755   0.5660
epoch 4  +42.4%  60.44%   0.6543    0.8134  0.6481  0.5894   0.5661
epoch 5  +40.6%  61.34%   0.6626    0.8257  0.6610  0.5975   0.5662
```

## Comparison: the arms are identical until N=0 acquires, and N=36 never does

Both arms use the **same 50/50 format mixture**, with realized counts identical
to the instance:

```text
                          parallel_cot     filler     num_filler
N=0  c968fce9af66aa32         999,626    1,000,374         0
N=36 c923f49572cadb88         999,626    1,000,374        36
```

`num_filler=0` does not mean "no chain-of-thought arm". It means the
filler-format sequences carry zero filler tokens. Both arms see the same
1,000,374 filler-format instances per epoch, so the arms are **not** separated
by filler exposure or effective sample budget, and any explanation resting on
one arm getting fewer filler gradient steps is mistaken.

### Through epoch 3 the arms are indistinguishable

```text
AUC all      ep1     ep2     ep3     ep4     ep5
N=0        0.5924  0.6255  0.6497  0.7265  0.7500
N=36       0.5992  0.6192  0.6375  0.6543  0.6626
```

N=36 is marginally *ahead* at epoch 1 and within 0.012 of N=0 through epoch 3.
The arms separate only at epoch 4, and only because of what happens in one of
them.

### The separation is one discrete event, not a dose-response

```text
near_3plus AUC   ep1     ep2     ep3     ep4     ep5
N=0            0.5354  0.5523  0.5201  0.7120  0.7104
N=36           0.5389  0.5650  0.5660  0.5661  0.5662
```

N=0 acquires hard-negative discrimination at epoch 4. N=36's is **flat to three
decimals across the final four epochs**, pinned just above chance. Both arms
learn easy-negative rejection equally well (`near_0` AUC reaches 0.8314 and
0.8257), so the divergence is confined to the strata that require actually
computing the task.

Final per-stratum AUC:

```text
              near_0  near_1  near_2  near_3+   AUC all
N=0  ep5      0.8314  0.7496  0.7088   0.7104    0.7500
N=36 ep5      0.8257  0.6610  0.5975   0.5662    0.6626
```

### Seed 44 replicates the N=36 arm (added 2026-08-24)

The seed study's first completed N=36 replicate, run on `antigravity-ampere`
(RTX 3070), `run_id` `cd865b1f9c9b1089`, cross-checked against an independent
computation of the expected hash before the run was trusted:

```text
              near_0  near_1  near_2  near_3+   AUC all
N=36 ep5      0.8257  0.6610  0.5975   0.5662    0.6626   seed 42
N=36 ep5      0.7812  0.6208  0.5849   0.5593    0.6365   seed 44
```

Same shape, uniformly slightly lower. The load-bearing cell is `near_3+`:
0.5593 against seed 42's 0.5662, both at chance, against N=0's 0.7104. Filler
accuracy climbed smoothly with no jump - 0.552, 0.560, 0.5685, 0.581, 0.5875 -
which is the N=36 signature rather than an acquisition trajectory.

**Two of two N=36 seeds show no acquisition event.** That is the first evidence
that the arm separation is not a one-seed lottery win in the N=36 direction. It
is still two draws, and the sampling objection below stands until the full k=5
per arm is in.

Provenance for this replicate closed completely: `challenge_id
e06f92897411fe2e` and `content_sha256 bef50bba1c80600d` both recompute from the
frozen challenge file, and the remote copy of that file is byte-identical to the
repo copy.

The AUC figures above were computed here from the eval's `per_instance` block
(score = `margin`, tie-corrected ranks), because that eval reported accuracy
only. Accuracy is not sufficient on these models: seed 44 predicts True for
78.2% of instances against a 33.3% base rate, so accuracy conflates
discrimination with calibration in exactly the way that produced a wrong
published reading of this data once already.

### Two readings this data does not support

**"N=36 also had an acquisition event at epoch 4."** It had a threshold shift.
Its True-bias moves +50.8% -> +42.4%, positive accuracy eases 95.2% -> 89.7%,
and `near_0` accuracy leaps 43.3% -> 58.1%. Every one of those is what moving a
decision threshold produces, and a threshold move **cannot change AUC**. Its
`near_3plus` AUC goes 0.5660 -> 0.5661. N=0's went 0.5201 -> 0.7120 across the
same boundary. Only one of these is an acquisition.

**"N=36 detects positives better."** It scores 89.1% against N=0's 81.6% on
`positive_arm_positive`, but it also sits at +40.6% True-bias against +27.0%. A
model that answers True more often scores higher on positives by construction.
The AUC comparison, which is invariant to that, puts N=36 behind on every
stratum.

### What the comparison does support, and what it cannot

The measured result is that **filler tokens did not help RWKV-7 at this budget**:
N=36 finishes 4.95 points behind on filler accuracy (0.6085 vs 0.6580) and
behind on AUC at every negative stratum.

It cannot support "filler tokens do not help RWKV". The mechanism separating the
arms is a single discrete acquisition event, and with **one seed per arm** this
experiment observes one Bernoulli draw from each condition. The 0A Llama arms at
this same budget make the concern concrete: the three N=36 seeds scored 0.788,
0.835 and 1.000, one transitioning to a perfect score and two not. Sampling one
ticket per arm and finding a transition in the control is entirely consistent
with a filler-token benefit that this experiment lacked the power to see.

Settling it needs seeds, not more epochs per seed. Both arms sit near the
transition, which is exactly the regime where seed variance dominates.


## What this cannot support

- **n=1 per arm.** One seed each, against two or three per Llama arm. Since the
  arms are separated by a discrete acquisition event rather than a smooth
  dose-response, this is one Bernoulli draw per condition.
- **The optimization settings are not matched to the Llama arms.** Both use
  `lr 1e-4` with the same schedule, but RWKV runs at batch 32 against Llama's
  384. With no gradient accumulation that is 12x as many updates at the same
  step size. Cross-architecture learning-speed claims are not supported.
- **The Llama arms have no AUC figures here.** Their per-epoch checkpoints were
  not retained, so the Llama rows stay accuracy-only and every AUC comparison is
  within-RWKV.
- **Undertrained, by design.** The 0A work established that N=0 at 5x this
  budget reaches 96.4% on the hard negatives, so weak absolute numbers at
  2M x 5 are expected control behaviour, not a capability ceiling.

## Reproducing

Each epoch checkpoint was evaluated identically, varying only `--checkpoint`
and `--out`.

The evaluation needs the RWKV CUDA kernel, which is JIT-compiled and therefore
needs `ninja` and MSVC `cl.exe` on PATH. Invoking `.venv\Scripts\python.exe`
directly provides neither: `ninja.exe` lives in `.venv\Scripts`, on PATH only
when the venv is activated, and `cl.exe` needs `vcvars64.bat`.
`scripts/run_cuda_tests.ps1` contains the supported setup; reuse it, or the
kernel load fails with a bare
"Failed to compile/load the RWKV-7 CUDA kernel".

```bash
python scripts/evaluate_structural_challenge.py \
  --checkpoint results/exp0_0b_pilot/n36/checkpoints/c923f49572cadb88/seed_42/epoch_005.pt \
  --challenge_set results/exp0_structural/structural_challenge_20260821.json \
  --precision bf16 --device cuda --batch_size 64 \
  --out results/exp0_structural/eval_rwkv_n36_seed_42.json
```
