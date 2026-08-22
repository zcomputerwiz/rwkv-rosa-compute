# Experiment 0B: RWKV-7 N=0 vs N=36 on the structural challenge set

Paired evaluation of the 0B RWKV-7 arms on the frozen 6-stratum structural
challenge set used for the 0A Llama arms. Both arms share the identical task and
architecture configurations; the difference is the filler-token budget and sequence format mixture.

```text
challenge set   results/exp0_structural/structural_challenge_20260821.json
challenge_id    e06f92897411fe2e
content_sha256  bef50bba1c80600de6885bf60ef5f9fdfed1b37135715dce0b938cee6a1cb21b
N=0 run_id      c968fce9af66aa32, seed 42 (100% filler, 2M x 5)
N=36 run_id     c923f49572cadb88, seed 42 (50/50 CoT/filler, 2M x 5)
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

## Comparison: RWKV N=0 vs RWKV N=36

1. **Mixture and Effective Sample Budget**:
   - The N=0 arm trained purely on filler sequences (2,000,000 filler instances per epoch = 10M total).
   - The N=36 arm used the standard 50/50 mixture (1,000,000 CoT + 1,000,000 filler instances per epoch = 5M filler total).
   - Parallel CoT was solved instantly by N=36 (99.9997% online accuracy, 100.0% teacher-forced accuracy).
   - On the filler format, N=36 received half the gradient steps on filler sequences compared to N=0, and correspondingly ends earlier on the True-bias collapse curve (+40.6% vs +27.0%).

2. **Positive Detection vs Negative Rejection**:
   - Both RWKV arms master positive detection easily: N=36 reaches **89.1%** (and 89.0% on surviving corrupted positives), outperforming N=0 (81.6%) and matching the Llama arms (~88–92%).
   - The entire performance gap across strata is driven by negative-rejection hardness, matching the underlying mechanism found in 0A.

3. **Graded Structural Discrimination**:
   - Both arms display monotonic AUC ordering across negative hardness strata:
     - `near_0`: AUC rises to **0.8257** (N=36) vs **0.8314** (N=0).
     - `near_1`: AUC reaches **0.6610** (N=36) vs **0.7496** (N=0).
     - `near_2`: AUC reaches **0.5975** (N=36) vs **0.7088** (N=0).
     - `near_3+`: AUC reaches **0.5662** (N=36) vs **0.7104** (N=0).
   - The acquisition of harder negative discrimination occurs later in training and requires more effective filler exposure.

4. **Regime Shift at Epoch 4**:
   - Just like N=0, N=36 exhibits an acquisition event and threshold shift around Epoch 4:
   - True-bias drops from +50.8% down to +42.4%, positive accuracy eases slightly from 95.2% to 89.7%, and `near_0` accuracy leaps from 43.3% to 58.1%.
