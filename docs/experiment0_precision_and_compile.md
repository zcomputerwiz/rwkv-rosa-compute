# Experiment 0 precision and compilation levers

Measured throughput levers for Experiment 0 training, each reported with the
numerical deviation it causes. A speedup on its own is not actionable here:
every lever below is a numerical or execution intervention, is recorded in
`TrainConfig` and the run report, and changes the deterministic `run_id`.

Reproduce with:

```bash
python scripts/benchmark_training_precision.py --arch both
```

## Results

RTX 4060 Ti (Ada, cc8.9), CUDA 12.9, torch 2.13, at the shapes of the actual
runs. `d(loss)` is measured on a single forward/backward from a fixed model and
batch; `floor` is the spread across identical repeats.

```text
0A Llama (batch 384, 4 layers, hidden 384)
  variant            ms/step  speedup           loss      d(loss)    floor
  fp32                281.42    1.00x    10.30972385            -   0.0e+00
  fp32 + TF32         228.28    1.23x    10.30995655   +2.327e-04   0.0e+00
  bf16                174.33    1.61x    10.30812645   -1.597e-03   0.0e+00
  bf16 + compile      132.59    2.12x    10.30806160   -1.662e-03   0.0e+00

0B RWKV-7 (batch 128, 12 layers, hidden 768)
  bf16                479.23    1.00x     9.79593754            -   0.0e+00
  bf16 + compile      272.78    1.76x     9.79593754   +0.000e+00   0.0e+00
```

**These 0B compile figures were re-measured.** The original run recorded
342.21 ms and 1.40x, taken while `torch.compile` was silently falling back to
eager for RWKV layers 8-11: dynamo specialized the block frame on the integer
`self.layer_id`, and 12 layers exceeded its recompile limit of 8. With that
fixed the same benchmark gives 272.78 ms and **1.76x**. The Llama figures are
unaffected - it has 4 layers, under the limit. See
`docs/experiment0_grouped_execution.md` for the diagnosis and the fix.

Projected against the completed runs: the 8.9h 0A N=0 run becomes about 4.2h at
`bf16 + compile`, and the projected 49.6h 0B run becomes about 28.2h.

### Why numerics are measured on one pass

Measuring deviation across many training steps on a fixed batch does not work.
Differences amplify chaotically, and the run-to-run floor grows with step count:
identical repeats of the same configuration differ by ~2e-03 at 30 steps and
~2e-02 at 40, which swamps every effect being measured. A single forward and
backward is bitwise reproducible on both architectures (`floor 0.0e+00`), so the
deviations above are signal.

This also means the architectures are **not** bitwise reproducible across a
multi-step run. RWKV drifts roughly 150x more than Llama per unit of training
(2.1e-03 versus 1.4e-05 across three identical 30-step repeats). That is
accumulation, not a defect in the fused kernel, whose single call reproduces
exactly.

## Interpretation

**TF32 is the cheapest lever, and is opt-in.** 1.23x for a deviation roughly 7x
smaller than BF16's, at identical memory. It applies only to FP32 matmuls, so it
is inert under BF16 autocast — combining them is pointless, not additive.

`--precision fp32` deliberately still means *strict* FP32. TF32 lowers the
internal precision of FP32 matmuls, so silently redefining the existing protocol
would retroactively change what every completed FP32 run claims to be. It is
exposed as `--tf32` / `--no-tf32`, defaults off, and changes the `run_id`.
`--compile` / `--no-compile` is recorded the same way.

**BF16 is a protocol change, not a free speedup.** Its 1.61x comes with a
deviation ~7x larger than TF32 and 1.8 GiB less memory. The repository already
treats precision this way.

**`torch.compile` is nearly free numerically.** On RWKV it is bitwise identical
to eager while delivering 1.76x. On Llama it adds only -6.5e-05 on top of BF16's
-1.597e-03, about 4% of a deviation already accepted.

**Fused AdamW is not worth a protocol note.** It measured ~2-4% and sits inside
the run-to-run variability of a real run.

**Batch size is not a lever for 0B.** Memory scales at 70.7 MiB/sample, so batch
128 already occupies 10.0 GiB of 16 GiB, and a 4x batch increase bought only
1.30x throughput.

## Windows toolchain constraint

`torch.compile` needs Triton for GPU codegen. Upstream publishes no Windows
wheel under the `triton` name; Windows uses the `triton-windows` distribution,
now maintained under the `triton-lang` organization. It is installed by the
`cuda` extra:

```bash
pip install -e ".[cuda]"
```

There is a second, less obvious constraint. **MSVC 14.38 cannot compile the C
that Triton generates**, failing with:

```text
cuda_utils.c(939): error C2059: syntax error: '}'
```

`vcvars64.bat` activates the installation's *default* toolset, which is not
necessarily the newest installed one, so a machine with both 14.38 and 14.44
installed will fail by default. `scripts/run_cuda_tests.ps1` now selects the
newest installed toolset and prints it; override with `EXP0_MSVC_TOOLSET`.

This constraint is easy to misdiagnose because the failure modes are silent or
misleading:

- Without Triton, `torch.compile` runs but reports no speedup at all.
- `torch.compile(model)` wraps `forward` only. `OptimizedModule.__getattr__`
  forwards every other attribute to the original module, so compiling a module
  and then calling `model.loss_logits(...)` silently runs eager and also reports
  no speedup. Compile the bound method that is actually invoked.

A compile that is genuinely happening takes 20-50s of warmup. A sub-2s warmup
means one of the two conditions above.

## Adoption sequence

Speedups are not adopted on the strength of a per-step deviation. The protocol
to validate is the one intended for production use, and it is validated by
reproducing a completed result.

**0B RWKV: `BF16 + compile` first.** RWKV is already constrained to BF16 by the
fused kernel, so precision does not change relative to the intended CUDA
protocol, and compile measured bitwise-identical on a single pass. This is the
lowest-risk optimized candidate available.

**0A Llama: validate `BF16 + compile` directly**, rather than spending one run
validating BF16 and a second validating compile. BF16 is the numerically
dominant intervention by roughly 25x, so the stack is what matters.

The controlled replication is the completed N=0 arm, unchanged in every other
respect — same seed 42, length 6, dimension 3, 10M examples, 5 epochs, same
validation set — compared against its FP32 trajectory:

```text
0.8895  0.9665  0.9740  0.9830  0.9930
```

The comparison must not demand identical trajectories. Tiny numerical
differences diverge optimizer paths, and this document already shows that
repeated identical multi-step runs diverge on their own. The question is whether
the *scientific result* reproduces:

```text
same qualitative learning curve
final accuracy in the same regime (roughly 0.99-0.995)
no new degenerate predictor
CoT diagnostics unchanged
construction strata qualitatively normal
```

If it reproduces, the optimized protocol is validated for the whole 0A sweep. If
it does not, fall back to `FP32 + TF32` and validate that instead.

At 8.9h per arm an eight-point sweep costs about 71 GPU-hours; at 4.2h it costs
about 34.

## Scientific handling

Each lever changes the deterministic `run_id`, which is correct: they are
different experiments. Two rules follow.

Do not enable a lever partway through a sweep. An accuracy-versus-N curve whose
points were produced under different precision or compilation settings mixes
protocols, exactly as the early-stopping caveat in
[`experiment0_positive_control_repair.md`](experiment0_positive_control_repair.md)
describes for stopping rules.

Do not treat a speedup as free because its deviation is small. The deviations
above are measured at initialization on one batch. They bound the per-step
numerical difference; they do not bound how a full training trajectory diverges,
and the multi-step drift figures above show that divergence is real. That is
precisely why adoption runs through the anchor replication described above
rather than through the per-step deviation alone.
