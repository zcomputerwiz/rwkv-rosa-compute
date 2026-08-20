# Experiment 0 CUDA Validation

This document is the CUDA-specific companion to
[`experiment0_execution.md`](experiment0_execution.md) and
[`experiment0_performance.md`](experiment0_performance.md).

The CPU CI suite protects the Experiment 0 data, reporting, and reference-model
contracts. The tests below validate CUDA execution details that GitHub's CPU
runners cannot exercise.

## Why `loss.item()` changed

PR #9 introduced per-batch `loss.item()` while adding epoch-wise training-loss
reporting. Its purpose was to collect detached scalar losses for the epoch mean;
it was not required by backpropagation, gradient clipping, optimizer stepping,
or convergence logic.

On CUDA, however, `.item()` also forces the host to wait for the device. Calling
it once per batch therefore serialized a reporting operation into the training
hot path.

The current loop preserves the reporting contract by accumulating
`loss.detach()` into a device-side FP64 scalar and converting the epoch mean to
a Python scalar only once per epoch. A CPU regression compares this mean against
the historical per-batch `.item()` calculation.

Removing the per-batch synchronization would make naive CPU wall-clock timing
optimistic, because CUDA work is asynchronous. The loop therefore performs one
explicit device synchronization at the end of the training epoch before
recording `epoch_seconds`. This keeps `samples_per_second` a wall-clock measure
without forcing a synchronization after every batch.

Validation follows the same principle:

- the `ANS` structural contract and answer positions are checked on the CPU
  batch before transfer;
- prediction correctness accumulates on-device;
- one scalar is read back at the end of each validation pass.

Per-seed history records:

```text
loss_reporting_syncs_per_epoch
validation_result_syncs_per_pass
```

so this execution behavior remains inspectable.

## Running the CUDA suite on Windows

On Windows a bare `pytest -m cuda` fails before it tests anything. The fused
RWKV-7 kernel is compiled on first use, and that needs ninja (which lives in
`.venv\Scripts`, invisible unless the venv is activated) plus the MSVC
toolchain. Without them every fused test fails with `Ninja is required to load
C++ extensions`, which reads as a code failure and is not one.

`scripts/run_cuda_tests.ps1` assembles that environment and runs the same
selection as the `CUDA Tests` workflow, with the same `EXP0_REQUIRE_RWKV_CUDA=1`:

```powershell
.\scripts\run_cuda_tests.ps1
.\scripts\run_cuda_tests.ps1 -Cold          # clear the kernel cache first
.\scripts\run_cuda_tests.ps1 -k rwkv7_fused # extra args go to pytest
```

It prints the selected Visual Studio installation plus the resolved `cl.exe`,
`nvcc`, `ninja`, and interpreter before running, and exits with pytest's exit
code. The Windows extension path has been validated with Visual Studio 2022
(17.x). `vswhere` still selects the latest installed C++ toolchain; if that is a
different Visual Studio family, the script emits a warning and continues rather
than blocking a potentially compatible newer toolchain. Treat a compiler/build
failure under that warning as a possible host-toolchain compatibility issue.

This reproduces the workflow's *test* step, not its install step. Two
differences are expected and are not failures:

- `rosa_soft` reports `variant='reference'` locally while CI builds it with
  CUDA. Do not `pip install` the submodule on Windows to close that gap: MSVC
  links its `_C` extension without exporting `PyInit__C`, and because
  `rosa_compute.rosa_compat` puts that directory on `sys.path`, the artifact
  then breaks every `import rosa_compute`. Run
  `scripts/clean_rwkv_cuda_cache.ps1` to remove one if it appears.
- `test_llama_bf16_flash_sdpa_smoke` skips, because PyTorch's Windows wheels
  ship without the flash-attention backend. Linux wheels provide it.

Use `-Cold` when the point is to prove the build rather than the tests. A warm
run reuses the cached kernel and finishes in a few seconds; a cold one takes
roughly twenty and is what a fresh CI runner actually does.

## Quick CUDA gate

Run the non-slow CUDA tests before a long 0A run:

```bash
pytest -m "exp0 and cuda and not slow" -v
```

These tests cover:

- FP32 Llama training;
- BF16 autocast training when supported;
- FP16 training through `torch.amp.GradScaler`;
- BF16 with fused AdamW;
- answer-only evaluation projection versus the compatibility full-logit path;
- forced FlashAttention SDPA execution when the installed PyTorch/GPU supports
  that backend;
- finite losses/gradients;
- CUDA peak allocated/reserved memory instrumentation;
- non-blocking transfer instrumentation;
- synchronized epoch timing.

A skipped BF16/Flash test means that the local hardware/software stack did not
advertise that capability. A failure is a reason not to use the corresponding
performance option for a production experiment until investigated.

## RWKV recurrence backends

Experiment 0 keeps two explicit RWKV recurrence implementations.

### Reference

```text
--rwkv_kernel reference
```

This remains the default. It is the existing PyTorch FP32 recurrent-state oracle
and is the implementation used by the CPU equivalence suite.

### Pinned upstream CUDA recurrence

```text
--rwkv_kernel cuda
--precision bf16
```

This path lazily compiles the `rwkv7_clampw` x070 recurrence sources from the
repository's pinned `external/RWKV-LM` submodule. It deliberately does not use
an automatic machine-dependent fallback: the requested recurrence backend is
part of `ModelConfig`, the run report, and the deterministic run/sweep identity.
If the CUDA backend is requested and unavailable, the run fails explicitly.

Current requirements are:

```text
architecture = rwkv
precision    = bf16
head_dim     = 64
CUDA device with BF16 support
CUDA toolkit / nvcc for first extension build
initialized external/RWKV-LM submodule
```

Install the optional build helper with:

```bash
pip install -e ".[cuda]"
```

The upstream kernel consumes BF16 recurrence inputs while maintaining its
recurrent/saved backward state in FP32. Its chunk length is 16. Experiment 0
pads only the causal tail to the next multiple of 16 and slices the padded
outputs away; this allows arbitrary Experiment 0 sequence lengths without
adding prefix computation.

## Fused RWKV oracle gate

The first fused-kernel test may compile the CUDA extension, so it is also marked
`slow`:

```bash
pytest -m "exp0 and cuda and slow" -v
```

The tests compare both **forward values and backward gradients** against the
reference recurrence and deliberately use `T=17` so the tail-padding path is
exercised.

If the machine has a CUDA-capable PyTorch build but lacks a local CUDA toolkit,
the slow fused tests skip by default. To make absence of the toolkit a hard
failure:

### PowerShell

```powershell
$env:EXP0_REQUIRE_RWKV_CUDA = "1"
pytest -m "exp0 and cuda and slow" -v
```

### bash

```bash
EXP0_REQUIRE_RWKV_CUDA=1 pytest -m "exp0 and cuda and slow" -v
```

The fused backend should not be used for a production 0B run until these oracle
tests pass on the intended GPU/toolchain.

## Recommended 0A performance comparison

The RWKV kernel is irrelevant to 0A. Compare the following on the target GPU
while keeping all other run parameters fixed:

```text
1. fp32
2. bf16
3. bf16 + fused AdamW
```

Use the report fields rather than only external GPU monitoring:

```text
epoch_seconds
samples_per_second
data_wait_seconds
cuda_peak_memory_allocated_bytes
cuda_peak_memory_reserved_bytes
```

Do not enable TF32 or `torch.compile` in the same comparison; both are separate
numerical/execution interventions and should be measured independently after the
basic CUDA path is validated.

## Recommended 0B comparison

After the stock pretrained checkpoint loads successfully and the fused oracle
gate passes, compare:

```text
reference recurrence + bf16 surrounding model
fused CUDA recurrence + bf16
```

under identical model/checkpoint/data/batch settings. The recurrence backend is
part of run identity, so reports from these two paths cannot silently collide.
