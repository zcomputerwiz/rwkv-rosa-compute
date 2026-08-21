# ROSA Compute Research Repository

## Purpose
This repository is a compatibility scaffold, research testbed, and execution harness for:
- **RWKV-8 ROSA-4bit 0.1B** semantic compatibility and full model structure fidelity
- **CUDA-accelerated ROSA execution** using the `rosa_soft` implementation
- Safe checkpoint loading and state dictionary validation matching the BlinkDL model contract
- Incremental GPU ROSA runtime development and test-time compute research

## Model Architecture
The local `ROSAModelSkeleton` structurally matches BlinkDL's RWKV-8 ROSA-4bit target architecture:
- `emb`: Token embedding (`vocab_size x n_embd`)
- `blocks`: `ModuleList` of ROSA Blocks:
  - Block 0 contains `ln0` (LayerNorm)
  - Pre-ROSA LayerNorm `ln3`
  - ROSA Layer with `x_q`, `x_k`, `x_v`, linear `q`, `k`, `v` (with bias), `rosa_qkv.emb` parameter, and linear `o` (with bias)
  - Residual addition: `x = x + rosa(ln3(x))`
  - Pre-FFN LayerNorm `ln2`
  - ChannelMix / FFN: `x_k` parameter, linear `key` (`bias=False`), linear `value` (`bias=False`), computing `value(relu(key(x + xx * x_k)) ** 2)`
  - Residual addition: `x = x + ffn(ln2(x))`
- `ln_out`: Output LayerNorm
- `head`: Final linear classification head (`bias=False`)

## Core Value Representation
BlinkDL's ROSA-4bit layer represents pure routing results as signed values in `{-1.0, 0.0, +1.0}`:
- **+1.0**: Matched value bit was 1
- **-1.0**: Matched value bit was 0
- ** 0.0**: Unmatched route

BlinkDL's learned parameter `emb` (`rosa_qkv.emb`) is applied separately via element-wise multiplication:
```text
output = signed_rosa_result * emb
```

## Checkpoint Loading & Safety
Checkpoints are safely deserialized using PyTorch `weights_only=True` state dictionaries:
- `load_rosa_checkpoint()` loads state dicts and performs strict key and shape validation against `ROSAConfig`.
- `inspect_checkpoint()` inspects `.pth` files without instantiating model code.

## Development Setup

```bash
# Initialize submodules
git submodule update --init --recursive

# Install development dependencies and package in editable mode
pip install -r requirements-dev.txt
pip install -e .

# Run test suite
python -m pytest -v

# Run linter
ruff check .
```

## Running Benchmarks and Diagnostics

```bash
# Run latency benchmark (mean ± std)
python benchmarks/benchmark_rosa.py --smoke

# Run diagnostic comparison between BlinkDL reference and rosa_soft reference
python scripts/compare_rosa.py

# Inspect environment and optional checkpoint
ROSA_MODEL_PATH=path/to/checkpoint.pth python scripts/inspect_environment.py

# Run Experiment 0 positive-control smoke test (Llama)
python scripts/run_experiment.py --architecture llama --device cpu --num_samples 64 --val_samples 32 --epochs 1

# Measure precision and compilation throughput levers
python scripts/benchmark_training_precision.py --arch both

# Evaluate a completed training checkpoint
python scripts/evaluate_exp0_checkpoint.py --checkpoint path/to/checkpoint.pt --run_report path/to/report.json --out results/eval.json
```

## Documentation Index

Comprehensive research and engineering documentation is available under [`docs/`](docs/):

- **Research Roadmap & Hypotheses**:
  - [`docs/experiments.md`](docs/experiments.md) — Research sequence (H1–H4), experimental protocols, and success criteria.
  - [`docs/complementary_research_roadmap.md`](docs/complementary_research_roadmap.md) — Complementary research tracks and milestones.
- **Experiment 0 Protocols & Diagnostics**:
  - [`docs/experiment0_execution.md`](docs/experiment0_execution.md) — Operational execution protocol for 0A (Llama) and 0B (RWKV-7).
  - [`docs/experiment0_positive_control_repair.md`](docs/experiment0_positive_control_repair.md) — Positive control diagnosis, shared input projection, and metric separation.
  - [`docs/experiment0_construction_diagnostics.md`](docs/experiment0_construction_diagnostics.md) — Construction-prior strata tracking and challenge sets.
  - [`docs/experiment0_checkpoint_analysis.md`](docs/experiment0_checkpoint_analysis.md) — Checkpoint re-evaluation and cross-seed error comparison.
  - [`docs/experiment0_checkpointing.md`](docs/experiment0_checkpointing.md) — Atomic checkpointing, exact resume, and completed-run continuation.
- **Performance, Precision & CUDA**:
  - [`docs/experiment0_precision_and_compile.md`](docs/experiment0_precision_and_compile.md) — Benchmarking `torch.compile`, TF32, and BF16 on Ada GPUs.
  - [`docs/experiment0_performance.md`](docs/experiment0_performance.md) — Memory optimizations, grouped execution, and CUDA benchmarks.
  - [`docs/experiment0_cuda.md`](docs/experiment0_cuda.md) — CUDA testing on Windows, fused kernel build requirements, and verification gates.
  - [`docs/experiment0_input_pipeline.md`](docs/experiment0_input_pipeline.md) — Input pipeline throughput profiling and bottleneck analysis.
- **Architecture & Kernel Audits**:
  - [`docs/architecture.md`](docs/architecture.md) — Overview of `rosa_compute` and `exp0` package structures.
  - [`docs/rwkv7_upstream_kernel_audit.md`](docs/rwkv7_upstream_kernel_audit.md) — Source audit of upstream RWKV-7 training kernels.
  - [`docs/rwkv7_single_step.md`](docs/rwkv7_single_step.md) — Persistent single-step CUDA recurrence prototype.
  - [`docs/compatibility.md`](docs/compatibility.md) — BlinkDL target parameters and upstream submodule provenance.
  - [`docs/development.md`](docs/development.md) — Development rules and CPU oracle requirements.

## Attribution and Reference Material
Experiment 0 and 3SUM task implementation in `src/exp0` are based on:
- Pfau, J., Merrill, W., & Bowman, S. R. (2024). *Let's Think Dot by Dot: Hidden Computation in Transformer Language Models.* arXiv:2404.15758 (Licensed CC-BY-4.0).
