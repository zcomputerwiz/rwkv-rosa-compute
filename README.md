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
```
