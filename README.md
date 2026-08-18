# ROSA Compute Research Repository

## Purpose
This repository is an experimental research project focused on:
- **RWKV-8 ROSA-4bit 0.1B** semantic compatibility and harness verification
- **CUDA-accelerated ROSA execution** using the `rosa_soft` implementation
- Later experimentation with test-time compute (`<think>`, `<scratchpad>`, `<checkpoint>`)
- Incremental GPU ROSA runtime development

## Current Status
**Phase 0 — Repository Skeleton & Semantic Validation Harness**

The repository provides compatibility adapters, CPU reference oracle, diagnostic scripts, benchmark scaffold, and test suite.

### Core Value Representation
BlinkDL's ROSA-4bit layer represents pure routing results as signed values in `{-1.0, 0.0, +1.0}`:
- **+1.0**: Matched value bit was 1
- **-1.0**: Matched value bit was 0
- ** 0.0**: Unmatched route

BlinkDL's learned parameter `emb` is applied separately via element-wise multiplication:
```text
output = signed_rosa_result * emb
```

## Repository Layout
```text
.
├── .github/
│   └── workflows/          # GitHub Actions CI workflows (CPU & CUDA)
├── external/               # Git submodules (RWKV-LM, rosa_soft)
├── src/
│   └── rosa_compute/       # Package source
├── tests/                  # Pytest test suite
├── benchmarks/             # Latency/throughput benchmarks
├── scripts/                # Environment inspection & comparison diagnostics
├── docs/                   # Documentation (architecture, compatibility, dev, experiments)
├── pyproject.toml          # Package configuration
└── requirements-dev.txt    # Development dependencies
```

## Upstream Projects
- **BlinkDL / RWKV-LM**: Target model script `RWKV-v8/260222_rosa4bitLM_L12.py`
- **wjie98 / rosa_soft**: Optimized CUDA runtime and reference operator

## Reproducibility
Upstream repositories are tracked as pinned Git submodules under `external/`.

## Development Setup

```bash
# Initialize submodules
git submodule update --init --recursive

# Install development dependencies and package in editable mode
pip install -r requirements-dev.txt
pip install -e .

# Run CPU test suite
python -m pytest -q
```
