# ROSA Compute Research Repository

## Purpose
This repository is an experimental research project around:
- **RWKV-8 ROSA-4bit 0.1B** compatibility
- **CUDA-accelerated ROSA execution** using the `rosa_soft` implementation
- Later experimentation with test-time compute (`<think>`, `<scratchpad>`, `<checkpoint>`)
- Incremental GPU ROSA runtime development

## Current Status
**Phase 0 — Repository Skeleton**

The repository provides the compatibility adapters, CPU reference oracle, diagnostic scripts, benchmark scaffold, and test suite.

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
git submodule update --init --recursive
pip install -r requirements-dev.txt
pip install -e .
python -m pytest -q
```
