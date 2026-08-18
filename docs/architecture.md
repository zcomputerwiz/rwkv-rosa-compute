# Architecture Overview

## Overview
`rosa-compute` is an experimental research package providing compatibility layers, correctness oracles, and benchmarking tools for ROSA (Routing On Suffix Automata / Attention) in RWKV-8 architectures.

## Modules

- `rosa_compute.config`: `ROSAConfig` data class defining model hyperparameters and validation.
- `rosa_compute.blinkdl_reference`: CPU oracle implementing BlinkDL ROSA-4bit logic and route search.
- `rosa_compute.rosa_compat`: Adapter converting between BlinkDL tensor layouts `[B, T, 768]` and `rosa_soft` layouts `[B, T, 192, 4]`, overriding `max_suffix_length=512`. Includes learned embedding reconstruction.
- `rosa_compute.checkpoint`: Checkpoint loader, shape validator, missing/unexpected key handler, and SHA-256 hash calculator for 0.1B ROSA checkpoints.
- `rosa_compute.diagnostics`: System, CUDA, and `rosa_soft.BUILD_CAPABILITIES` inspection utilities.
- `rosa_compute.model`: 0.1B ROSA-4bit model wrapper skeleton.
