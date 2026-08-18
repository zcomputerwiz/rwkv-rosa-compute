# Architecture Overview

## Package Structure

- `rosa_compute.config`: `ROSAConfig` data class defining model hyperparameters and post-init validation.
- `rosa_compute.blinkdl_reference`: CPU oracle implementing BlinkDL ROSA-4bit logic (`rosa_slow_ref`) producing pure signed output in `{-1.0, 0.0, +1.0}` when `emb=None`, or `signed_result * emb` when `emb` is supplied.
- `rosa_compute.rosa_compat`: Adapter converting between BlinkDL tensor layouts `[B, T, 768]` and `rosa_soft` layouts `[B, T, 192, 4]`, overriding `max_suffix_length=512`. Includes `apply_blinkdl_embedding` for `signed_rosa * emb`.
- `rosa_compute.checkpoint`: Checkpoint loader, inspector (`inspect_checkpoint`), shape validator, missing/unexpected key handler, and SHA-256 hash calculator for 0.1B ROSA checkpoints.
- `rosa_compute.diagnostics`: System, CUDA, git submodule commits, and `rosa_soft.BUILD_CAPABILITIES` inspection utilities.
- `rosa_compute.model`: `ROSAModelSkeleton` compatibility scaffold matching BlinkDL block structure.
