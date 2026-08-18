# Architecture Overview

## Package Structure

- `rosa_compute.config`: `ROSAConfig` data class defining model hyperparameters and invariant validations (`n_layer`, `n_embd`, `vocab_size`, `context_length`, `rosa_bits`, `rosa_groups`, `dtype`).
- `rosa_compute.model`:
  - `RWKV_CMix_x070`: ChannelMix/FFN module using `time_shift`, learned `x_k`, `key` linear (`bias=False`), and `value` linear (`bias=False`), computing `value(relu(key(x + xx * x_k)) ** 2)`.
  - `ROSABlock`: RWKV-8 block containing `ln0` (block 0 only), `ln3`, ROSA layer, residual addition `x + rosa(ln3(x))`, `ln2`, FFN, and residual addition `x + ffn(ln2(x))`.
  - `ROSAModelSkeleton`: Full model assembly matching BlinkDL block state dict structure (`emb`, `blocks`, `ln_out`, `head`).
- `rosa_compute.blinkdl_reference`: CPU oracle implementing BlinkDL ROSA-4bit logic (`rosa_slow_ref`) producing pure signed output in `{-1.0, 0.0, +1.0}` when `emb=None`, or `signed_result * emb` when `emb` is supplied.
- `rosa_compute.rosa_compat`:
  - `rosa_4bit_forward`: Adapter converting between BlinkDL tensor layouts `[B, T, C]` and `rosa_soft` layouts `[B, T, H, D]`, enforcing `max_suffix_length=512`, input rank/shape/dtype/device checks, and device selection.
  - `apply_blinkdl_embedding`: Transforms pure signed ROSA symbols `{-1.0, 0.0, +1.0}` with `rosa_qkv.emb` (`+1 -> +emb`, `-1 -> -emb`, `0 -> 0`).
  - `ROSALayerCompat`: Projections + ROSA routing + `rosa_qkv.emb` + output projection layer wrapper.
- `rosa_compute.checkpoint`: Checkpoint loader (`load_rosa_checkpoint`), inspector (`inspect_checkpoint`), and shape validator derived dynamically from `ROSAModelSkeleton(config).state_dict()`. Uses `weights_only=True` for safe state dict loading.
- `rosa_compute.diagnostics`: System, CUDA, git submodule commits, and `rosa_soft.BUILD_CAPABILITIES` inspection utilities.
