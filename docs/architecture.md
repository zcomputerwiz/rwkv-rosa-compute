# Architecture Overview

## Package Structure

### `rosa_compute` (ROSA-4bit Foundation & Compatibility)

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

### `exp0` (Experiment 0 Synthetic Testbed & Measurement Harness)

- `exp0.config`: Dataclasses for model (`ModelConfig`), task (`Task3SumConfig`), and training/optimization (`TrainConfig`) configurations, plus deterministic hash identifiers.
- `exp0.models`:
  - `exp0.models.base`: `BaseTaskModel` and `InputEmbedWrapper` (shared tuple/CoT feature mapping and classifier head).
  - `exp0.models.llama`: `LlamaTaskModel` (Llama-style causal transformer with RoPE).
  - `exp0.models.rwkv`: `RWKVTaskModel` using pure PyTorch reference recurrence.
  - `exp0.models.rwkv_cuda`: `RWKVTaskModel` with fused CUDA recurrence kernel (`rwkv7_clampw`).
- `exp0.task3sum` & `exp0.generation`: 3SUM problem generation, format construction (immediate, filler, parallel CoT, serial CoT, neutral tokens), and tensor-packed buffer generation (`generate_protocol_packed_instances`).
- `exp0.dataset`: `Task3SumDataset`, `Vocabulary3Sum`, `pad_collate_fn`, and vocabulary reduction mapping.
- `exp0.train` & `exp0.evaluate`: Core training loop, AMP/mixed precision, loss computation, and vectorized evaluation.
- `exp0.grouped_execution`: Length-aware grouped batch execution for mixed-format training (token-weighted loss and gradient accumulation across variable sequence lengths).
- `exp0.checkpointing`: Atomic training checkpointing and exact-state recovery (`latest.pt`, `epoch_XXX.pt`).
- `exp0.checkpoint_analysis`: Post-hoc evaluation of saved checkpoints, deterministic validation and challenge dataset generation, structural instance feature extraction, and per-instance prediction margins.
- `exp0.error_comparison`: Multi-seed cross-evaluation, exact hypergeometric overlap tests, and stratum-wise error comparison.
- `exp0.construction_strata`: Construction-arm provenance tracking and stratum partitioning (positive arm vs corrupted arm vs surviving positives).
- `exp0.rwkv_checkpoint`: Stock pretrained x070 backbone loader, parameter name/shape adaptation, and strict validation.
- `exp0.diagnostics`: Metric calculation, CoT diagnostic reporting, and report serialization.

