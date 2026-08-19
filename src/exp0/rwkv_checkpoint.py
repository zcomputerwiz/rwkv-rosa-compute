"""Strict adapter for stock RWKV-7 x070 checkpoints used by Experiment 0B."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import torch

from exp0.config import ModelConfig
from exp0.models.rwkv import RWKV7Backbone

_BLOCK_KEY_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_ALLOWED_SOURCE_ONLY = {"emb.weight", "head.weight"}
_OPTIONAL_UNUSED_TARGET_KEYS = {
    "layers.0.time_mix.v0",
    "layers.0.time_mix.v1",
    "layers.0.time_mix.v2",
}


@lru_cache(maxsize=16)
def _sha256_for_stat(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a cached SHA-256 keyed by resolved path, size, and mtime."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _sha256_for_stat(str(resolved), stat.st_size, stat.st_mtime_ns)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError(
            "RWKV-7 checkpoint must contain a tensor state-dict mapping; "
            f"got {type(payload).__name__}."
        )

    state: Mapping[str, Any] = payload
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]

    if not state:
        raise ValueError("RWKV-7 checkpoint state dict is empty.")

    bad_keys = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if bad_keys:
        preview = ", ".join(map(str, bad_keys[:5]))
        raise ValueError(
            "RWKV-7 checkpoint state dict must contain only string -> Tensor "
            f"entries; invalid keys: {preview}"
        )

    return dict(state)


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a stock checkpoint safely on CPU using weights-only deserialization."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"RWKV-7 checkpoint not found: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    return _extract_state_dict(payload)


def infer_checkpoint_architecture(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, int | str]:
    """Infer the x070 backbone dimensions required by a stock checkpoint."""
    required = {
        "ln_out.weight",
        "blocks.0.ln1.weight",
        "blocks.0.ffn.key.weight",
        "blocks.0.att.r_k",
    }
    missing = sorted(required - set(state_dict))
    if missing:
        raise ValueError(
            "Checkpoint is not a supported stock RWKV-7 x070 state dict; "
            f"missing required keys: {missing}"
        )

    layer_ids = sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := _BLOCK_KEY_RE.match(key)) is not None
        }
    )
    if not layer_ids or layer_ids != list(range(layer_ids[-1] + 1)):
        raise ValueError(
            "RWKV-7 checkpoint block indices must be contiguous from zero; "
            f"found {layer_ids}."
        )

    hidden_size = state_dict["ln_out.weight"].numel()
    ln1_shape = tuple(state_dict["blocks.0.ln1.weight"].shape)
    if ln1_shape != (hidden_size,):
        raise ValueError(
            "RWKV-7 checkpoint hidden-size inference is inconsistent: "
            f"ln_out={hidden_size}, blocks.0.ln1.weight={ln1_shape}."
        )

    ffn_shape = tuple(state_dict["blocks.0.ffn.key.weight"].shape)
    if len(ffn_shape) != 2 or ffn_shape[1] != hidden_size:
        raise ValueError(
            "RWKV-7 checkpoint FFN shape is incompatible with inferred hidden "
            f"size {hidden_size}: {ffn_shape}."
        )
    intermediate_size = ffn_shape[0]

    r_k_shape = tuple(state_dict["blocks.0.att.r_k"].shape)
    if len(r_k_shape) != 2:
        raise ValueError(
            f"RWKV-7 checkpoint r_k must be rank 2, got shape {r_k_shape}."
        )
    num_heads, head_dim = r_k_shape
    if num_heads * head_dim != hidden_size:
        raise ValueError(
            "RWKV-7 checkpoint head geometry is inconsistent: "
            f"{num_heads} * {head_dim} != {hidden_size}."
        )

    return {
        "architecture": "rwkv7_x070",
        "num_hidden_layers": len(layer_ids),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "head_dim": head_dim,
        "num_attention_heads": num_heads,
    }


def _validate_target_config(
    model_cfg: ModelConfig,
    source_config: Mapping[str, int | str],
) -> None:
    expected = {
        "num_hidden_layers": model_cfg.num_hidden_layers,
        "hidden_size": model_cfg.hidden_size,
        "intermediate_size": model_cfg.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_attention_heads,
    }
    mismatches = {
        name: (expected[name], source_config[name])
        for name in expected
        if expected[name] != source_config[name]
    }
    if mismatches:
        details = ", ".join(
            f"{name}: requested={requested}, checkpoint={checkpoint}"
            for name, (requested, checkpoint) in mismatches.items()
        )
        suggested = (
            f"--hidden_size {source_config['hidden_size']} "
            f"--num_hidden_layers {source_config['num_hidden_layers']} "
            f"--intermediate_size {source_config['intermediate_size']} "
            f"--head_dim {source_config['head_dim']}"
        )
        raise ValueError(
            "RWKV-7 checkpoint architecture does not match ModelConfig: "
            f"{details}. Use checkpoint-compatible dimensions, e.g. {suggested}."
        )


def _map_source_key(key: str) -> str | None:
    if key in _ALLOWED_SOURCE_ONLY:
        return None
    if key.startswith("ln_out."):
        return key

    match = _BLOCK_KEY_RE.match(key)
    if match is None:
        raise ValueError(f"Unexpected RWKV-7 checkpoint key: {key}")

    layer_id = int(match.group(1))
    suffix = match.group(2)

    if suffix.startswith("att."):
        return f"layers.{layer_id}.time_mix.{suffix[len('att.') :]}"
    if suffix.startswith("ffn."):
        return f"layers.{layer_id}.channel_mix.{suffix[len('ffn.') :]}"
    if suffix.startswith("ln1."):
        return f"layers.{layer_id}.ln1.{suffix[len('ln1.') :]}"
    if suffix.startswith("ln2."):
        return f"layers.{layer_id}.ln2.{suffix[len('ln2.') :]}"
    if suffix.startswith("ln0."):
        if layer_id == 0:
            return f"layers.0.ln0.{suffix[len('ln0.') :]}"
        # Upstream instantiates ln0 on every block, but only block 0 uses it.
        return None

    raise ValueError(f"Unexpected RWKV-7 checkpoint key: {key}")


def load_pretrained_backbone(
    backbone: RWKV7Backbone,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    """Map and strictly load a stock x070 checkpoint into the local backbone."""
    if model_cfg.architecture != "rwkv":
        raise ValueError("RWKV-7 checkpoints can only initialize architecture='rwkv'.")
    if model_cfg.init_mode != "pretrained":
        raise ValueError("load_pretrained_backbone requires init_mode='pretrained'.")
    if not model_cfg.rwkv_checkpoint:
        raise ValueError(
            "Pretrained RWKV initialization requires an explicit rwkv_checkpoint path."
        )

    checkpoint_path = Path(model_cfg.rwkv_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RWKV-7 checkpoint not found: {checkpoint_path}")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        model_cfg.rwkv_checkpoint_sha256 is not None
        and checkpoint_sha256 != model_cfg.rwkv_checkpoint_sha256
    ):
        raise ValueError(
            "RWKV-7 checkpoint contents changed after run configuration was "
            "resolved: expected SHA-256 "
            f"{model_cfg.rwkv_checkpoint_sha256}, got {checkpoint_sha256}."
        )

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    source_config = infer_checkpoint_architecture(state_dict)
    _validate_target_config(model_cfg, source_config)

    target_state = backbone.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    ignored_source_keys: list[str] = []

    for source_key, value in state_dict.items():
        target_key = _map_source_key(source_key)
        if target_key is None:
            ignored_source_keys.append(source_key)
            continue
        if target_key not in target_state:
            raise ValueError(
                f"Unexpected RWKV-7 checkpoint key {source_key}: "
                f"maps to unknown target key {target_key}."
            )
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            raise ValueError(
                f"Checkpoint shape mismatch for {source_key} -> {target_key}: "
                f"checkpoint={tuple(value.shape)}, "
                f"target={tuple(target_state[target_key].shape)}."
            )
        if target_key in mapped:
            raise ValueError(f"Multiple checkpoint keys map to target key {target_key}.")
        mapped[target_key] = value

    missing_target_keys = set(target_state) - set(mapped)
    unsupported_missing = missing_target_keys - _OPTIONAL_UNUSED_TARGET_KEYS
    if unsupported_missing:
        raise ValueError(
            "RWKV-7 checkpoint does not cover the complete local backbone. "
            f"Missing target keys: {sorted(unsupported_missing)}"
        )

    retained_target_defaults = sorted(
        missing_target_keys & _OPTIONAL_UNUSED_TARGET_KEYS
    )
    for key in retained_target_defaults:
        # Upstream stock x070 checkpoints may omit first-layer v-residual
        # parameters because layer 0 never consumes them. Populate the mapped
        # dictionary with the local initialized values so strict loading still
        # proves an exact target key set.
        mapped[key] = target_state[key]

    if set(mapped) != set(target_state):
        raise AssertionError("Mapped checkpoint key set does not equal target key set.")

    backbone.load_state_dict(mapped, strict=True)

    return {
        "mode": "pretrained",
        "pretrained_scope": "backbone_only",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_architecture": source_config,
        "target_architecture": {
            "architecture": "rwkv7_x070",
            "num_hidden_layers": model_cfg.num_hidden_layers,
            "hidden_size": model_cfg.hidden_size,
            "intermediate_size": model_cfg.intermediate_size,
            "head_dim": model_cfg.head_dim,
            "num_attention_heads": model_cfg.num_attention_heads,
        },
        "strict_backbone_load": True,
        "ignored_source_keys": sorted(ignored_source_keys),
        "retained_target_defaults": retained_target_defaults,
    }
