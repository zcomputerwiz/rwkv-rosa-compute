import hashlib
import os
from typing import Any

import torch

from .config import DEFAULT_CONFIG, ROSAConfig


def compute_checkpoint_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a checkpoint file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def inspect_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Inspects an actual PyTorch checkpoint file on disk without loading model code."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    file_size_bytes = os.path.getsize(checkpoint_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_hash = compute_checkpoint_hash(checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        state_dict = getattr(state_dict, "state_dict", lambda: {})()

    tensor_info = {}
    total_params = 0
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            num_el = val.numel()
            total_params += num_el
            tensor_info[key] = {
                "shape": tuple(val.shape),
                "dtype": str(val.dtype),
                "numel": num_el,
            }

    return {
        "checkpoint_path": checkpoint_path,
        "file_size_mb": file_size_mb,
        "sha256": file_hash,
        "num_tensors": len(tensor_info),
        "total_parameters": total_params,
        "tensors": tensor_info,
    }


def validate_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: ROSAConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Validates state dict parameter names and tensor shapes against target 0.1B ROSA config.

    Based on RWKV-v8/260222_rosa4bitLM_L12.py parameter schema.
    """
    expected_shapes = {
        "emb.weight": (config.vocab_size, config.n_embd),
        "ln_out.weight": (config.n_embd,),
        "ln_out.bias": (config.n_embd,),
        "head.weight": (config.vocab_size, config.n_embd),
    }

    for i in range(config.n_layer):
        prefix = f"blocks.{i}."
        expected_shapes[f"{prefix}ln2.weight"] = (config.n_embd,)
        expected_shapes[f"{prefix}ln2.bias"] = (config.n_embd,)
        expected_shapes[f"{prefix}ln3.weight"] = (config.n_embd,)
        expected_shapes[f"{prefix}ln3.bias"] = (config.n_embd,)
        if i == 0:
            expected_shapes[f"{prefix}ln0.weight"] = (config.n_embd,)
            expected_shapes[f"{prefix}ln0.bias"] = (config.n_embd,)

        expected_shapes[f"{prefix}ffn.x_k"] = (1, 1, config.n_embd)
        expected_shapes[f"{prefix}ffn.key.weight"] = (config.n_embd * 4, config.n_embd)
        expected_shapes[f"{prefix}ffn.value.weight"] = (config.n_embd, config.n_embd * 4)

        expected_shapes[f"{prefix}rosa.time_shift"] = None  # Non-parameter module, or zero-pad
        expected_shapes[f"{prefix}rosa.x_q"] = (1, 1, config.n_embd)
        expected_shapes[f"{prefix}rosa.x_k"] = (1, 1, config.n_embd)
        expected_shapes[f"{prefix}rosa.x_v"] = (1, 1, config.n_embd)
        expected_shapes[f"{prefix}rosa.q.weight"] = (config.n_embd, config.n_embd)
        expected_shapes[f"{prefix}rosa.k.weight"] = (config.n_embd, config.n_embd)
        expected_shapes[f"{prefix}rosa.v.weight"] = (config.n_embd, config.n_embd)
        expected_shapes[f"{prefix}rosa.rosa_qkv.emb"] = (1, 1, config.n_embd)
        expected_shapes[f"{prefix}rosa.o.weight"] = (config.n_embd, config.n_embd)

    # Filter out None entries (modules without parameters)
    expected_shapes = {k: v for k, v in expected_shapes.items() if v is not None}

    present_keys = set(state_dict.keys())
    expected_keys = set(expected_shapes.keys())

    missing_keys = sorted(list(expected_keys - present_keys))
    unexpected_keys = sorted(list(present_keys - expected_keys))
    mismatched_shapes = {}

    for key in present_keys.intersection(expected_keys):
        tensor = state_dict[key]
        expected_shape = tuple(expected_shapes[key])
        if tuple(tensor.shape) != expected_shape:
            mismatched_shapes[key] = {
                "expected": expected_shape,
                "actual": tuple(tensor.shape),
            }

    tensor_shapes = {k: tuple(v.shape) for k, v in state_dict.items()}

    return {
        "is_valid": len(missing_keys) == 0 and len(unexpected_keys) == 0 and len(mismatched_shapes) == 0,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "mismatched_shapes": mismatched_shapes,
        "tensor_shapes": tensor_shapes,
    }


def load_rosa_checkpoint(
    checkpoint_path: str | None = None,
    config: ROSAConfig = DEFAULT_CONFIG,
    compute_hash: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Loads ROSA .pth checkpoint from filepath or ROSA_MODEL_PATH environment variable.

    Returns (state_dict, validation_info).
    """
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("ROSA_MODEL_PATH")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    validation_info = validate_checkpoint_state_dict(state_dict, config=config)

    if compute_hash:
        validation_info["sha256"] = compute_checkpoint_hash(checkpoint_path)

    return state_dict, validation_info
