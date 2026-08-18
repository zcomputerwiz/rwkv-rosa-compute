import hashlib
import os
from typing import Any, Dict, Optional, Tuple

import torch

from .config import DEFAULT_CONFIG, ROSAConfig


def compute_checkpoint_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a checkpoint file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def validate_checkpoint_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: ROSAConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Validates state dict parameter names and tensor shapes against expected ROSA config.
    Returns diagnostic dict with missing keys, unexpected keys, and tensor shapes.
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

        expected_shapes[f"{prefix}rosa.x_q"] = (1, 1, config.n_embd)
        expected_shapes[f"{prefix}rosa.x_k"] = (1, 1, config.n_embd)
        expected_shapes[f"{prefix}rosa.x_v"] = (1, 1, config.n_embd)
        expected_shapes[f"{prefix}rosa.q.weight"] = (config.n_embd, config.n_embd)
        expected_shapes[f"{prefix}rosa.k.weight"] = (config.n_embd, config.n_embd)
        expected_shapes[f"{prefix}rosa.v.weight"] = (config.n_embd, config.n_embd)
        expected_shapes[f"{prefix}rosa.o.weight"] = (config.n_embd, config.n_embd)
        expected_shapes[f"{prefix}rosa.rosa_qkv.emb"] = (1, 1, config.n_embd)

    present_keys = set(state_dict.keys())
    expected_keys = set(expected_shapes.keys())

    missing_keys = list(expected_keys - present_keys)
    unexpected_keys = list(present_keys - expected_keys)
    mismatched_shapes = {}

    for key in present_keys.intersection(expected_keys):
        tensor = state_dict[key]
        expected_shape = tuple(expected_shapes[key])
        if tuple(tensor.shape) != expected_shape:
            mismatched_shapes[key] = {
                "expected": expected_shape,
                "actual": tuple(tensor.shape)
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
    checkpoint_path: Optional[str] = None,
    config: ROSAConfig = DEFAULT_CONFIG,
    compute_hash: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Loads ROSA .pth checkpoint from filepath or ROSA_MODEL_PATH environment variable.
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
