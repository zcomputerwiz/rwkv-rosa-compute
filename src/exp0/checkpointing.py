"""Checkpoint and exact-resume primitives for Experiment 0 training."""

from __future__ import annotations

import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Sampler

CHECKPOINT_VERSION = 1


class ResumableRandomSampler(Sampler[int]):
    """Random permutation sampler that can restart at a sample offset.

    The permutation is generated from an explicit epoch seed rather than from
    mutable global RNG state.  A checkpoint therefore only needs the epoch seed
    and number of samples already consumed to replay the exact remainder of an
    interrupted epoch.  DataLoader prefetching is harmless: prefetched but
    unconsumed indices are regenerated after the saved offset on resume.
    """

    def __init__(
        self,
        data_source: Sequence[Any],
        *,
        epoch_seed: int = 0,
        start_index: int = 0,
    ) -> None:
        self.data_source = data_source
        self.set_state(epoch_seed=epoch_seed, start_index=start_index)

    def set_state(self, *, epoch_seed: int, start_index: int) -> None:
        size = len(self.data_source)
        if start_index < 0 or start_index > size:
            raise ValueError(
                f"start_index must be in [0, {size}], got {start_index}."
            )
        self.epoch_seed = int(epoch_seed)
        self.start_index = int(start_index)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch_seed)
        order = torch.randperm(len(self.data_source), generator=generator)
        if self.start_index:
            order = order[self.start_index :]
        yield from order.tolist()

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


def epoch_shuffle_seed(base_seed: int, epoch: int) -> int:
    """Derive a stable positive 63-bit shuffle seed from run seed and epoch."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative.")
    mask = (1 << 64) - 1
    mixed = (int(base_seed) & mask) ^ (((epoch + 1) * 0x9E3779B97F4A7C15) & mask)
    return mixed % ((1 << 63) - 1)


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state that can affect model-side stochastic computation."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])

    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable."
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Checkpoint CUDA RNG device count does not match this process: "
                f"checkpoint={len(cuda_states)}, current={torch.cuda.device_count()}."
            )
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Durably write a torch checkpoint and atomically replace the destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )

    try:
        with open(temporary, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def atomic_copy(source: str | Path, destination: str | Path) -> Path:
    """Atomically copy a completed checkpoint to another path."""
    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        shutil.copyfile(source_path, temporary)
        with open(temporary, "rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trusted Experiment 0 training checkpoint on CPU."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Training checkpoint root must be a mapping.")
    version = payload.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported Experiment 0 checkpoint version: "
            f"expected {CHECKPOINT_VERSION}, found {version!r}."
        )
    return payload


def validate_checkpoint_signature(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Reject resumes whose scientific/training configuration changed."""
    if dict(saved) == dict(expected):
        return

    keys = sorted(set(saved) | set(expected))
    differing = [key for key in keys if saved.get(key) != expected.get(key)]
    raise ValueError(
        "Training checkpoint does not match the requested run. "
        "Differing signature sections: " + ", ".join(differing)
    )


def optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move optimizer tensor state to the parameter device after CPU loading."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
