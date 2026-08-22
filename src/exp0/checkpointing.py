"""Checkpoint and exact-resume primitives for Experiment 0 training."""

from __future__ import annotations

import os
import random
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Sampler

from exp0.config import DATALOADER_NEUTRAL_FIELDS

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


# Fields inside the "training" section that describe how work is scheduled
# rather than what is computed. A checkpoint whose only disagreement is here
# describes the same experiment, so it is accepted with a warning instead of
# being rejected.
CHECKPOINT_TOLERATED_FIELDS = frozenset(DATALOADER_NEUTRAL_FIELDS)


def _training_disagreements(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[str]:
    """Field names differing inside the training section, tolerated or not."""
    saved_training = saved.get("training") or {}
    expected_training = expected.get("training") or {}
    if not isinstance(saved_training, Mapping) or not isinstance(
        expected_training, Mapping
    ):
        return ["training"]
    keys = set(saved_training) | set(expected_training)
    return [
        key
        for key in sorted(keys)
        if saved_training.get(key) != expected_training.get(key)
    ]


def _resume_differs_only_by_seed_list(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
    differing: Sequence[str],
) -> bool:
    """True when the disagreement is exactly a single-seed resume of a sweep.

    ``--resume_checkpoint`` accepts one seed, so resuming a seed of a run
    launched with several necessarily changes ``seeds_run`` and the ``run_id``
    hashed from it. That is legitimate; a changed ``eval_seed`` or
    ``val_samples`` is not, and those are the only other run_id inputs.

    Returns False for checkpoints written before the evaluation section existed,
    so a legacy checkpoint is never waved through on an unverifiable claim.
    """
    if not set(differing) <= {"run_id", "evaluation"}:
        return False
    saved_eval = saved.get("evaluation")
    expected_eval = expected.get("evaluation")
    if not isinstance(saved_eval, Mapping) or not isinstance(expected_eval, Mapping):
        return False
    for key in ("eval_seed", "val_samples"):
        if saved_eval.get(key) != expected_eval.get(key):
            return False
    # Every seed being resumed must have been part of the original sweep.
    saved_seeds = saved_eval.get("seeds_run")
    expected_seeds = expected_eval.get("seeds_run")
    if not isinstance(saved_seeds, (list, tuple)) or not isinstance(
        expected_seeds, (list, tuple)
    ):
        return False
    return bool(expected_seeds) and set(expected_seeds) <= set(saved_seeds)


def validate_checkpoint_signature(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Reject resumes whose scientific/training configuration changed.

    DataLoader settings are exempt. They cannot change what a run computes, so
    refusing to resume a run because it now wants more workers would be an
    obstacle with no scientific content. Such a checkpoint is accepted with a
    warning.

    ``run_id`` is NOT exempt, and the reason is worth recording. It is tempting
    to argue that if every signature section matches, a differing run_id can
    only have come from the DataLoader fields - but the run_id is also a hash of
    ``seeds_run``, ``eval_seed`` and ``val_samples``, none of which appear in
    this signature. A single-seed resume of a three-seed run therefore produces
    a different run_id with every recorded section identical, and exempting the
    run_id would wave that through while claiming the science matched. Since
    DataLoader fields are now normalized out of the run_id, a genuine
    worker-count change leaves it untouched and needs no exemption.
    """
    if dict(saved) == dict(expected):
        return

    keys = sorted(set(saved) | set(expected))
    differing = [key for key in keys if saved.get(key) != expected.get(key)]

    # Resuming ONE seed of a multi-seed run legitimately changes seeds_run, and
    # therefore run_id. Both are exempt together, and only when eval_seed and
    # val_samples match - those are the other run_id inputs, and without this
    # section nothing else in the signature covers them.
    if _resume_differs_only_by_seed_list(saved, expected, differing):
        differing = [
            key for key in differing if key not in ("run_id", "evaluation")
        ]
        warnings.warn(
            "Resuming a single seed of a multi-seed run: seeds_run and the "
            "run_id derived from it differ from the checkpoint. eval_seed and "
            "val_samples matched exactly, and every other signature section is "
            "unchanged, so the resume is allowed.",
            RuntimeWarning,
            stacklevel=2,
        )
        if not differing:
            return

    training_diffs = (
        _training_disagreements(saved, expected) if "training" in differing else []
    )
    only_training_differs = differing == ["training"]
    if (
        only_training_differs
        and training_diffs
        and set(training_diffs) <= CHECKPOINT_TOLERATED_FIELDS
    ):
        warnings.warn(
            "Resuming a checkpoint whose DataLoader settings differ from the "
            f"requested run ({', '.join(training_diffs)}). These do not change "
            "what the run computes, so the resume is allowed and the signature "
            "is repaired.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

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
