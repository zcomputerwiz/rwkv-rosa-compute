"""Evaluation-only model snapshot archives, format version 1.

An evaluation archive carries model weights between machines and nothing else.
It deliberately cannot resume training: the optimizer, scheduler, scaler, and
RNG state are absent, the manifest says so in machine-readable form, and the
loader here is separate from the training resume path.

Layout, one run per archive:

    run_<run_id>_seed_<n>/
      MANIFEST.json
      model_epoch_001.pt
      model_epoch_005.pt

Identity is a canonical content digest over the tensors themselves, not over
Torch's container bytes, so re-serialization by a different Torch version does
not change a snapshot's identity. The digest commits to every key, dtype, and
shape directly rather than delegating to a separate schema field that a reader
might forget to check.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

FORMAT_VERSION = 1
CONTENT_DOMAIN = b"rosa-compute/eval-archive/content-v1\x00"
SCHEMA_DOMAIN = b"rosa-compute/eval-archive/schema-v1\x00"

# Every dtype we are willing to hash gets an explicit token in the digest and an
# integer type of the same width to read its bytes through. Reading through an
# integer view lets numpy apply an explicit little-endian byte order, which is
# what makes the digest independent of the host's endianness. bfloat16 has no
# numpy dtype at all, so the view is the only way to reach its bytes.
_DTYPE_TOKENS: dict[torch.dtype, tuple[str, torch.dtype, str]] = {
    torch.float64: ("f64", torch.int64, "<i8"),
    torch.float32: ("f32", torch.int32, "<i4"),
    torch.float16: ("f16", torch.int16, "<i2"),
    torch.bfloat16: ("bf16", torch.int16, "<i2"),
    torch.int64: ("i64", torch.int64, "<i8"),
    torch.int32: ("i32", torch.int32, "<i4"),
    torch.int16: ("i16", torch.int16, "<i2"),
    torch.int8: ("i8", torch.int8, "|i1"),
    torch.uint8: ("u8", torch.uint8, "|u1"),
    torch.bool: ("bool", torch.uint8, "|u1"),
}


class ArchiveError(Exception):
    """Any failure to produce, verify, or load an evaluation archive."""


def _length_prefixed(raw: bytes) -> bytes:
    return struct.pack("<I", len(raw)) + raw


def _tensor_bytes(tensor: torch.Tensor) -> tuple[str, bytes]:
    """Return the dtype token and explicitly little-endian bytes of a tensor."""
    if tensor.layout is not torch.strided:
        raise ArchiveError(
            f"unsupported tensor layout {tensor.layout!r}; "
            "evaluation archives store dense strided tensors only"
        )
    entry = _DTYPE_TOKENS.get(tensor.dtype)
    if entry is None:
        raise ArchiveError(
            f"unsupported dtype {tensor.dtype!r}; refusing to convert it "
            "numerically to fit the digest"
        )
    token, view_dtype, numpy_dtype = entry
    flat = tensor.detach().to("cpu").contiguous()
    # .view() reinterprets the same storage at equal element width, so no value
    # is converted; astype applies the byte order and copies only if the host
    # disagrees with it.
    raw = flat.view(view_dtype).numpy().astype(numpy_dtype, copy=False).tobytes()
    return token, raw


def _iter_state(state: Mapping[str, Any]) -> Iterable[tuple[str, torch.Tensor]]:
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ArchiveError(
                f"state entry {key!r} is {type(value).__name__}, not a tensor; "
                "evaluation archives store tensor state only"
            )
        yield key, value


def canonical_content_sha256(state: Mapping[str, Any]) -> str:
    """Digest committing to every key, dtype, shape, and tensor byte.

    The stream is, after a domain separator carrying the format version, for
    each key in sorted order: the length-prefixed UTF-8 key, the length-prefixed
    dtype token, the rank and every dimension as little-endian integers, the
    byte count, and the tensor bytes in little-endian order.
    """
    digest = hashlib.sha256()
    digest.update(CONTENT_DOMAIN)
    digest.update(struct.pack("<I", FORMAT_VERSION))
    for key, tensor in _iter_state(state):
        token, raw = _tensor_bytes(tensor)
        digest.update(_length_prefixed(key.encode("utf-8")))
        digest.update(_length_prefixed(token.encode("ascii")))
        digest.update(struct.pack("<I", tensor.dim()))
        for dim in tensor.shape:
            digest.update(struct.pack("<Q", dim))
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def schema_sha256(state: Mapping[str, Any]) -> str:
    """Digest over keys, dtypes, and shapes only. Diagnostic, not identity."""
    digest = hashlib.sha256()
    digest.update(SCHEMA_DOMAIN)
    digest.update(struct.pack("<I", FORMAT_VERSION))
    for key, tensor in _iter_state(state):
        token, _, _ = _DTYPE_TOKENS.get(tensor.dtype, (None, None, None))
        if token is None:
            raise ArchiveError(f"unsupported dtype {tensor.dtype!r}")
        digest.update(_length_prefixed(key.encode("utf-8")))
        digest.update(_length_prefixed(token.encode("ascii")))
        digest.update(struct.pack("<I", tensor.dim()))
        for dim in tensor.shape:
            digest.update(struct.pack("<Q", dim))
    return digest.hexdigest()


def decoded_size(state: Mapping[str, Any]) -> int:
    return sum(len(_tensor_bytes(t)[1]) for _, t in _iter_state(state))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


@dataclass(frozen=True)
class SourceRun:
    """Provenance of the training run a snapshot came from."""

    run_id: str
    seed: int
    commit: str | None = None
    producer: str | None = None
    device: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None


def export_snapshots(
    *,
    source_checkpoints: Mapping[int, Path],
    out_dir: Path,
    run: SourceRun,
) -> Path:
    """Write an evaluation-only archive for one run.

    ``source_checkpoints`` maps epoch number to the full training checkpoint it
    came from. Those files are read and never modified: export is not a move,
    and the local full checkpoint remains the only resumable copy.

    Entries are staged privately, published with their sidecars, and only then
    is ``MANIFEST.json`` written. A reader that finds the manifest therefore
    finds every entry it names.
    """
    if not source_checkpoints:
        raise ArchiveError("refusing to write an archive with no entries")

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ArchiveError(f"refusing to write into a non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for epoch in sorted(source_checkpoints):
        source = Path(source_checkpoints[epoch])
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
            raise ArchiveError(
                f"{source} does not look like a training checkpoint: "
                "no model_state_dict"
            )
        state = payload["model_state_dict"]

        content = canonical_content_sha256(state)
        schema = schema_sha256(state)
        decoded = decoded_size(state)

        name = f"model_epoch_{epoch:03d}.pt"
        staged = out_dir / f".{name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            # Only the tensors. A weights_only=True load must succeed on this
            # file, which it cannot if any other object rides along.
            with open(staged, "wb") as handle:
                torch.save(dict(state), handle)
                handle.flush()
                os.fsync(handle.fileno())
            target = out_dir / name
            os.replace(staged, target)
        finally:
            if staged.exists():
                staged.unlink()

        stored_sha = _write_sidecar(target)
        entries.append(
            {
                "epoch": epoch,
                "kind": "full",
                "path": name,
                "content_sha256": content,
                "schema_sha256": schema,
                "decoded_bytes": decoded,
                "stored_sha256": stored_sha,
                "stored_bytes": target.stat().st_size,
                "source_checkpoint_sha256": sha256_file(source),
                "codec": {"name": "torch_save", "version": 1, "parameters": {}},
                "base_epoch": None,
            }
        )

    manifest = {
        "format_version": FORMAT_VERSION,
        "purpose": "evaluation_only",
        "resume_capable": False,
        "contains_optimizer_state": False,
        "omitted_state": ["optimizer", "scheduler", "scaler", "rng"],
        "run_id": run.run_id,
        "seed": run.seed,
        "commit": run.commit,
        "producer": run.producer,
        "device": run.device,
        "torch_version": run.torch_version,
        "cuda_version": run.cuda_version,
        "entries": entries,
    }

    # Written last, so the manifest's existence is the READY signal.
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_sidecar(manifest_path)
    return manifest_path


def read_manifest(archive_dir: Path) -> dict[str, Any]:
    archive_dir = Path(archive_dir)
    manifest_path = archive_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ArchiveError(f"archive is not READY: no MANIFEST.json in {archive_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ArchiveError(
            f"unsupported archive format_version {version!r}; expected {FORMAT_VERSION}"
        )
    if manifest.get("purpose") != "evaluation_only" or manifest.get("resume_capable"):
        raise ArchiveError("manifest does not declare an evaluation-only archive")
    return manifest


def _entry_for(manifest: Mapping[str, Any], epoch: int) -> Mapping[str, Any]:
    for entry in manifest["entries"]:
        if entry["epoch"] == epoch:
            return entry
    have = sorted(e["epoch"] for e in manifest["entries"])
    raise ArchiveError(f"archive has no epoch {epoch}; it has {have}")


def verify_archive(archive_dir: Path) -> dict[str, Any]:
    """Check every manifest entry exists and matches, without loading tensors.

    Cheap enough to run before a transfer or before allocating anything.
    """
    archive_dir = Path(archive_dir)
    manifest = read_manifest(archive_dir)
    for entry in manifest["entries"]:
        name = entry["path"]
        if "/" in name or "\\" in name or name.startswith("."):
            raise ArchiveError(f"manifest entry path is not a plain filename: {name!r}")
        path = archive_dir / name
        if not path.is_file():
            raise ArchiveError(f"archive is not READY: missing entry {name}")
        size = path.stat().st_size
        if size != entry["stored_bytes"]:
            raise ArchiveError(
                f"{name}: stored size {size} does not match manifest "
                f"{entry['stored_bytes']}"
            )
        actual = sha256_file(path)
        if actual != entry["stored_sha256"]:
            raise ArchiveError(
                f"{name}: stored digest {actual} does not match manifest "
                f"{entry['stored_sha256']}"
            )
    return manifest


def load_snapshot(
    archive_dir: Path,
    epoch: int,
    *,
    expect_source_checkpoint_sha256: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load one epoch's weights, failing closed on any mismatch.

    Separate from the training resume path by construction: it returns a state
    dict and nothing that could restart an optimizer.
    """
    archive_dir = Path(archive_dir)
    manifest = verify_archive(archive_dir)
    entry = _entry_for(manifest, epoch)

    if (
        expect_source_checkpoint_sha256 is not None
        and entry["source_checkpoint_sha256"] != expect_source_checkpoint_sha256
    ):
        raise ArchiveError(
            f"epoch {epoch} came from source checkpoint "
            f"{entry['source_checkpoint_sha256']}, not the expected "
            f"{expect_source_checkpoint_sha256}"
        )

    state = torch.load(
        archive_dir / entry["path"], map_location="cpu", weights_only=True
    )
    if not isinstance(state, dict):
        raise ArchiveError(f"epoch {epoch} payload is not a state dict")

    schema = schema_sha256(state)
    if schema != entry["schema_sha256"]:
        raise ArchiveError(
            f"epoch {epoch} schema digest {schema} does not match manifest "
            f"{entry['schema_sha256']}"
        )
    decoded = decoded_size(state)
    if decoded != entry["decoded_bytes"]:
        raise ArchiveError(
            f"epoch {epoch} decoded size {decoded} does not match manifest "
            f"{entry['decoded_bytes']}"
        )
    content = canonical_content_sha256(state)
    if content != entry["content_sha256"]:
        raise ArchiveError(
            f"epoch {epoch} content digest {content} does not match manifest "
            f"{entry['content_sha256']}"
        )
    return state
