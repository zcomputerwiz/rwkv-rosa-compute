"""Acceptance checks for the v1 evaluation-only archive.

These are the five checks required by REVIEW_CHECKPOINT_ARCHIVE_FORMAT_SHANNON:
round-trip identity, fail-closed on four corruptions, the evaluation loader
accepting a snapshot, the training-resume loader refusing one, and the archive
not reading as READY until the manifest and its sidecar land last.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from exp0.checkpointing import load_training_checkpoint
from rosa_compute.eval_archive import (
    ArchiveError,
    SourceRun,
    canonical_content_sha256,
    export_snapshots,
    load_snapshot,
    read_manifest,
    verify_archive,
)

RUN = SourceRun(run_id="d6d23abcab7a898b", seed=44, commit="0" * 40,
                producer="tests/test_eval_archive.py", device="cpu",
                torch_version=torch.__version__, cuda_version=None)


def make_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "backbone.0.weight": torch.randn(4, 3),
        "backbone.0.bias": torch.randn(4),
        "head.weight": torch.randn(2, 4).to(torch.bfloat16),
        "counts": torch.tensor([1, 2, 3], dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
    }


def write_source(
    path: Path,
    state: dict[str, torch.Tensor],
    *,
    run_id: str = RUN.run_id,
    seeds: tuple[int, ...] = (RUN.seed,),
) -> Path:
    """A full training checkpoint, complete with the state export must drop."""
    torch.save(
        {
            "checkpoint_version": 1,
            "model_state_dict": state,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "lr_scheduler_state_dict": {"last_epoch": 5},
            "scaler_state_dict": None,
            "rng_state": {"python": None},
            "progress": {"epoch": 5},
            "initialization": {},
            "signature": {
                "run_id": run_id,
                "evaluation": {"seeds_run": list(seeds)},
            },
        },
        path,
    )
    return path


def rewrite_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def rewrite_manifest(out: Path, manifest: dict) -> None:
    path = out / "MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    rewrite_sidecar(path)


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor], Path]:
    state = make_state()
    source = write_source(tmp_path / "epoch_005.pt", state)
    out = tmp_path / f"run_{RUN.run_id}_seed_{RUN.seed}"
    export_snapshots(source_checkpoints={5: source}, out_dir=out, run=RUN)
    return out, state, source


# 1. round trip


def test_round_trip_is_bit_identical_by_content_digest(archive):
    out, state, _ = archive
    loaded = load_snapshot(out, 5)
    assert canonical_content_sha256(loaded) == canonical_content_sha256(state)
    for key in state:
        assert torch.equal(loaded[key], state[key]), key


def test_content_digest_commits_to_dtype_not_just_values():
    a = {"w": torch.ones(4, dtype=torch.float32)}
    b = {"w": torch.ones(4, dtype=torch.float16)}
    assert canonical_content_sha256(a) != canonical_content_sha256(b)


def test_content_digest_commits_to_shape():
    flat = torch.arange(6, dtype=torch.float32)
    assert (canonical_content_sha256({"w": flat})
            != canonical_content_sha256({"w": flat.reshape(2, 3)}))


def test_content_digest_commits_to_key_boundaries():
    """Concatenating names must not collide: 'ab'+'c' is not 'a'+'bc'."""
    t = torch.zeros(1)
    assert (canonical_content_sha256({"ab": t, "c": t})
            != canonical_content_sha256({"a": t, "bc": t}))


def test_unsupported_dtype_is_rejected_not_converted():
    with pytest.raises(ArchiveError, match="unsupported dtype"):
        canonical_content_sha256({"w": torch.ones(2, dtype=torch.complex64)})


def test_non_tensor_state_is_rejected():
    with pytest.raises(ArchiveError, match="not a tensor"):
        canonical_content_sha256({"w": [1, 2, 3]})


# 2. fail closed


def test_flipped_stored_byte_fails_closed(archive):
    out, _, _ = archive
    target = out / "model_epoch_005.pt"
    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    target.write_bytes(bytes(raw))
    with pytest.raises(ArchiveError, match="sidecar digest"):
        load_snapshot(out, 5)


def test_missing_entry_fails_closed(archive):
    out, _, _ = archive
    (out / "model_epoch_005.pt").unlink()
    with pytest.raises(ArchiveError, match="missing entry"):
        verify_archive(out)


def test_wrong_schema_fails_closed(archive):
    """A payload whose keys changed must not load, even if the file is intact."""
    out, state, _ = archive
    renamed = dict(state)
    renamed["renamed"] = renamed.pop("counts")
    target = out / "model_epoch_005.pt"
    torch.save(renamed, target)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    # Re-point the stored digest so the schema check is what fires, not the
    # file digest.
    manifest["entries"][0]["stored_sha256"] = hashlib.sha256(
        target.read_bytes()).hexdigest()
    manifest["entries"][0]["stored_bytes"] = target.stat().st_size
    rewrite_sidecar(target)
    rewrite_manifest(out, manifest)
    with pytest.raises(ArchiveError, match="schema digest"):
        load_snapshot(out, 5)


def test_wrong_source_digest_fails_closed(archive):
    out, _, _ = archive
    with pytest.raises(ArchiveError, match="source checkpoint"):
        load_snapshot(out, 5, expect_source_checkpoint_sha256="0" * 64)


def test_correct_source_digest_is_accepted(archive):
    out, _, source = archive
    manifest = read_manifest(out)
    expected = manifest["entries"][0]["source_checkpoint_sha256"]
    assert load_snapshot(out, 5, expect_source_checkpoint_sha256=expected)


def test_absent_epoch_fails_closed(archive):
    out, _, _ = archive
    with pytest.raises(ArchiveError, match="no epoch 3"):
        load_snapshot(out, 3)


def test_manifest_path_traversal_is_rejected(archive):
    out, _, _ = archive
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["entries"][0]["path"] = "../escape.pt"
    rewrite_manifest(out, manifest)
    with pytest.raises(ArchiveError, match="plain filename"):
        verify_archive(out)


# 3. the evaluation loader accepts the snapshot


def test_a_real_module_loads_the_snapshot(tmp_path):
    module = torch.nn.Linear(3, 4)
    source = tmp_path / "epoch_001.pt"
    torch.save({"checkpoint_version": 1, "model_state_dict": module.state_dict()}, source)
    out = tmp_path / "arch"
    export_snapshots(source_checkpoints={1: source}, out_dir=out, run=RUN)

    fresh = torch.nn.Linear(3, 4)
    fresh.load_state_dict(load_snapshot(out, 1), strict=True)
    x = torch.randn(2, 3)
    assert torch.equal(fresh(x), module(x))


# 4. the training resume loader refuses the archive


def test_training_resume_loader_refuses_an_evaluation_snapshot(archive):
    out, _, _ = archive
    with pytest.raises(ValueError, match="checkpoint version"):
        load_training_checkpoint(out / "model_epoch_005.pt")


def test_the_snapshot_carries_no_resumable_state(archive):
    out, _, _ = archive
    payload = torch.load(out / "model_epoch_005.pt", map_location="cpu",
                         weights_only=True)
    for banned in ("optimizer_state_dict", "lr_scheduler_state_dict",
                   "scaler_state_dict", "rng_state", "progress"):
        assert banned not in payload


def test_manifest_declares_evaluation_only(archive):
    out, _, _ = archive
    manifest = read_manifest(out)
    assert manifest["purpose"] == "evaluation_only"
    assert manifest["resume_capable"] is False
    assert manifest["contains_optimizer_state"] is False
    assert manifest["omitted_state"] == ["optimizer", "scheduler", "scaler", "rng"]


# 5. publication order


@pytest.mark.parametrize("missing", ["MANIFEST.json", "MANIFEST.json.sha256"])
def test_archive_is_not_ready_until_the_manifest_pair_lands(archive, missing):
    out, _, _ = archive
    (out / missing).unlink()
    # The data entry and its sidecar are still there; without the manifest the
    # directory must not read as an archive at all. The sidecar is part of the
    # READY signal, so a manifest without its sidecar is also incomplete.
    assert (out / "model_epoch_005.pt").is_file()
    assert (out / "model_epoch_005.pt.sha256").is_file()
    with pytest.raises(ArchiveError, match="not READY"):
        verify_archive(out)


def test_archive_is_not_ready_without_an_entry_sidecar(archive):
    out, _, _ = archive
    (out / "model_epoch_005.pt.sha256").unlink()
    with pytest.raises(ArchiveError, match="not READY"):
        verify_archive(out)


def test_every_published_file_has_a_sidecar(archive):
    out, _, _ = archive
    payloads = [p for p in out.iterdir() if not p.name.endswith(".sha256")]
    for payload in payloads:
        sidecar = payload.with_name(payload.name + ".sha256")
        assert sidecar.is_file(), payload.name
        recorded, name = sidecar.read_text(encoding="utf-8").split()
        assert name == payload.name
        assert recorded == hashlib.sha256(payload.read_bytes()).hexdigest()


def test_export_refuses_a_non_empty_directory(tmp_path):
    state = make_state()
    source = write_source(tmp_path / "epoch_005.pt", state)
    out = tmp_path / "arch"
    out.mkdir()
    (out / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ArchiveError, match="non-empty"):
        export_snapshots(source_checkpoints={5: source}, out_dir=out, run=RUN)


def test_export_leaves_the_source_checkpoint_untouched(archive):
    _, _, source = archive
    payload = torch.load(source, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in payload
    assert payload["progress"] == {"epoch": 5}


def test_export_refuses_a_checkpoint_from_another_run(tmp_path):
    source = write_source(
        tmp_path / "epoch_005.pt", make_state(), run_id="another-run"
    )
    with pytest.raises(ArchiveError, match="belongs to run"):
        export_snapshots(
            source_checkpoints={5: source}, out_dir=tmp_path / "arch", run=RUN
        )


def test_export_refuses_a_checkpoint_from_another_seed(tmp_path):
    source = write_source(tmp_path / "epoch_005.pt", make_state(), seeds=(45,))
    with pytest.raises(ArchiveError, match="does not record seed"):
        export_snapshots(
            source_checkpoints={5: source}, out_dir=tmp_path / "arch", run=RUN
        )


def test_multiple_epochs_are_independently_loadable(tmp_path):
    a, b = make_state(), make_state()
    b["counts"] = torch.tensor([9, 9, 9], dtype=torch.int64)
    s1 = write_source(tmp_path / "e1.pt", a)
    s2 = write_source(tmp_path / "e5.pt", b)
    out = tmp_path / "arch"
    export_snapshots(source_checkpoints={1: s1, 5: s2}, out_dir=out, run=RUN)
    assert torch.equal(load_snapshot(out, 1)["counts"], a["counts"])
    assert torch.equal(load_snapshot(out, 5)["counts"], b["counts"])
    manifest = read_manifest(out)
    digests = {e["content_sha256"] for e in manifest["entries"]}
    assert len(digests) == 2
