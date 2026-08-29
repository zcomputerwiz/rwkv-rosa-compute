#!/usr/bin/env python3
"""Export training checkpoints as an evaluation-only archive.

The archive carries weights and nothing that could resume training. Source
checkpoints are read, never moved or deleted: the local full checkpoint stays
the only resumable copy.

    python scripts/export_eval_archive.py \
        --out-dir transfer/run_d6d23abc_seed_44 \
        --checkpoint 2=results/.../epoch_002.pt \
        --checkpoint 5=results/.../epoch_005.pt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from rosa_compute.eval_archive import ArchiveError, SourceRun, export_snapshots


def _git(*args: str) -> str | None:
    try:
        res = subprocess.run(["git", *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _epoch_path(spec: str) -> tuple[int, Path]:
    epoch, _, path = spec.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected EPOCH=PATH, got {spec!r}")
    return int(epoch), Path(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=_epoch_path, action="append",
                        required=True, metavar="EPOCH=PATH",
                        help="repeatable; one full training checkpoint per epoch")
    parser.add_argument("--run-id", default=None,
                        help="defaults to the checkpoint's signature.run_id")
    parser.add_argument("--seed", type=int, default=None,
                        help="defaults to the checkpoint's recorded seed")
    parser.add_argument("--allow-unverified-identity", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="allow export if checkpoint lacks a verified run_id/signature; "
                             "the identity will be recorded as unverified")
    args = parser.parse_args(argv)

    sources = dict(args.checkpoint)
    first = sources[min(sources)]
    payload = torch.load(first, map_location="cpu", weights_only=False)
    signature = payload.get("signature", {}) if isinstance(payload, dict) else {}

    run_id = args.run_id or signature.get("run_id")
    seed = args.seed
    if seed is None:
        seeds = (signature.get("evaluation") or {}).get("seeds_run") or []
        seed = seeds[0] if len(seeds) == 1 else None
    if run_id is None or seed is None:
        parser.error(
            "could not read run_id/seed from the checkpoint signature; "
            "pass --run-id and --seed explicitly"
        )

    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    run = SourceRun(
        run_id=run_id,
        seed=seed,
        commit=None if commit is None else commit + ("-dirty" if dirty else ""),
        producer="scripts/export_eval_archive.py",
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )

    try:
        manifest = export_snapshots(
            source_checkpoints=sources, out_dir=args.out_dir, run=run,
            allow_unverified_identity=args.allow_unverified_identity
        )
    except ArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"archive READY: {manifest.parent}")
    for epoch in sorted(sources):
        print(f"  epoch {epoch:>3}  model_epoch_{epoch:03d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
