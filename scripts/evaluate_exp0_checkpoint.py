#!/usr/bin/env python3
"""Evaluate a completed Experiment-0 checkpoint without resuming training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exp0.checkpoint_analysis import (  # noqa: E402
    evaluate_checkpoint,
    write_diagnostic_artifact,
)


def _report_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    evaluation = report.get("run_config", {}).get("evaluation", {})
    metadata: dict[str, Any] = {
        "run_report": str(path.resolve()),
        "run_config": report.get("run_config"),
        "eval_seed": report.get("eval_seed", evaluation.get("eval_seed")),
        "val_samples": report.get("val_samples", evaluation.get("val_samples")),
        "seeds_run": report.get("seeds_run", evaluation.get("seeds_run")),
    }
    challenge_sections = report.get("construction_diagnostics", {}).get(
        "diagnostic_challenge_validation", []
    )
    if challenge_sections:
        specs = {
            json.dumps(
                section.get("provenance", {}).get("spec", {}),
                sort_keys=True,
            )
            for section in challenge_sections
        }
        if len(specs) != 1:
            raise ValueError(
                "Run report contains inconsistent challenge specifications."
            )
        spec = json.loads(specs.pop())
        metadata["challenge_seed"] = spec.get("seed")
        metadata["challenge_per_class"] = spec.get("per_stratum")
        metadata["challenge_id"] = (
            challenge_sections[0].get("provenance", {}).get("challenge_id")
        )
    return metadata


def _resolve(
    name: str,
    cli_value: Any,
    report_value: Any,
    *,
    required: bool,
) -> tuple[Any, str | None]:
    if cli_value is not None and report_value is not None and cli_value != report_value:
        raise ValueError(
            f"--{name}={cli_value!r} conflicts with run-report value {report_value!r}."
        )
    if cli_value is not None:
        return cli_value, "cli"
    if report_value is not None:
        return report_value, "run_report"
    if required:
        raise ValueError(
            f"--{name} is required because it cannot be recovered from the checkpoint."
        )
    return None, None


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--run_report",
        type=Path,
        help="Optional completed run report supplying evaluation provenance.",
    )
    parser.add_argument("--eval_seed", type=int)
    parser.add_argument("--val_samples", type=int)
    parser.add_argument(
        "--construction_diagnostics",
        action="store_true",
        help=(
            "Record complete construction diagnostics. Standalone artifacts always "
            "contain canonical per-instance records; this flag documents intent."
        ),
    )
    parser.add_argument("--challenge_per_class", type=int)
    parser.add_argument("--challenge_seed", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    report = _report_metadata(args.run_report)
    eval_seed, eval_source = _resolve(
        "eval_seed", args.eval_seed, report.get("eval_seed"), required=True
    )
    val_samples, val_source = _resolve(
        "val_samples", args.val_samples, report.get("val_samples"), required=True
    )
    challenge_per_class, challenge_count_source = _resolve(
        "challenge_per_class",
        args.challenge_per_class,
        report.get("challenge_per_class"),
        required=False,
    )
    challenge_seed, challenge_seed_source = _resolve(
        "challenge_seed",
        args.challenge_seed,
        report.get("challenge_seed"),
        required=bool(challenge_per_class),
    )
    if val_samples <= 0:
        raise ValueError("val_samples must be positive.")
    if challenge_per_class is not None and challenge_per_class < 0:
        raise ValueError("challenge_per_class must be non-negative.")

    provenance = {
        "eval_seed_source": eval_source,
        "val_samples_source": val_source,
        "challenge_per_class_source": challenge_count_source,
        "challenge_seed_source": challenge_seed_source,
        "construction_diagnostics_requested": args.construction_diagnostics,
    }
    if args.run_report is not None:
        provenance["run_report"] = str(args.run_report.resolve())

    artifact = evaluate_checkpoint(
        args.checkpoint,
        device=args.device,
        eval_seed=eval_seed,
        val_samples=val_samples,
        challenge_per_class=challenge_per_class or 0,
        challenge_seed=challenge_seed,
        batch_size=args.batch_size,
        precision=args.precision,
        evaluation_provenance=provenance,
        expected_run_config=report.get("run_config"),
    )
    challenge = artifact.get("diagnostic_challenge_validation")
    if challenge is not None and report.get("challenge_id") is not None:
        if challenge["challenge_id"] != report["challenge_id"]:
            raise ValueError(
                "Regenerated challenge ID does not match run report: "
                f"{challenge['challenge_id']} != {report['challenge_id']}"
            )

    output = write_diagnostic_artifact(artifact, args.out)
    canonical = artifact["canonical_validation"]
    print(f"Diagnostic artifact written to {output}")
    print(
        f"seed={artifact['checkpoint']['training_seed']} "
        f"canonical_id={canonical['canonical_validation_id']} "
        f"accuracy={canonical['accuracy']:.6f} "
        f"errors={canonical['error_count']}/{canonical['population_size']}"
    )
    if challenge is not None:
        print(
            f"challenge_id={challenge['challenge_id']} "
            f"accuracy={challenge['accuracy']:.6f} "
            f"errors={challenge['error_count']}/{challenge['population_size']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

