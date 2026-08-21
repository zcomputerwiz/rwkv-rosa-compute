#!/usr/bin/env python3
"""Compare compatible Experiment-0 checkpoint diagnostic artifacts."""

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

from exp0.error_comparison import (  # noqa: E402
    compare_diagnostic_artifacts,
    load_diagnostic_artifact,
    write_comparison_artifact,
)


def _reference_indices(value: str | None, repeated: list[int]) -> list[int] | None:
    indices = list(repeated)
    if value is None:
        return sorted(set(indices)) if indices else None
    path = Path(value)
    if path.is_file():
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("indices", payload.get("error_indices"))
        if not isinstance(payload, list):
            raise ValueError(
                "Reference JSON must be a list or contain 'indices'/'error_indices'."
            )
        indices.extend(int(index) for index in payload)
    else:
        indices.extend(
            int(index.strip()) for index in value.split(",") if index.strip()
        )
    return sorted(set(indices))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        type=Path,
        nargs="+",
        help="Two or more checkpoint diagnostic JSON artifacts.",
    )
    parser.add_argument(
        "--reference-errors",
        help="JSON path or comma-separated generic reference error indices.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        action="append",
        default=[],
        help="Additional reference error index; may be repeated.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    if len(args.artifacts) < 2:
        raise ValueError("At least two diagnostic artifacts are required.")
    artifacts = [load_diagnostic_artifact(path) for path in args.artifacts]
    reference = _reference_indices(args.reference_errors, args.reference_index)
    comparison = compare_diagnostic_artifacts(
        artifacts,
        reference_indices=reference,
    )
    output = write_comparison_artifact(comparison, args.out)
    print(f"Comparison artifact written to {output}")
    print("canonical validation")
    for row in comparison["canonical_validation"]["per_seed"]:
        print(
            f"  seed {row['seed']}: errors={row['error_count']}/"
            f"{row['population_size']} rate={row['error_rate']:.6f}"
        )
    for row in comparison["canonical_validation"]["pairwise"]:
        print(
            f"  {row['seed_a']}/{row['seed_b']}: "
            f"observed={row['observed_intersection']} "
            f"expected={row['expected_independent_intersection']:.3f} "
            f"enrichment={row['observed_expected_enrichment']:.3f} "
            f"J={row['jaccard']:.4f} "
            f"p={row['hypergeometric_upper_tail_p']:.6g}"
        )
    frequencies = comparison["canonical_validation"]["miss_frequency"]
    print(
        "  miss frequencies: "
        + ", ".join(f"{count} seed(s)={total}" for count, total in frequencies.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
