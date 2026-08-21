#!/usr/bin/env python3
"""Summarize Experiment-0 construction-stratum diagnostics from a run report.

Console output is a convenience view. The JSON written by the run is the source
of truth, and this script neither recomputes nor modifies it.

    python scripts/analyze_exp0_errors.py results/<run>/<report>.json
    python scripts/analyze_exp0_errors.py <report>.json --errors
    python scripts/analyze_exp0_errors.py <report>.json --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exp0.construction_strata import (  # noqa: E402
    CORRUPTED_ARM_SURVIVING_POSITIVE,
    POSITIVE_ARM_POSITIVE,
    compare_strata,
)

CANONICAL_KEY = "canonical_validation"
CHALLENGE_KEY = "diagnostic_challenge_validation"


def load_diagnostics(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = payload.get("construction_diagnostics")
    if not diagnostics:
        raise SystemExit(
            f"{path} contains no construction diagnostics. Re-run with "
            "--construction_diagnostics to produce them."
        )
    return diagnostics


def _rate(summary: Dict[str, Any]) -> str:
    if summary["accuracy"] is None:
        return "     n/a"
    return f"{summary['accuracy']:8.4f}"


def render_strata(title: str, stratified: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print(f"  {'group':38} {'n':>6} {'errors':>7} {'accuracy':>9}")
    overall = stratified["overall"]
    print(f"  {'overall':38} {overall['count']:6d} {overall['errors']:7d} "
          f"{_rate(overall)}")
    for name, summary in stratified["construction_strata"].items():
        if summary["count"] == 0 and name.startswith(("positive_arm_neg", "unknown")):
            continue  # structurally impossible or absent; keep the table readable
        print(f"  {name:38} {summary['count']:6d} {summary['errors']:7d} "
              f"{_rate(summary)}")
        for sub, sub_summary in stratified["corruption_strata"].items():
            if not _belongs(sub, name):
                continue
            print(f"    {sub:36} {sub_summary['count']:6d} "
                  f"{sub_summary['errors']:7d} {_rate(sub_summary)}")


def _belongs(corruption_stratum: str, primary: str) -> bool:
    if primary == CORRUPTED_ARM_SURVIVING_POSITIVE:
        return corruption_stratum.startswith("corrupted_positive")
    if primary == "corrupted_arm_negative":
        return corruption_stratum.startswith("corrupted_negative")
    return False


def render_errors(errors: List[Dict[str, Any]]) -> None:
    print(f"\nerrors ({len(errors)}), ascending validation index")
    if not errors:
        print("  none")
        return
    print(f"  {'idx':>6} {'label':>6} {'pred':>6} {'arm':>10} {'corr':>5} "
          f"{'triples':>8} {'margin':>9}  witness")
    for error in errors:
        margin = error.get("prediction_margin")
        margin_text = "      n/a" if margin is None else f"{margin:9.3f}"
        print(f"  {error['index']:6d} {str(error['realized_label']):>6} "
              f"{str(error['predicted_label']):>6} "
              f"{str(error['construction_arm']):>10} "
              f"{str(error['corruption_count']):>5} "
              f"{error['num_valid_triples']:8d} {margin_text}  "
              f"{error['first_witness']}")


def render_comparisons(stratified: Dict[str, Any]) -> None:
    pairs = [
        (POSITIVE_ARM_POSITIVE, CORRUPTED_ARM_SURVIVING_POSITIVE),
        ("corrupted_positive_c1", "corrupted_positive_c2"),
        ("corrupted_negative_c1", "corrupted_negative_c2"),
    ]
    print("\nstratum comparisons")
    for left, right in pairs:
        comparison = compare_strata(stratified, left, right).to_dict()
        left_rate = comparison["left_error_rate"]
        right_rate = comparison["right_error_rate"]
        print(f"  {left} vs {right}")
        print(f"    {comparison['left_errors']}/{comparison['left_count']}"
              f" = {'n/a' if left_rate is None else f'{left_rate:.4f}'}"
              f"   vs   {comparison['right_errors']}/{comparison['right_count']}"
              f" = {'n/a' if right_rate is None else f'{right_rate:.4f}'}")
        p_value = comparison["p_value"]
        print(f"    p = {'n/a' if p_value is None else f'{p_value:.4f}'}"
              f"{'   ' + comparison['note'] if comparison['note'] else ''}")


def summarize(section: Dict[str, Any], title: str, args) -> None:
    stratified = section.get("stratified")
    if not stratified:
        return
    render_strata(title, stratified)
    if args.compare:
        render_comparisons(stratified)
    if args.errors:
        render_errors(section.get("errors", []))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="run report JSON")
    parser.add_argument("--errors", action="store_true",
                        help="list individual errors in validation-index order")
    parser.add_argument("--compare", action="store_true",
                        help="run exact tests between selected strata")
    parser.add_argument("--seed-index", type=int, default=0,
                        help="which training seed's diagnostics to summarize")
    args = parser.parse_args(argv)

    diagnostics = load_diagnostics(args.report)
    for key, title in (
        (CANONICAL_KEY, "CANONICAL validation (source-faithful distribution)"),
        (CHALLENGE_KEY, "DIAGNOSTIC CHALLENGE set (deliberately rebalanced)"),
    ):
        sections = diagnostics.get(key)
        if not sections:
            continue
        if args.seed_index >= len(sections):
            raise SystemExit(
                f"--seed-index {args.seed_index} out of range for {key} "
                f"({len(sections)} seed(s) present)"
            )
        section = sections[args.seed_index]
        summarize(section, title, args)
        if key == CHALLENGE_KEY:
            provenance = section.get("provenance", {})
            print(f"\n  challenge_id: {provenance.get('challenge_id')}")
            print(f"  acceptance:   {provenance.get('acceptance_rate_per_stratum')}")

    if CANONICAL_KEY in diagnostics and CHALLENGE_KEY in diagnostics:
        print("\nNOTE: the two sections above describe different distributions. "
              "Do not average or compare their accuracies directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
