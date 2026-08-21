#!/usr/bin/env python3
"""Generate and freeze the 6-stratum structural challenge set for Experiment 0A follow-up."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exp0.structural_challenge import (
    StructuralChallengeSpec,
    generate_structural_challenge_set,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821, help="Fixed structural challenge seed")
    parser.add_argument("--per_stratum", type=int, default=1000, help="Examples per stratum")
    parser.add_argument("--out", type=Path, default=Path("results/exp0_structural/structural_challenge_20260821.json"))
    args = parser.parse_args(argv)

    spec = StructuralChallengeSpec(
        seed=args.seed,
        per_stratum=args.per_stratum,
    )
    print(f"Generating structural challenge set with seed={spec.seed}, per_stratum={spec.per_stratum}...")
    dataset = generate_structural_challenge_set(spec)

    prov = dataset["provenance"]
    print(f"Challenge ID   : {prov['challenge_id']}")
    print(f"Content SHA256 : {prov['content_sha256']}")
    print(f"Total instances: {prov['total_instances']}")
    print("Stratum breakdown & acceptance rates:")
    for stratum, count in prov["realized_strata"].items():
        attempts = prov["attempts_per_stratum"][stratum]
        rate = prov["acceptance_rate_per_stratum"][stratum]
        rate_str = f"{rate:6.2%}" if rate is not None else "   n/a"
        print(f"  {stratum:36}: {count:5d} / {attempts:7d} attempts ({rate_str})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(f"\nFrozen structural challenge set saved to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
