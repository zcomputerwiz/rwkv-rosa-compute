"""Pre-registered analysis for the Experiment 0B seed study.

Implements exactly what `PREREGISTRATION_0B_SEED_STUDY.md` fixes, and nothing
else. Committed before the remaining seeds were evaluated so the analysis cannot
be tuned to the data.

Primary   one-sided exact Mann-Whitney U on per-seed corrupted_negative_near_3plus
          ROC AUC at epoch 5, alternative N=0 > N=36, alpha 0.05.
Secondary transition counts with Fisher's exact test, descriptive only. A seed
          transitioned if near_3plus AUC rose by >= 0.10 between any two
          consecutive epochs.

The study population is defined by run_id, not seed number: earlier result
families in results/ reuse the same seed numbers at different configurations,
and multi-seed batch runs share one run_id across several seeds.

Run::

    python scripts/analyze_0b_seed_study.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Frozen by pre-registration. Do not edit to accommodate a run that does not fit.
STUDY = {
    0: {42: "c968fce9af66aa32", 43: "706b5459779b201d", 44: "d6d23abcab7a898b",
        45: "0c3f9edbcb2c310f", 46: "304c24dc614f6b1a"},
    36: {42: "c923f49572cadb88", 44: "cd865b1f9c9b1089",
         45: "cf9e58a1052dc20a", 46: "e1ee93fa823e4523"},
}
STRATUM = "corrupted_negative_near_3plus"
CHALLENGE_ID = "e06f92897411fe2e"
# Section 4 pins the challenge by content, not only by id. Checking the id
# alone would accept a regenerated set that happened to reuse the id.
CONTENT_SHA256 = "bef50bba1c80600de6885bf60ef5f9fdfed1b37135715dce0b938cee6a1cb21b"
TRANSITION_DELTA = 0.10
# Pre-registration section 4: evaluation settings are part of the outcome
# definition, because batch size shifts the reported AUC by ~0.003.
PINNED_SETTINGS = {"batch_size": 128, "precision": "bf16"}
# Section 4 pins the outcome to epoch 5. Taking the highest epoch *present*
# instead would let a seed that is still training contribute its epoch-4 value
# as if it were final, and would make the completeness count report a mid-flight
# study as complete.
PINNED_EPOCH = 5
ALPHA = 0.05


def is_pinned(settings: Optional[dict]) -> bool:
    """True when both pinned keys match, whatever else the block carries."""
    return settings is not None and all(
        settings.get(k) == v for k, v in PINNED_SETTINGS.items())


def auc_from_per_instance(per_instance: Sequence[dict]) -> float:
    """Tie-corrected Mann-Whitney AUC, positives vs one negative stratum.

    Duplicated rather than imported so the pre-registered analysis does not
    change if a shared helper is refactored later.
    """
    pos = [r["margin"] for r in per_instance if r["realized_label"] is True]
    neg = [r["margin"] for r in per_instance
           if r["realized_label"] is False and r["stratum"] == STRATUM]
    if not pos or not neg:
        raise ValueError("missing positives or negatives for the stratum")
    xs = sorted([(v, 0) for v in pos] + [(v, 1) for v in neg])
    ranks: Dict[int, float] = {}
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1][0] == xs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rank_pos = sum(ranks[k] for k, (_, lab) in enumerate(xs) if lab == 0)
    return (rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def read_auc(payload: dict) -> float:
    """Prefer the emitted auc_summary; fall back to recomputing it."""
    sc = payload["structural_challenge"]
    if sc.get("challenge_id") != CHALLENGE_ID:
        raise ValueError(f"wrong challenge: {sc.get('challenge_id')}")
    if sc.get("content_sha256") != CONTENT_SHA256:
        raise ValueError(f"wrong challenge content: {sc.get('content_sha256')}")
    summary = sc.get("auc_summary") or {}
    if STRATUM in summary:
        return float(summary[STRATUM])
    return auc_from_per_instance(sc["per_instance"])


def eval_settings_of(payload: dict) -> Optional[dict]:
    """Recorded evaluation settings, from either schema in use.

    Two nodes fixed the missing-settings gap independently and landed on
    different shapes: a top-level ``evaluation_settings`` block here, and
    ``batch_size``/``precision`` inside ``canonical_validation`` and
    ``structural_challenge`` on antigravity-ampere. Both are legitimate records
    of the same fact, so read either rather than treating one as unrecorded.
    """
    top = payload.get("evaluation_settings")
    if isinstance(top, dict) and "batch_size" in top:
        return top
    for block in ("structural_challenge", "canonical_validation"):
        inner = payload.get(block)
        if isinstance(inner, dict) and "batch_size" in inner:
            return {"batch_size": inner.get("batch_size"),
                    "precision": inner.get("precision")}
    return None


def checkpoint_epoch(payload: dict) -> Optional[int]:
    """Epoch of the evaluated checkpoint, from its filename.

    The ``epochs`` field is the run's *configured* total and is 5 on every
    record including per-epoch ones, so it cannot order a series. Rolling
    ``latest.pt`` checkpoints are not epoch boundaries and are skipped.
    """
    match = re.search(r"epoch_(\d+)\.pt", str(payload.get("checkpoint", "")))
    return int(match.group(1)) if match else None


def collect(eval_dirs: Sequence[Path]) -> Tuple[Dict[int, Dict[int, float]],
                                                Dict[int, Dict[int, List[float]]],
                                                List[tuple]]:
    """Return (final-epoch AUC by arm/seed, per-epoch AUC series by arm/seed).

    Scans recursively: run_id is the filter, so where a file happens to live
    does not matter. The two pilot arms sit in different directories and the
    remote arm arrives through the shared folder.
    """
    final: Dict[int, Dict[int, float]] = {0: {}, 36: {}}
    series: Dict[int, Dict[int, List[float]]] = {0: {}, 36: {}}
    conflicts: List[tuple] = []
    unpinned: List[tuple] = []
    wanted = {run_id: (arm, seed)
              for arm, seeds in STUDY.items() for seed, run_id in seeds.items()}
    # epoch -> auc, keyed per run, so a base file duplicating _e5 collapses
    found: Dict[str, Dict[int, float]] = {}

    for eval_dir in eval_dirs:
        for path in sorted(eval_dir.rglob("*.json")):
            # Superseded artifacts are quarantined rather than deleted, so they
            # stay as evidence. They must not be read back as live results.
            if any(part.startswith("superseded") for part in path.parts):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            run_id = payload.get("run_id")
            if run_id not in wanted:
                continue
            arm, seed = wanted[run_id]
            if payload.get("seed") != seed or payload.get("num_filler") != arm:
                continue
            epoch = checkpoint_epoch(payload)
            if epoch is None:
                continue
            try:
                auc = read_auc(payload)
            except (KeyError, ValueError):
                continue
            settings = eval_settings_of(payload)
            prior = found.setdefault(run_id, {}).get(epoch)
            if prior is not None and abs(prior[0] - auc) > 1e-9:
                # Two artifacts describe the same checkpoint and disagree. Silently
                # keeping whichever the glob reached last would make the result
                # depend on filename order. Prefer the one whose settings match the
                # pre-registered pinning; if neither does, record the conflict.
                # Compare the two pinned keys explicitly: eval_settings_of may
                # return a block carrying extra keys (device, for one), and dict
                # equality would then call a genuinely pinned artifact unpinned.
                prior_ok = is_pinned(prior[1])
                this_ok = is_pinned(settings)
                if this_ok and not prior_ok:
                    pass                       # this one wins
                elif prior_ok and not this_ok:
                    continue                   # keep the prior
                else:
                    conflicts.append((run_id, epoch, prior[0], auc, prior_ok))
                    continue
            found[run_id][epoch] = (auc, settings)

    for run_id, by_epoch in found.items():
        arm, seed = wanted[run_id]
        series[arm][seed] = [by_epoch[e][0] for e in sorted(by_epoch)]
        if PINNED_EPOCH not in by_epoch:
            continue
        auc, settings = by_epoch[PINNED_EPOCH]
        # Section 4 makes the pinned settings part of the outcome definition,
        # so an unpinned artifact is not a valid outcome even when it is the
        # only one present. Preferring pinned artifacts on disagreement is a
        # tie-break; this is the requirement the tie-break was serving.
        if not is_pinned(settings):
            unpinned.append((arm, seed, settings))
            continue
        final[arm][seed] = auc
    if unpinned:
        print("OUTCOME ARTIFACT NOT AT THE PINNED SETTINGS - excluded:")
        for arm, seed, settings in unpinned:
            print(f"  N={arm} seed {seed} epoch {PINNED_EPOCH}: {settings}")
        print("  Re-evaluate at batch_size=128, precision=bf16." + chr(10))
    if conflicts:
        print("CONFLICTING EVALUATIONS - same checkpoint, different values:")
        for run_id, epoch, a, b, both_pinned in conflicts:
            arm, seed = wanted[run_id]
            why = ("both carry the pinned settings and still disagree"
                   if both_pinned else "neither carries the pinned settings")
            print(f"  N={arm} seed {seed} epoch {epoch}: {a:.6f} vs {b:.6f} "
                  f"(delta {abs(a-b):.6f}) - {why}")
        print("  Re-evaluate at the pinned settings; the result below is not "
              "well defined until this is resolved." + chr(10))
    return final, series, conflicts


def mann_whitney_one_sided(x: Sequence[float], y: Sequence[float]) -> float:
    """P(rank-sum of x >= observed) by exact enumeration. Alternative: x > y.

    Ranks are tie-corrected, as section 4 requires. The previous implementation
    keyed ranks by value (``{v: i + 1}``), which collapsed duplicate margins
    onto one arbitrary position while the null enumeration summed *positional*
    indices. The two agree only when no ties exist; all-tied inputs returned
    p = 0, which is not merely wrong but the opposite tail.
    """
    pooled = sorted(list(x) + list(y))
    n = len(pooled)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    rank_of = {}
    for k, value in enumerate(pooled):
        rank_of.setdefault(value, ranks[k])
    observed = sum(rank_of[v] for v in x)
    total = hits = 0
    for combo in combinations(range(n), len(x)):
        total += 1
        if sum(ranks[i] for i in combo) >= observed - 1e-12:
            hits += 1
    return hits / total


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c

    def pr(x: int) -> float:
        return (math.comb(r1, x) * math.comb(r2, c1 - x)) / math.comb(n, c1)

    observed = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= observed + 1e-12)


def transitioned(series: Sequence[float]) -> Optional[bool]:
    """None when there are too few epochs to judge."""
    if len(series) < 2:
        return None
    return any(series[i + 1] - series[i] >= TRANSITION_DELTA
               for i in range(len(series) - 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, nargs="+",
                    default=[Path("results"), Path("D:/ProjectSync/exp0_0b_seed_study/inbox")],
                    help="directories scanned recursively; run_id is the filter")
    args = ap.parse_args()

    final, series, conflicts = collect([d for d in args.eval_dir if d.exists()])

    print(f"per-seed near_3plus AUC at epoch {PINNED_EPOCH}\n")
    print(f"  {'arm':>5} {'seed':>5} {'run_id':>18} {'AUC':>8} {'epochs':>7} {'transition':>11}")
    for arm in (0, 36):
        for seed, run_id in STUDY[arm].items():
            seen = len(series[arm].get(seed, ()))
            if seed in final[arm]:
                tr = transitioned(series[arm][seed])
                tr_s = "-" if tr is None else ("yes" if tr else "no")
                print(f"  {arm:5d} {seed:5d} {run_id:>18} {final[arm][seed]:8.4f} "
                      f"{seen:7d} {tr_s:>11}")
            else:
                # PARTIAL and MISSING are different states: one is a run in
                # flight, the other is a run with no artifacts at all.
                state = "PARTIAL" if seen else "MISSING"
                print(f"  {arm:5d} {seed:5d} {run_id:>18} {state:>8} "
                      f"{seen if seen else '-':>7} {'-':>11}")

    x, y = list(final[0].values()), list(final[36].values())
    n_expected = (len(STUDY[0]), len(STUDY[36]))
    print(f"\nobserved {len(x)} of {n_expected[0]} (N=0), "
          f"{len(y)} of {n_expected[1]} (N=36)")
    complete = len(x) == n_expected[0] and len(y) == n_expected[1]
    if not x or not y:
        print("\nnothing to test yet")
        return 0

    if not complete or conflicts:
        # Section 8 promises the analyzer refuses to present a result while any
        # run is missing. Printing the p-value under a warning is not refusing:
        # the number is still quotable, and every interim run of this tool
        # reaches exactly this branch.
        #
        # An unresolved conflict is the same defect wearing different clothes.
        # The conflict block above already says the result is not well defined;
        # printing a p-value underneath it said otherwise, and the printed value
        # would depend on which artifact the sorted scan reached first.
        reason = ("the pre-registered test is defined on the full set"
                  if not complete else
                  "conflicting evaluations of the same checkpoint are unresolved")
        print(f"\nPRIMARY WITHHELD - {reason}.\n  Per-seed values above are "
              "diagnostic. No p-value is computed.")
    else:
        p = mann_whitney_one_sided(x, y)
        print("\nPRIMARY  one-sided exact Mann-Whitney U, alternative N=0 > N=36")
        print(f"  N=0  n={len(x)} median {statistics.median(x):.4f}")
        print(f"  N=36 n={len(y)} median {statistics.median(y):.4f}")
        print(f"  p = {p:.4f}   {'REJECT' if p <= ALPHA else 'no rejection'} at alpha={ALPHA}")
        print(f"  minimum achievable p at this design: {1/math.comb(len(x)+len(y), len(x)):.4f}")

    t0 = [transitioned(series[0][s]) for s in final[0]]
    t36 = [transitioned(series[36][s]) for s in final[36]]
    a, b = sum(1 for t in t0 if t), sum(1 for t in t0 if t is False)
    c, d = sum(1 for t in t36 if t), sum(1 for t in t36 if t is False)
    print(f"\nSECONDARY (descriptive)  transitions: N=0 {a}/{a+b}, N=36 {c}/{c+d}")
    if a + b and c + d:
        print(f"  Fisher two-sided p = {fisher_two_sided(a, b, c, d):.4f}  (not a significance claim)")
    unjudged = sum(1 for t in t0 + t36 if t is None)
    if unjudged:
        print(f"  {unjudged} seed(s) lacked per-epoch evaluations and are excluded here")
    return 0


def _self_check() -> None:
    # exact values verified independently against the power analysis
    assert abs(fisher_two_sided(5, 0, 0, 5) - 0.0079) < 5e-5
    assert abs(fisher_two_sided(4, 1, 1, 4) - 0.2063) < 5e-5
    assert abs(mann_whitney_one_sided([3, 4, 5], [0, 1, 2]) - 1 / 20) < 1e-12
    assert abs(mann_whitney_one_sided([0, 1, 2], [3, 4, 5]) - 1.0) < 1e-12
    # Ties in the Mann-Whitney path. The earlier asserts were all tie-free, so
    # they exercised only the inputs the broken implementation got right - the
    # all-tied case returned p = 0, the impossible value.
    assert abs(mann_whitney_one_sided([0.5, 0.5], [0.5]) - 1.0) < 1e-12
    assert abs(mann_whitney_one_sided([0.5, 0.5], [0.5, 0.5]) - 1.0) < 1e-12
    assert abs(mann_whitney_one_sided([0.7, 0.5], [0.5, 0.3]) - 1 / 3) < 1e-12
    # a pinned block is still pinned when it carries extra keys
    assert is_pinned({"batch_size": 128, "precision": "bf16", "device": "cuda"})
    assert not is_pinned({"batch_size": 64, "precision": "bf16"})
    assert not is_pinned(None)
    # The challenge is pinned by content, not only by id: a regenerated set that
    # reused the id must be rejected rather than silently accepted.
    good = {"challenge_id": CHALLENGE_ID, "content_sha256": CONTENT_SHA256,
            "auc_summary": {STRATUM: 0.5}}
    assert read_auc({"structural_challenge": good}) == 0.5
    for bad in ({**good, "content_sha256": "0" * 64},
                {**good, "challenge_id": "deadbeef"}):
        try:
            read_auc({"structural_challenge": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted a mismatched challenge: {bad}")
    # An even-sized arm has a real median, not its upper-middle observation.
    assert abs(statistics.median([0.5474, 0.5585, 0.5588, 0.5671]) - 0.55865) < 1e-9
    # ties are corrected, not broken arbitrarily
    assert abs(auc_from_per_instance([
        {"margin": 1.0, "realized_label": True, "stratum": "positive_arm_positive"},
        {"margin": 1.0, "realized_label": False, "stratum": STRATUM},
    ]) - 0.5) < 1e-12
    assert transitioned([0.52, 0.71]) is True
    assert transitioned([0.566, 0.5661]) is False
    assert transitioned([0.6]) is None
    print("self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
