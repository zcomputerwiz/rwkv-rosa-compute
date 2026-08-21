#!/usr/bin/env python3
"""Statistical analysis of paired N=0 vs N=36 structural challenge evaluations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import binomtest


def mcnemar_exact_test(b: int, c: int) -> Tuple[float, Dict[str, Any]]:
    """Exact two-sided binomial test on discordant pairs (b = N0 wrong/N36 correct, c = N0 correct/N36 wrong)."""
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0, {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}
    res = binomtest(b, n_discordant, 0.5, alternative="two-sided")
    return res.pvalue, {
        "b_n0wrong_n36correct": b,
        "c_n0correct_n36wrong": c,
        "n_discordant": n_discordant,
        "p_value": res.pvalue,
    }


def analyze_structural_experiment(
    n0_eval_files: Dict[int, Path],
    n36_eval_files: Dict[int, Path],
) -> Dict[str, Any]:
    """Perform primary contrasts, paired McNemar tests, and error tracking."""
    seeds = sorted(set(n0_eval_files.keys()) & set(n36_eval_files.keys()))
    if not seeds:
        raise ValueError("No common seeds between N=0 and N=36 evaluation files.")

    strata_names = [
        "positive_arm_positive",
        "corrupted_arm_surviving_positive",
        "corrupted_negative_near_0",
        "corrupted_negative_near_1",
        "corrupted_negative_near_2",
        "corrupted_negative_near_3plus",
    ]

    per_seed_results = {}
    n0_all_instances = {}
    n36_all_instances = {}

    for seed in seeds:
        n0_data = json.loads(n0_eval_files[seed].read_text(encoding="utf-8"))
        n36_data = json.loads(n36_eval_files[seed].read_text(encoding="utf-8"))

        n0_chal = n0_data["structural_challenge"]
        n36_chal = n36_data["structural_challenge"]
        n0_canon = n0_data["canonical_validation"]
        n36_canon = n36_data["canonical_validation"]

        n0_inst = n0_chal["per_instance"]
        n36_inst = n36_chal["per_instance"]
        n0_all_instances[seed] = n0_inst
        n36_all_instances[seed] = n36_inst

        strata_table = {}
        paired_mcnemar = {}

        for s in strata_names:
            n0_acc = n0_chal["strata_summary"][s]["accuracy"]
            n36_acc = n36_chal["strata_summary"][s]["accuracy"]
            delta = n36_acc - n0_acc

            # Paired discordant pairs
            b = 0
            c = 0
            for i in range(len(n0_inst)):
                if n0_inst[i]["stratum"] == s:
                    n0_c = n0_inst[i]["is_correct"]
                    n36_c = n36_inst[i]["is_correct"]
                    if (not n0_c) and n36_c:
                        b += 1
                    elif n0_c and (not n36_c):
                        c += 1

            pval, mcnemar_info = mcnemar_exact_test(b, c)

            strata_table[s] = {
                "n0_acc": n0_acc,
                "n36_acc": n36_acc,
                "delta": delta,
                "mcnemar": mcnemar_info,
            }

        # Mechanistic contrasts
        easy_neg_gain = strata_table["corrupted_negative_near_0"]["delta"]
        hard_neg_gain = strata_table["corrupted_negative_near_3plus"]["delta"]
        interaction = hard_neg_gain - easy_neg_gain

        per_seed_results[seed] = {
            "canonical_accuracy": {
                "n0": n0_canon["accuracy"],
                "n36": n36_canon["accuracy"],
                "delta": n36_canon["accuracy"] - n0_canon["accuracy"],
            },
            "challenge_overall_accuracy": {
                "n0": n0_chal["overall_accuracy"],
                "n36": n36_chal["overall_accuracy"],
                "delta": n36_chal["overall_accuracy"] - n0_chal["overall_accuracy"],
            },
            "strata": strata_table,
            "mechanistic_contrasts": {
                "easy_negative_gain": easy_neg_gain,
                "hard_negative_gain": hard_neg_gain,
                "structural_interaction": interaction,
            },
        }

    # Cross-seed persistent error identification
    # Find instances where all 3 N0 models miss, but all 3 N36 models solve
    num_instances = len(n0_all_instances[seeds[0]])
    n0_miss_all_n36_solve_all = []
    n0_solve_all_n36_miss_all = []

    for i in range(num_instances):
        n0_correct_all = all(n0_all_instances[s][i]["is_correct"] for s in seeds)
        n0_miss_all = all(not n0_all_instances[s][i]["is_correct"] for s in seeds)
        n36_correct_all = all(n36_all_instances[s][i]["is_correct"] for s in seeds)
        n36_miss_all = all(not n36_all_instances[s][i]["is_correct"] for s in seeds)

        rec = n0_all_instances[seeds[0]][i]
        if n0_miss_all and n36_correct_all:
            n0_miss_all_n36_solve_all.append({
                "index": i,
                "stratum": rec["stratum"],
                "realized_label": rec["realized_label"],
                "near_match_2of3_count": rec.get("near_match_2of3_count"),
                "n0_margins": [n0_all_instances[s][i]["margin"] for s in seeds],
                "n36_margins": [n36_all_instances[s][i]["margin"] for s in seeds],
            })
        elif n0_correct_all and n36_miss_all:
            n0_solve_all_n36_miss_all.append({
                "index": i,
                "stratum": rec["stratum"],
                "realized_label": rec["realized_label"],
                "near_match_2of3_count": rec.get("near_match_2of3_count"),
                "n0_margins": [n0_all_instances[s][i]["margin"] for s in seeds],
                "n36_margins": [n36_all_instances[s][i]["margin"] for s in seeds],
            })

    # Aggregate means across seeds
    mean_easy_gain = sum(per_seed_results[s]["mechanistic_contrasts"]["easy_negative_gain"] for s in seeds) / len(seeds)
    mean_hard_gain = sum(per_seed_results[s]["mechanistic_contrasts"]["hard_negative_gain"] for s in seeds) / len(seeds)
    mean_interaction = sum(per_seed_results[s]["mechanistic_contrasts"]["structural_interaction"] for s in seeds) / len(seeds)

    all_positive_interaction = all(per_seed_results[s]["mechanistic_contrasts"]["structural_interaction"] > 0 for s in seeds)

    if all_positive_interaction and mean_interaction > 0.02:
        conclusion = "supports preferential hard-structure benefit"
    elif mean_interaction <= 0:
        conclusion = "does not support a filler benefit on this structural axis"
    else:
        conclusion = "supports only a generic filler benefit or mixed/insufficient"

    return {
        "seeds": seeds,
        "per_seed": per_seed_results,
        "aggregate": {
            "mean_easy_negative_gain": mean_easy_gain,
            "mean_hard_negative_gain": mean_hard_gain,
            "mean_structural_interaction": mean_interaction,
            "all_positive_interaction": all_positive_interaction,
            "conclusion": conclusion,
        },
        "persistent_differences": {
            "n0_miss_all_n36_solve_all_count": len(n0_miss_all_n36_solve_all),
            "n0_solve_all_n36_miss_all_count": len(n0_solve_all_n36_miss_all),
            "n0_miss_all_n36_solve_all_examples": n0_miss_all_n36_solve_all,
            "n0_solve_all_n36_miss_all_examples": n0_solve_all_n36_miss_all,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n0_evals", nargs="+", type=Path, required=True, help="List of N=0 evaluation JSON files")
    parser.add_argument("--n36_evals", nargs="+", type=Path, required=True, help="List of N=36 evaluation JSON files")
    parser.add_argument("--out", type=Path, required=True, help="Output analysis JSON")
    args = parser.parse_args(argv)

    def extract_seed(p: Path) -> int:
        data = json.loads(p.read_text(encoding="utf-8"))
        return int(data["seed"])

    n0_map = {extract_seed(p): p for p in args.n0_evals}
    n36_map = {extract_seed(p): p for p in args.n36_evals}

    analysis = analyze_structural_experiment(n0_map, n36_map)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("EXPERIMENT 0A FOLLOW-UP: STRUCTURAL HARDNESS ANALYSIS")
    print("=" * 78)
    print(f"Seeds evaluated: {analysis['seeds']}")
    print(f"Scientific conclusion: {analysis['aggregate']['conclusion'].upper()}\n")

    for seed, data in analysis["per_seed"].items():
        print(f"--- SEED {seed} ---")
        print(f"  Canonical Acc: N0={data['canonical_accuracy']['n0']:6.2%}, N36={data['canonical_accuracy']['n36']:6.2%}, Δ={data['canonical_accuracy']['delta']:+6.2%}")
        print(f"  Challenge Acc: N0={data['challenge_overall_accuracy']['n0']:6.2%}, N36={data['challenge_overall_accuracy']['n36']:6.2%}, Δ={data['challenge_overall_accuracy']['delta']:+6.2%}")
        print("  Strata Accuracy Table:")
        print(f"    {'Stratum':36} | {'N0 Acc':>8} | {'N36 Acc':>8} | {'Delta':>8} | {'McNemar (b/c)':>13} | {'p-value':>9}")
        print("    " + "-" * 90)
        for s, sdata in data["strata"].items():
            m = sdata["mcnemar"]
            m_str = f"{m['b_n0wrong_n36correct']}/{m['c_n0correct_n36wrong']}"
            p_str = f"{m['p_value']:9.3e}" if m['p_value'] is not None else "     n/a"
            print(f"    {s:36} | {sdata['n0_acc']:8.2%} | {sdata['n36_acc']:8.2%} | {sdata['delta']:+8.2%} | {m_str:>13} | {p_str:>9}")

        mc = data["mechanistic_contrasts"]
        print("  Mechanistic Contrasts:")
        print(f"    Easy Negative Gain (near=0)   : {mc['easy_negative_gain']:+6.2%}")
        print(f"    Hard Negative Gain (near>=3)  : {mc['hard_negative_gain']:+6.2%}")
        print(f"    Structural Interaction (Hard - Easy): {mc['structural_interaction']:+6.2%}\n")

    agg = analysis["aggregate"]
    print("--- AGGREGATE SUMMARY ---")
    print(f"  Mean Easy Negative Gain : {agg['mean_easy_negative_gain']:+6.2%}")
    print(f"  Mean Hard Negative Gain : {agg['mean_hard_negative_gain']:+6.2%}")
    print(f"  Mean Structural Interaction: {agg['mean_structural_interaction']:+6.2%}")
    print(f"  Consistent Direction across Seeds: {agg['all_positive_interaction']}")
    print(f"  N0-miss-all -> N36-solve-all persistent instances: {analysis['persistent_differences']['n0_miss_all_n36_solve_all_count']}")
    print(f"  N0-solve-all -> N36-miss-all persistent instances: {analysis['persistent_differences']['n0_solve_all_n36_miss_all_count']}")
    print(f"\nAnalysis saved to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
