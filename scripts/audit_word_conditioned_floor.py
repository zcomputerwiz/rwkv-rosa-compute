import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

# Matches every other script in scripts/. The `src.` package form works under
# pytest, which puts the repository root on the path, and fails when the script
# is run directly -- which is the only way it is actually used.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exp1.pointer_chase import execute, generate_memory  # noqa: E402


def estimate_floor(num_nodes: int, num_maps: int, words: int, memories: int, bootstrap: int, seed: int, depth: int = None, word_list: list = None):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if word_list is None:
        if depth is None:
            raise ValueError("Must provide either depth or word_list")
        word_list = []
        for _ in range(words):
            word_list.append(tuple(rng.randrange(num_maps) for _ in range(depth)))
    else:
        words = len(word_list)

    A_scores = np.zeros((words, memories))
    B_scores = np.zeros((words, memories))

    for w_idx, word in enumerate(word_list):
        for m_idx in range(memories):
            mem_A = generate_memory(rng, num_nodes=num_nodes, num_maps=num_maps, permutations=True)
            fixed_points_A = sum(execute(mem_A, start, word) == start for start in range(num_nodes))
            A_scores[w_idx, m_idx] = fixed_points_A / num_nodes

            mem_B = generate_memory(rng, num_nodes=num_nodes, num_maps=num_maps, permutations=True)
            fixed_points_B = sum(execute(mem_B, start, word) == start for start in range(num_nodes))
            B_scores[w_idx, m_idx] = fixed_points_B / num_nodes

    p_A = A_scores.mean(axis=1)
    p_B = B_scores.mean(axis=1)

    predict_start = p_A > (1 - p_A) / (num_nodes - 1)
    scores = np.where(predict_start, p_B, (1 - p_B) / (num_nodes - 1))
    E_Bw = scores.mean()

    if bootstrap == 0:
        return float(E_Bw), 0.0

    bootstrap_excesses = []
    for _ in range(bootstrap):
        w_indices = np_rng.choice(words, size=words, replace=True)
        m_indices_A = np_rng.choice(memories, size=(words, memories), replace=True)
        m_indices_B = np_rng.choice(memories, size=(words, memories), replace=True)

        sample_A = A_scores[w_indices[:, None], m_indices_A]
        sample_B = B_scores[w_indices[:, None], m_indices_B]

        p_A_boot = sample_A.mean(axis=1)
        p_B_boot = sample_B.mean(axis=1)

        predict_start_boot = p_A_boot > (1 - p_A_boot) / (num_nodes - 1)
        score_boot = np.where(predict_start_boot, p_B_boot, (1 - p_B_boot) / (num_nodes - 1))

        bootstrap_excesses.append(score_boot.mean() - (1 / num_nodes))

    bound_95 = np.percentile(bootstrap_excesses, 95)

    return float(E_Bw), float(bound_95)

def main():
    parser = argparse.ArgumentParser(description="Estimate word-conditioned shortcut floor for pointer chase.")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 12, 16, 24, 32],
                        help="Depths to evaluate.")
    parser.add_argument("--num-nodes", type=int, default=16, help="Number of nodes (M).")
    parser.add_argument("--num-maps", type=int, default=4, help="Number of maps (K).")
    parser.add_argument("--words", type=int, default=200, help="Number of selector words (publication-grade: 1000+).")
    parser.add_argument("--memories", type=int, default=200, help="Number of memories per word (publication-grade: 1000+).")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=1004, help="Random seed.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file.")

    args = parser.parse_args()

    results = []

    print(f"{'Depth':>6} | {'E_w[B_w]':>10} | {'Excess':>10} | {'95% UB':>10}")
    print("-" * 45)

    for depth in args.depths:
        # Vary seed deterministically by depth so we don't repeat the exact same RNG sequence
        # for generation, while remaining fully determined by the single `--seed`.
        depth_seed = args.seed + depth
        e_bw, bound_95 = estimate_floor(
            num_nodes=args.num_nodes,
            num_maps=args.num_maps,
            words=args.words,
            memories=args.memories,
            bootstrap=args.bootstrap,
            seed=depth_seed,
            depth=depth
        )

        excess = e_bw - (1.0 / args.num_nodes)

        print(f"{depth:6} | {e_bw:10.5f} | {excess:10.5f} | {bound_95:10.5f}")

        results.append({
            "depth": depth,
            "E_Bw": e_bw,
            "excess": excess,
            "upper_bound_95": bound_95
        })

    artifact = {
        "parameters": {
            "depths": args.depths,
            "num_nodes": args.num_nodes,
            "num_maps": args.num_maps,
            "words": args.words,
            "memories": args.memories,
            "bootstrap": args.bootstrap,
            "seed": args.seed
        },
        "results": results
    }

    out_json = json.dumps(artifact, indent=2)
    dir_name = os.path.dirname(os.path.abspath(args.out))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=dir_name, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(out_json)
        os.replace(temp_path, args.out)
    except Exception:
        os.remove(temp_path)
        raise

if __name__ == "__main__":
    main()
