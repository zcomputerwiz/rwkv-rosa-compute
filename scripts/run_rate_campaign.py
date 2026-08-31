#!/usr/bin/env python3
"""Driver for the pre-registered p_ref rate campaign.

The campaign estimates a success *rate* rather than a central tendency, because
the outcome at this cell is bimodal: repeated measurement showed five of nine
seed-node combinations reaching the upper mode and nothing landing between the
modes. A median over three seeds is a lossy decision rule that estimates
neither the rate nor its uncertainty.

Three rules here are pre-registered and exist to stop the campaign from
selecting on noise. They are not conveniences and should not be relaxed:

**Rotating condition order.** Conditions cycle within each seed rather than
running all of one condition and then the next, so condition is not confounded
with time of day or accumulated machine state.

**First valid run wins.** The first complete, provenance-valid artifact for a
(seed, condition) is the primary observation. An existing output is never
overwritten, so a rerun cannot quietly replace a result that was already
recorded. Accuracy is compared against the threshold at full stored precision,
never at display precision.

**Frozen tree.** Every run in a panel must come from one clean commit. A dirty
tree aborts the campaign rather than producing artifacts that cannot be
compared with each other.

Usage::

    python scripts/run_rate_campaign.py --node ada --panel discovery
    python scripts/run_rate_campaign.py --node ada --panel discovery --dry-run
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Shared by every condition. Anything not listed here is an audit-tool default
# and is recorded in each artifact's config block.
COMMON = [
    "--lrs", "1e-3", "--d-models", "128", "--layers", "2",
    "--val-memories", "448", "--batch-size", "256",
    "--compile", "--compile-backend", "cudagraphs",
]

# Complete protocols, not isolated levers. C1 and C2 are step-matched to each
# other at 39,936 and both are four times B0; they differ in epochs, exposures
# per memory, cosine-update granularity and evaluation cadence as well as bank
# size, so a difference between them selects a protocol and does not identify a
# mechanism.
CONDITIONS = {
    "B0": {"memories": 19968, "epochs": 32},    # 9,984 steps
    "C1": {"memories": 19968, "epochs": 128},   # 39,936 steps
    "C2": {"memories": 79872, "epochs": 32},    # 39,936 steps
}

PANELS = {
    "discovery": list(range(2001, 2011)),      # paired across all conditions
    "confirmatory": list(range(2101, 2121)),   # frozen winner only
}

THRESHOLD = 0.95


def rotation(index):
    """Condition order for the index-th seed: B0/C1/C2, C1/C2/B0, C2/B0/C1."""
    order = ["B0", "C1", "C2"]
    shift = index % 3
    return order[shift:] + order[:shift]


def frozen_tree():
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    if git("status", "--porcelain"):
        raise SystemExit(
            "refusing to run: the working tree is dirty. Every run in a panel "
            "must come from one clean commit, or the artifacts cannot be "
            "compared with each other.")
    return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node", required=True,
                   help="node tag used in artifact names, e.g. ada or turing")
    p.add_argument("--panel", required=True, choices=sorted(PANELS))
    p.add_argument("--condition", choices=sorted(CONDITIONS),
                   help="confirmatory panel only: the single frozen winner")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "rate_campaign")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.panel == "confirmatory" and not args.condition:
        p.error("the confirmatory panel runs one frozen winner; pass --condition")

    commit, tree = frozen_tree()
    seeds = PANELS[args.panel]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, seed in enumerate(seeds):
        conds = [args.condition] if args.condition else rotation(i)
        for cond in conds:
            name = f"{args.node}_{args.panel}_{cond}_s{seed}.json"
            jobs.append((seed, cond, args.out_dir / name))

    print(f"panel      {args.panel}   node {args.node}")
    print(f"commit     {commit}")
    print(f"tree       {tree}")
    print(f"seeds      {seeds[0]}-{seeds[-1]} ({len(seeds)})")
    print(f"jobs       {len(jobs)}")
    done = sum(1 for _, _, o in jobs if o.exists())
    if done:
        print(f"already present, will be left untouched: {done}")
    print()

    started = time.time()
    for n, (seed, cond, out) in enumerate(jobs, 1):
        if out.exists():
            print(f"[{n}/{len(jobs)}] skip  {out.name}  (first valid run stands)")
            continue
        cfg = CONDITIONS[cond]
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "audit_recall_capacity.py"),
               *COMMON,
               "--memories", str(cfg["memories"]),
               "--epochs", str(cfg["epochs"]),
               "--model-seed", str(seed),
               "--out", str(out),
               "--label", f"{args.node}-{args.panel}-{cond}-s{seed}"]
        print(f"[{n}/{len(jobs)}] {cond} seed {seed} -> {out.name}")
        if args.dry_run:
            print("        " + " ".join(cmd[1:]))
            continue
        r = subprocess.run(cmd, cwd=REPO_ROOT)
        if r.returncode != 0:
            # Retained deliberately: a failed run is part of the record, and
            # the validity rule requires the reason to be reportable.
            print(f"        FAILED with exit {r.returncode}; leaving no artifact")
            continue
        acc = json.loads(out.read_text())["results"][0]["held_out_final"]
        print(f"        held_out_final {acc!r}  "
              f"{'PASS' if acc >= THRESHOLD else 'no'}  "
              f"({(time.time() - started) / 60:.0f} min elapsed)")

    print(f"\npanel complete in {(time.time() - started) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
