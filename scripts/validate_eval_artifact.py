#!/usr/bin/env python3
"""Validate that an evaluation artifact carries required provenance fields."""

import argparse
import json
import os
import sys
from pathlib import Path

ALIASES = {
    # Expected fields can be mapped to a list of possible json paths (dot separated).
    # e.g., 'a.b.c' will look for {"a": {"b": {"c": value}}}

    # identity
    "run_id": ["run_id"],
    "seed": ["seed", "seeds_run", "canonical_validation.eval_seed", "evaluation.seeds_run"],
    "epoch": ["epochs", "training_protocol.epochs", "eval_epoch"],

    # inputs
    "challenge_or_dataset": ["structural_challenge.challenge_id", "task", "task_config", "challenge", "dataset"],
    "content_hash": ["structural_challenge.content_sha256", "input_sha256"],

    # code
    "commit": ["commit"],
    "script_hash": ["script_sha256"],

    # checkpoint
    "checkpoint": ["checkpoint", "model.rwkv_checkpoint_sha256"],

    # settings
    "batch_size": ["evaluation_settings.batch_size", "settings.batch_size", "training_protocol.batch_size"],
    "precision": ["evaluation_settings.precision", "settings.precision", "precision"],

    # device
    "device_name": ["environment.gpu_name", "environment.gpu"],
    "compute_capability": ["environment.gpu_compute_capability", "environment.capability"]
}

def resolve_alias(data, paths, enforce_hash=False, hash_len=64):
    for path in paths:
        parts = path.split('.')
        curr = data
        found = True
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                found = False
                break

        # Must be present and not explicitly an empty collection/string
        if found and curr is not None and curr != "" and curr != [] and curr != {}:
            if enforce_hash:
                if isinstance(curr, str) and len(curr) == hash_len and all(c in "0123456789abcdefABCDEF" for c in curr):
                    return True
                # Not a valid hash, check next alias
                continue
            return True
    return False

def check_identity_is_eval(data):
    return (resolve_alias(data, ALIASES["run_id"]) or
            resolve_alias(data, ALIASES["seed"]) or
            resolve_alias(data, ALIASES["epoch"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSON files or directories to check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any required field is missing",
    )
    args = parser.parse_args(argv)

    json_files = []
    for p in args.paths:
        if p.is_file():
            if p.suffix == ".json":
                json_files.append(p)
        elif p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".json"):
                        json_files.append(Path(root) / f)
        else:
            print(f"Warning: {p} not found or not a valid path", file=sys.stderr)

    if not json_files:
        print("No JSON files found to validate.", file=sys.stderr)
        return 1 if args.strict else 0

    results = {}
    any_missing = False

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            results[str(fpath)] = {"error": f"Failed to parse JSON: {e}"}
            any_missing = True
            continue

        if not check_identity_is_eval(data):
            results[str(fpath)] = {"error": "not an eval artifact"}
            continue

        missing = []

        # Identity group
        if not resolve_alias(data, ALIASES["run_id"]):
            missing.append("run id")
        if not resolve_alias(data, ALIASES["seed"]):
            missing.append("seed")
        if not resolve_alias(data, ALIASES["epoch"]):
            missing.append("the evaluated epoch")

        # Inputs group
        if not (resolve_alias(data, ALIASES["challenge_or_dataset"]) and resolve_alias(data, ALIASES["content_hash"], enforce_hash=True)):
            missing.append("challenge or dataset identifier AND its content hash")

        # Code group
        if not resolve_alias(data, ALIASES["commit"], enforce_hash=True, hash_len=40):
            missing.append("commit")
        if not resolve_alias(data, ALIASES["script_hash"]):
            missing.append("script hash")

        # Checkpoint group
        if not resolve_alias(data, ALIASES["checkpoint"]):
            missing.append("checkpoint identifier or hash")

        # Settings group
        if not (resolve_alias(data, ALIASES["batch_size"]) and resolve_alias(data, ALIASES["precision"])):
            missing.append("evaluation batch size and precision")

        # Device group
        if not (resolve_alias(data, ALIASES["device_name"]) and resolve_alias(data, ALIASES["compute_capability"])):
            missing.append("device name and compute capability")

        if missing:
            any_missing = True

        results[str(fpath)] = {"missing": missing}

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for fpath, res in results.items():
            if "error" in res:
                print(f"{fpath}: {res['error']}")
            elif res["missing"]:
                print(f"{fpath}: missing {', '.join(res['missing'])}")
            else:
                pass # Silent on pass unless requested, or maybe print ok? We will just print if there are missing

    if args.strict and any_missing:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
