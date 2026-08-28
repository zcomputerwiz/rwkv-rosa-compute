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
    "commit": ["provenance.repository_commit", "commit"],

    # new script provenance
    "producer_script_path": ["provenance.producer_script_path"],
    "producer_script_hash": ["provenance.producer_script_git_blob_sha256"],
    "producer_script_hash_basis": ["provenance.producer_script_hash_basis"],

    # legacy script hash
    "legacy_script_hash": ["script_sha256"],

    # checkpoint
    "checkpoint_id": ["checkpoint", "model.checkpoint", "model.rwkv_checkpoint", "initialization.checkpoint_path"],
    "checkpoint_hash": ["model.rwkv_checkpoint_sha256", "initialization.checkpoint_sha256", "checkpoint_sha256"],

    # settings
    "batch_size": ["evaluation_settings.batch_size", "settings.batch_size", "training_protocol.batch_size"],
    "precision": ["evaluation_settings.precision", "settings.precision", "precision"],

    # device and environment (with new provenance.device.* fields)
    "device_name": ["provenance.device.gpu_name", "environment.gpu_name", "environment.gpu"],
    "compute_capability": ["provenance.device.gpu_compute_capability", "environment.gpu_compute_capability", "environment.capability"],
    "env_python": ["provenance.device.python_version", "environment.python_version", "environment.python"],
    "env_torch": ["provenance.device.torch_version", "environment.torch_version", "environment.torch"],
    "env_cuda": ["provenance.device.cuda_version", "environment.cuda_version", "environment.cuda"]
}

def get_alias_value(data, paths):
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

        if found and curr is not None and curr != "" and curr != [] and curr != {}:
            return curr
    return None

def resolve_alias(data, paths, enforce_hash=False, hash_len=64, lowercase_only=False):
    val = get_alias_value(data, paths)
    if val is not None:
        if enforce_hash:
            if isinstance(val, str) and len(val) == hash_len:
                valid_chars = "0123456789abcdef" if lowercase_only else "0123456789abcdefABCDEF"
                if all(c in valid_chars for c in val):
                    return True
            return False
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

        script_missing = False
        legacy_script = get_alias_value(data, ALIASES["legacy_script_hash"])
        if legacy_script:
            if not resolve_alias(data, ALIASES["legacy_script_hash"], enforce_hash=True, hash_len=64):
                missing.append("script hash")
            else:
                missing.append("legacy working-tree script hash (not platform-independent)")
        else:
            path_val = get_alias_value(data, ALIASES["producer_script_path"])
            if not path_val or not isinstance(path_val, str) or path_val.startswith('/'):
                script_missing = True

            if not resolve_alias(data, ALIASES["producer_script_hash"], enforce_hash=True, hash_len=64, lowercase_only=True):
                script_missing = True

            basis_val = get_alias_value(data, ALIASES["producer_script_hash_basis"])
            if basis_val != "git_blob_at_repository_commit":
                script_missing = True

            if script_missing:
                missing.append("script hash (requires repository-relative path, lowercase 64-hex blob hash, and correct basis string)")

        # Checkpoint group
        if not (resolve_alias(data, ALIASES["checkpoint_id"]) and resolve_alias(data, ALIASES["checkpoint_hash"], enforce_hash=True, hash_len=64)):
            missing.append("evaluated checkpoint identifier and 64-hex hash")

        # Settings group
        if not (resolve_alias(data, ALIASES["batch_size"]) and resolve_alias(data, ALIASES["precision"])):
            missing.append("evaluation batch size and precision")

        # Device group
        if not (resolve_alias(data, ALIASES["device_name"]) and resolve_alias(data, ALIASES["compute_capability"])):
            missing.append("device name and compute capability")
        if not (resolve_alias(data, ALIASES["env_python"]) and resolve_alias(data, ALIASES["env_torch"]) and resolve_alias(data, ALIASES["env_cuda"])):
            missing.append("environment Python, Torch, and CUDA versions")

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
