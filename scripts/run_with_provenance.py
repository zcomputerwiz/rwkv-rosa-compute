#!/usr/bin/env python3
"""Run a script and merge environment provenance into its JSON artifact."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Fix: Use the canonical diagnostics module instead of duplicating torch logic
# Ensure src is in the path to import rosa_compute
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
try:
    from rosa_compute.diagnostics import get_environment_info
except ImportError:
    def get_environment_info():
        return {
            "python_version": sys.version.split()[0],
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "gpu_compute_capability": None,
        }


def get_git_commit():
    """Call git rev-parse HEAD directly to get the FULL 40-character SHA."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_git_dirty():
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        return [line[3:] for line in res.stdout.splitlines() if line]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def get_script_sha256(command):
    for arg in command:
        if arg.endswith('.py') and os.path.exists(arg):
            try:
                h = hashlib.sha256()
                with open(arg, 'rb') as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                return h.hexdigest()
            except OSError:
                pass
    return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path, help="Path to the JSON artifact")
    parser.add_argument("--no-sidecar", action="store_true", help="Skip writing the .sha256 sidecar")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    args = parser.parse_args()

    if not args.command:
        print("Error: No command provided to run.", file=sys.stderr)
        return 1

    if args.command[0] == "--":
        args.command = args.command[1:]

    commit_before = get_git_commit()
    started_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    # Run the command
    res = subprocess.run(args.command)
    exit_code = res.returncode

    finished_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    commit_after = get_git_commit()
    dirty_files = get_git_dirty()
    script_hash = get_script_sha256(args.command)

    # Build device block using the canonical helper
    devices = get_environment_info()

    commits = []
    if commit_before:
        commits.append(commit_before)
    if commit_after and commit_after != commit_before:
        commits.append(commit_after)

    # "Split commit and script hash into two separate required fields"
    # The requirement is both the exact repository commit AND the producer script's own SHA-256.
    provenance = {
        "repository_commit": commits[0] if len(commits) == 1 else commits,
        "producer_script_sha256": script_hash,
        "dirty": dirty_files,
        "command": args.command,
        "device": devices,
        "platform": platform.platform(),
        "started": started_at,
        "finished": finished_at,
        "exit_code": exit_code
    }

    if not args.artifact.exists():
        print(f"Error: Artifact {args.artifact} does not exist.", file=sys.stderr)
        return exit_code if exit_code != 0 else 1

    try:
        with open(args.artifact, "r", encoding="utf-8") as f:
            artifact_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Artifact {args.artifact} is not valid JSON.", file=sys.stderr)
        return exit_code if exit_code != 0 else 1
    except OSError as e:
        print(f"Error reading artifact: {e}", file=sys.stderr)
        return exit_code if exit_code != 0 else 1

    if "provenance" in artifact_data:
        print(f"Error: Artifact {args.artifact} already contains a 'provenance' key.", file=sys.stderr)
        return exit_code if exit_code != 0 else 1

    artifact_data["provenance"] = provenance

    try:
        with open(args.artifact, "w", encoding="utf-8") as f:
            json.dump(artifact_data, f, indent=2)
    except OSError as e:
        print(f"Error writing artifact: {e}", file=sys.stderr)
        return exit_code if exit_code != 0 else 1

    if not args.no_sidecar:
        try:
            h = hashlib.sha256()
            with open(args.artifact, 'rb') as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            digest = h.hexdigest()
            filename = args.artifact.name
            sidecar_path = args.artifact.with_suffix(args.artifact.suffix + ".sha256")
            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.write(f"{digest}  {filename}\n")
        except OSError as e:
            print(f"Error writing sidecar: {e}", file=sys.stderr)
            return exit_code if exit_code != 0 else 1

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
