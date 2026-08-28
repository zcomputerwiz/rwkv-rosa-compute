#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rosa_compute.diagnostics import get_environment_info  # noqa: E402


def get_git_commit() -> tuple[str, bool]:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError:
        return "", False
    if res.returncode != 0:
        return "", False
    return res.stdout.strip(), True

def is_clean_checkout() -> bool:
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if res.returncode != 0:
        return False
    return len(res.stdout.strip()) == 0

def is_tracked_by_git(path: str) -> bool:
    res = subprocess.run(["git", "ls-files", "--error-unmatch", path], capture_output=True, text=True)
    return res.returncode == 0

def get_git_blob_sha256(commit: str, path: str) -> str:
    res = subprocess.run(["git", "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
    if res.returncode != 0:
        raise ValueError(f"Could not read git blob for {path} at {commit}")
    return hashlib.sha256(res.stdout).hexdigest()

def main():
    parser = argparse.ArgumentParser(
        description="Run a script and stamp its output artifact with provenance. "
                    "Must be run from inside the git repository checkout containing the producer script."
    )
    parser.add_argument("--artifact", required=True, help="Path to output JSON artifact")
    parser.add_argument("--producer", required=True, help="Repo-relative path to producer script")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    args = parser.parse_args()

    # The command should start after '--'
    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    # Fail closed pre-flight checks
    commit, success = get_git_commit()
    if not success:
        cwd = os.getcwd()
        print(f"Error: not inside a git repository (cwd: {cwd})", file=sys.stderr)
        sys.exit(1)

    if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit.lower()):
        print("Error: git rev-parse HEAD does not yield exactly 40 lowercase hexadecimal characters.", file=sys.stderr)
        sys.exit(1)

    if not is_clean_checkout():
        print("Error: Git checkout is not clean.", file=sys.stderr)
        sys.exit(1)

    if os.path.isabs(args.producer):
        print("Error: --producer must be a repository-relative path.", file=sys.stderr)
        sys.exit(1)

    if not is_tracked_by_git(args.producer):
        print("Error: --producer is not tracked by git.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(args.artifact):
        print("Error: --artifact already exists.", file=sys.stderr)
        sys.exit(1)

    # Get env info
    env_info_raw = get_environment_info()

    device_info = {
        "python_version": env_info_raw.get("python_version"),
        "torch_version": str(env_info_raw.get("torch_version")),
        "cuda_version": env_info_raw.get("cuda_version"),
        "cuda_available": env_info_raw.get("cuda_available"),
        "gpu_name": env_info_raw.get("gpu_name"),
        "gpu_compute_capability": list(env_info_raw.get("gpu_compute_capability")) if env_info_raw.get("gpu_compute_capability") else None,
    }

    try:
        blob_sha256 = get_git_blob_sha256(commit, args.producer)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Record start time
    started = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    # Run command
    res = subprocess.run(command)
    exit_code = res.returncode

    # Record finish time
    finished = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    # Post-flight check
    post_commit, post_success = get_git_commit()
    if not post_success or commit != post_commit:
        print("Error: Git commit changed during execution.", file=sys.stderr)
        sys.exit(1)

    provenance = {
        "repository_commit": commit,
        "producer_script_path": args.producer,
        "producer_script_git_blob_sha256": blob_sha256,
        "producer_script_hash_basis": "git_blob_at_repository_commit",
        "command": command,
        "exit_code": exit_code,
        "started": started,
        "finished": finished,
        "device": device_info,
    }

    artifact_data = {}
    if os.path.exists(args.artifact):
        try:
            with open(args.artifact, "r", encoding="utf-8") as f:
                artifact_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Artifact exists but is not valid JSON.", file=sys.stderr)
            sys.exit(exit_code)

    if "provenance" in artifact_data:
        print("Error: Artifact already contains a 'provenance' key.", file=sys.stderr)
        sys.exit(exit_code)

    artifact_data["provenance"] = provenance

    try:
        artifact_str = json.dumps(artifact_data, indent=2)
    except TypeError as e:
        print(f"Error serializing artifact: {e}", file=sys.stderr)
        sys.exit(1 if exit_code == 0 else exit_code)

    temp_artifact = args.artifact + ".tmp"
    with open(temp_artifact, "w", encoding="utf-8") as f:
        f.write(artifact_str)

    os.replace(temp_artifact, args.artifact)

    # Compute SHA-256 of the artifact
    h = hashlib.sha256()
    with open(args.artifact, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)

    digest = h.hexdigest()
    basename = os.path.basename(args.artifact)
    sidecar_path = args.artifact + ".sha256"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {basename}\n")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
