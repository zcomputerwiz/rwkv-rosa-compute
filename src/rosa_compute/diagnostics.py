import os
import subprocess
import sys

import torch


def get_git_commit(path: str) -> str:
    """Gets git commit SHA for a submodule or repository path robustly."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        res = subprocess.run(cmd, cwd=path, capture_output=True, text=True, check=True)
        return res.stdout.strip()[:8]
    except Exception:
        # Fallback to reading file if git command fails
        git_file = os.path.join(path, ".git")
        if os.path.isfile(git_file):
            try:
                with open(git_file, "r") as f:
                    content = f.read().strip()
                if content.startswith("gitdir:"):
                    git_dir = os.path.abspath(os.path.join(path, content.split("gitdir:")[1].strip()))
                    head_file = os.path.join(git_dir, "HEAD")
                    if os.path.exists(head_file):
                        with open(head_file, "r") as hf:
                            head_content = hf.read().strip()
                        if head_content.startswith("ref:"):
                            ref_path = os.path.join(git_dir, head_content.split()[1])
                            if os.path.exists(ref_path):
                                with open(ref_path, "r") as rf:
                                    return rf.read().strip()[:8]
                        else:
                            return head_content[:8]
            except Exception:
                pass
        elif os.path.isdir(git_file):
            head_file = os.path.join(git_file, "HEAD")
            if os.path.exists(head_file):
                try:
                    with open(head_file, "r") as hf:
                        head_content = hf.read().strip()
                    if head_content.startswith("ref:"):
                        ref_path = os.path.join(git_file, head_content.split()[1])
                        if os.path.exists(ref_path):
                            with open(ref_path, "r") as rf:
                                return rf.read().strip()[:8]
                    else:
                        return head_content[:8]
                except Exception:
                    pass
    return "unknown"


def get_environment_info() -> dict:
    """Gathers system and dependency diagnostic details cleanly without failing on CPU."""
    info = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    }

    try:
        import rosa_soft
        info["rosa_soft_imported"] = True
        info["rosa_soft_version"] = getattr(rosa_soft, "__version__", "unknown")
        caps = getattr(rosa_soft, "BUILD_CAPABILITIES", None)
        info["rosa_soft_build_capabilities"] = caps
    except Exception as e:
        info["rosa_soft_imported"] = False
        info["rosa_soft_error"] = str(e)

    for sub in ["RWKV-LM", "rosa_soft"]:
        sub_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../external/{sub}"))
        info[f"{sub}_commit"] = get_git_commit(sub_path)

    return info


def print_diagnostics():
    """Prints formatted system diagnostic information."""
    info = get_environment_info()
    print("=== ROSA Compute Environment Diagnostics ===")
    print(f"Python Version:          {info['python_version']}")
    print(f"PyTorch Version:         {info['torch_version']}")
    print(f"CUDA Available:          {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"CUDA Runtime Version:    {info['cuda_version']}")
        print(f"GPU Name:                {info['gpu_name']}")
        print(f"Compute Capability:      {info['gpu_compute_capability']}")
    print(f"rosa_soft Imported:      {info['rosa_soft_imported']}")
    if info["rosa_soft_imported"]:
        print(f"rosa_soft Version:       {info.get('rosa_soft_version')}")
        caps = info.get("rosa_soft_build_capabilities")
        if caps:
            print(f"rosa_soft Variant:       {caps.variant}")
            print(f"Compiled Extension:      {caps.compiled_extension}")
            print(f"RosaRuntime Available:   {caps.rosa_runtime}")
            print(f"CUDA Kernels Available:  {caps.rosa_soft_cuda}")
    if "RWKV-LM_commit" in info:
        print(f"RWKV-LM Submodule Commit: {info['RWKV-LM_commit']}")
    if "rosa_soft_commit" in info:
        print(f"rosa_soft Submodule Commit: {info['rosa_soft_commit']}")


def _torch_setting(read):
    """Return a PyTorch setting, or None if this build cannot report it.

    Provenance must never be the reason an artifact fails to be written. A
    CPU-only wheel, or a future torch that moves one of these flags, should
    cost the field and nothing else.
    """
    try:
        return read()
    except Exception:
        return None


def get_artifact_environment() -> dict:
    """The JSON-safe subset of `get_environment_info` for embedding in artifacts.

    `get_environment_info` cannot be handed to `json.dump`: it returns a
    `BuildCapabilities` object, and `gpu_compute_capability` is a tuple. A
    writer must select named fields and coerce them rather than merge the dict.

    Filtering by type instead is the trap, and it is worse than it looks:
    `gpu_compute_capability` is the field that separates sm_75 from sm_86 from
    sm_89, so a scalar-only filter silently drops exactly the field a
    cross-device comparison needs and leaves an artifact that looks complete.

    The two submodule commits are included because a claim about the recurrence
    is not interpretable without knowing which kernel source produced it.

    The three float32 execution settings are read from PyTorch here rather than
    from `get_environment_info`, because they are process state at
    artifact-production time rather than build information. They are provenance
    only: nothing in this module sets them, and they are deliberately absent
    from checkpoint identity.

    They are recorded because `precision: "fp32"` in a run config does not by
    itself establish that matmuls ran in true FP32. TF32 exists on sm_80 and
    later and not on sm_75, so the same nominal precision can mean different
    arithmetic on different nodes, and an artifact recording only the config
    string cannot be checked afterwards.
    """
    info = get_environment_info()
    cap = info.get("gpu_compute_capability")
    return {
        "python_version": info.get("python_version"),
        "torch_version": str(info.get("torch_version")),
        "cuda_available": info.get("cuda_available"),
        "cuda_version": info.get("cuda_version"),
        "gpu_name": info.get("gpu_name"),
        "gpu_compute_capability": list(cap) if cap is not None else None,
        "float32_matmul_precision": _torch_setting(
            torch.get_float32_matmul_precision),
        "cuda_matmul_allow_tf32": _torch_setting(
            lambda: torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": _torch_setting(
            lambda: torch.backends.cudnn.allow_tf32),
        "RWKV-LM_commit": info.get("RWKV-LM_commit"),
        "rosa_soft_commit": info.get("rosa_soft_commit"),
    }


def get_repo_provenance(path: str) -> dict:
    """Full commit SHA, tree hash and dirty flag for an artifact's provenance.

    `get_git_commit` truncates to eight characters, which is enough to read and
    not enough to audit: a fleet artifact recorded `ab973d8c`, the tip of a
    branch that was squash-merged and deleted, and the SHA no longer resolved
    from a clone. The tree hash survives that -- a squash merge preserves the
    tree even though it rewrites the commit -- so recording both lets a
    reviewer confirm two differently-named commits built the same source.

    Returns "unknown" for any field git cannot supply, rather than raising: an
    analysis tool must still write its results outside a checkout.
    """
    out = {"commit": "unknown", "tree": "unknown", "dirty": None}
    try:
        out["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True,
            text=True, check=True).stdout.strip()
        out["tree"] = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=path, capture_output=True,
            text=True, check=True).stdout.strip()
        out["dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=path, capture_output=True,
            text=True, check=True).stdout.strip())
    except Exception:
        pass
    return out
