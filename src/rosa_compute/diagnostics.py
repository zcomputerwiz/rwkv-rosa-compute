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
