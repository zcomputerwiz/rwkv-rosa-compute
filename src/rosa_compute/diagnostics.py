import os
import sys

import torch


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
        info["rosa_soft_build_capabilities"] = getattr(rosa_soft, "BUILD_CAPABILITIES", None)
    except Exception as e:
        info["rosa_soft_imported"] = False
        info["rosa_soft_error"] = str(e)

    for sub in ["RWKV-LM", "rosa_soft"]:
        sub_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../external/{sub}"))
        head_file = os.path.join(sub_path, ".git/HEAD")
        if os.path.exists(head_file):
            try:
                with open(head_file, "r") as f:
                    content = f.read().strip()
                if content.startswith("ref:"):
                    ref_path = os.path.join(sub_path, ".git", content.split()[1])
                    if os.path.exists(ref_path):
                        with open(ref_path, "r") as f:
                            info[f"{sub}_commit"] = f.read().strip()[:8]
                else:
                    info[f"{sub}_commit"] = content[:8]
            except Exception:
                info[f"{sub}_commit"] = "unknown"

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
        print(f"RWKV-LM Submodule Commit:{info['RWKV-LM_commit']}")
    if "rosa_soft_commit" in info:
        print(f"rosa_soft Submodule Commit: {info['rosa_soft_commit']}")
