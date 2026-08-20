"""Optional fused CUDA recurrence for Experiment 0 RWKV-7.

This module intentionally loads the pinned BlinkDL RWKV-7 CUDA source lazily.
The existing PyTorch recurrence remains the correctness oracle and default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

CHUNK_LEN = 16
SUPPORTED_HEAD_DIM = 64
PAD_RAW_W = -30.0
_KERNEL_LOADED = False
_KERNEL_LOAD_ERROR: Exception | None = None


def _source_dir() -> Path:
    override = os.environ.get("EXP0_RWKV7_CUDA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "external" / "RWKV-LM" / "RWKV-v7" / "train_temp" / "cuda"


def _operator_registered() -> bool:
    try:
        return hasattr(torch.ops.rwkv7_clampw, "forward") and hasattr(
            torch.ops.rwkv7_clampw,
            "backward",
        )
    except Exception:
        return False


def load_rwkv7_cuda_kernel() -> None:
    """Compile/load the pinned upstream RWKV-7 recurrence kernel once."""
    global _KERNEL_LOADED, _KERNEL_LOAD_ERROR

    if _KERNEL_LOADED:
        return
    if _operator_registered():
        raise RuntimeError(
            "A rwkv7_clampw operator was already registered before Experiment 0 "
            "loaded its pinned kernel. Refusing to reuse an unverified build; "
            "start a clean Python process."
        )
    if _KERNEL_LOAD_ERROR is not None:
        raise RuntimeError("RWKV-7 CUDA kernel previously failed to load") from _KERNEL_LOAD_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("RWKV-7 fused recurrence requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("RWKV-7 fused recurrence requires CUDA BF16 support")

    source_dir = _source_dir()
    cpp = source_dir / "rwkv7_clampw.cpp"
    cu = source_dir / "rwkv7_clampw.cu"
    missing = [str(path) for path in (cpp, cu) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pinned RWKV-LM CUDA sources are unavailable. Initialize the "
            "external/RWKV-LM submodule or set EXP0_RWKV7_CUDA_DIR. Missing: "
            + ", ".join(missing)
        )

    from torch.utils.cpp_extension import load

    # Mirror the flags in the pinned upstream x070 train_temp implementation.
    flags = [
        "-res-usage",
        f"-D_N_={SUPPORTED_HEAD_DIM}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas=-O3",
        "--extra-device-vectorization",
    ]
    try:
        load(
            name="rwkv7_clampw_exp0",
            sources=[str(cu), str(cpp)],
            is_python_module=False,
            verbose=False,
            extra_cuda_cflags=flags,
        )
        if not _operator_registered():
            raise RuntimeError("CUDA extension loaded without registering rwkv7_clampw")
        _KERNEL_LOADED = True
    except Exception as exc:
        _KERNEL_LOAD_ERROR = exc
        raise RuntimeError("Failed to compile/load the RWKV-7 CUDA kernel") from exc


class _RWKV7ClampW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        batch, timesteps, heads, head_dim = r.shape
        if timesteps % CHUNK_LEN != 0:
            raise ValueError("Internal RWKV-7 CUDA input must be chunk-aligned")
        if head_dim != SUPPORTED_HEAD_DIM:
            raise ValueError(
                f"RWKV-7 CUDA kernel supports head_dim={SUPPORTED_HEAD_DIM}, "
                f"got {head_dim}"
            )
        tensors = (r, w, k, v, a, b)
        if not all(t.dtype == torch.bfloat16 for t in tensors):
            raise TypeError("RWKV-7 CUDA kernel inputs must be bfloat16")
        if not all(t.is_cuda and t.is_contiguous() for t in tensors):
            raise ValueError("RWKV-7 CUDA kernel inputs must be contiguous CUDA tensors")

        out = torch.empty_like(v)
        state = torch.empty(
            batch,
            heads,
            timesteps // CHUNK_LEN,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=r.device,
        )
        sa = torch.empty(
            batch,
            timesteps,
            heads,
            head_dim,
            dtype=torch.float32,
            device=r.device,
        )
        torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b, out, state, sa)
        ctx.save_for_backward(r, w, k, v, a, b, state, sa)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        r, w, k, v, a, b, state, sa = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if grad_out.dtype != torch.bfloat16:
            grad_out = grad_out.to(torch.bfloat16)
        grads = [torch.empty_like(x) for x in (r, w, k, v, a, b)]
        torch.ops.rwkv7_clampw.backward(
            r,
            w,
            k,
            v,
            a,
            b,
            grad_out,
            state,
            sa,
            *grads,
        )
        return tuple(grads)


def _pad_time(
    tensor: torch.Tensor,
    pad: int,
    *,
    value: float = 0.0,
) -> torch.Tensor:
    if pad == 0:
        return tensor
    return F.pad(tensor, (0, 0, 0, pad), value=value)


def rwkv7_cuda_recurrence(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    """Run the upstream BF16 RWKV-7 recurrence with causal tail padding.

    ``raw_w`` is the pre-softplus decay parameter used by upstream x070. The
    CUDA kernel applies the decay transform internally. Inputs are explicitly
    converted to BF16, while the upstream kernel maintains recurrent state and
    saved backward state in FP32.
    """
    if head_dim != SUPPORTED_HEAD_DIM:
        raise ValueError(
            f"Fused RWKV-7 CUDA recurrence supports head_dim={SUPPORTED_HEAD_DIM}, "
            f"got {head_dim}"
        )
    if not all(x.is_cuda for x in (r, raw_w, k, v, a, b)):
        raise ValueError("Fused RWKV-7 recurrence requires CUDA tensors")

    load_rwkv7_cuda_kernel()

    batch, timesteps, channels = r.shape
    if channels % head_dim != 0:
        raise ValueError("RWKV channels must be divisible by head_dim")
    heads = channels // head_dim
    pad = (-timesteps) % CHUNK_LEN

    def prepare(
        x: torch.Tensor,
        *,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        x = _pad_time(x, pad, value=pad_value)
        return x.to(torch.bfloat16).contiguous().view(
            batch,
            timesteps + pad,
            heads,
            head_dim,
        )

    # The synthetic tail must not affect real causal outputs. Zero k/v/a/b is
    # sufficient for that. Give raw_w a strongly negative value so the kernel's
    # sigmoid(raw_w) is ~0 and padded steps have ~identity decay. This avoids
    # unnecessarily amplifying roundoff when the backward kernel reconstructs
    # states through the padded tail.
    inputs: Iterable[torch.Tensor] = (r, k, v, a, b)
    r4, k4, v4, a4, b4 = [prepare(x) for x in inputs]
    w4 = prepare(raw_w, pad_value=PAD_RAW_W)
    out = _RWKV7ClampW.apply(r4, w4, k4, v4, a4, b4)
    out = out.view(batch, timesteps + pad, channels)
    # Slicing a padded B>1 tensor leaves a larger batch stride. TimeMix uses
    # .view() immediately afterward, so return a contiguous logical sequence.
    return out[:, :timesteps, :].contiguous()
