import os
import sys

import torch
import torch.nn as nn

_rosa_soft_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/rosa_soft")
)
if not os.path.exists(_rosa_soft_path):
    raise ImportError(
        f"rosa_soft submodule path not found at {_rosa_soft_path}. "
        "Please run 'git submodule update --init --recursive' to initialize submodules."
    )

if _rosa_soft_path not in sys.path:
    sys.path.insert(0, _rosa_soft_path)

import rosa_soft

from .blinkdl_reference import blinkdl_rosa_4bit_reference


def apply_blinkdl_embedding(
    signed_rosa: torch.Tensor,
    emb: torch.Tensor,
) -> torch.Tensor:
    """Applies BlinkDL transformation to pure signed ROSA result {-1, 0, +1}:

    +1 -> +embedding
    -1 -> -embedding
     0 -> 0 (unmatched)
    signed_rosa: [B, T, C] where matched bits are +1.0 or -1.0, and unmatched are 0.0
    emb: [1, 1, C]
    """
    if not isinstance(signed_rosa, torch.Tensor) or not isinstance(emb, torch.Tensor):
        raise TypeError("signed_rosa and emb must be PyTorch Tensors")

    unmatched = (signed_rosa == 0.0) | torch.isnan(signed_rosa)
    out = signed_rosa * emb
    out[unmatched] = 0.0
    return out


def rosa_4bit_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    max_suffix_length: int = 512,
    use_cuda: bool = False,
) -> torch.Tensor:
    """Adapter converting BlinkDL [B, T, C] input layout to rosa_soft [B, T, H, D] layout (H=C//4, D=4),

    executing rosa_soft operator with max_suffix_length (default 512), and converting output back to [B, T, C].

    Returns pure signed ROSA output in {-1.0, 0.0, +1.0}.
    """
    if not (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        raise TypeError("query, key, and value must be PyTorch Tensors")

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            f"query, key, and value must be 3-D tensors [B, T, C], got ranks "
            f"{query.ndim}, {key.ndim}, {value.ndim}"
        )

    if (
        query.shape[:2] != key.shape[:2]
        or query.shape[:2] != value.shape[:2]
        or query.shape[2] != key.shape[2]
        or query.shape[2] != value.shape[2]
    ):
        raise ValueError(
            f"query, key, and value shapes must match, got "
            f"query={tuple(query.shape)}, key={tuple(key.shape)}, value={tuple(value.shape)}"
        )

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"query, key, and value dtypes must match, got "
            f"query={query.dtype}, key={key.dtype}, value={value.dtype}"
        )

    if not query.dtype.is_floating_point:
        raise ValueError(
            f"query, key, and value must have a floating-point dtype, got {query.dtype}"
        )

    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"query, key, and value devices must match, got "
            f"query={query.device}, key={key.device}, value={value.device}"
        )

    B, T, C = query.shape
    if B < 1 or T < 1 or C < 4:
        raise ValueError(
            f"Batch size, sequence length, and channels must satisfy B>=1, T>=1, C>=4; "
            f"got B={B}, T={T}, C={C}"
        )

    bits = 4
    if C % bits != 0:
        raise ValueError(f"C ({C}) must be divisible by {bits}")

    if max_suffix_length < 1:
        raise ValueError(f"max_suffix_length must be >= 1, got {max_suffix_length}")

    H = C // bits
    D = bits

    q_rs = query.reshape(B, T, H, D)
    k_rs = key.reshape(B, T, H, D)
    v_rs = value.reshape(B, T, H, D)

    if use_cuda:
        if not query.is_cuda:
            raise ValueError(
                f"use_cuda=True requires CUDA tensors, but query is on device {query.device}"
            )
        if not (
            hasattr(rosa_soft, "BUILD_CAPABILITIES")
            and rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda
        ):
            raise RuntimeError(
                "use_cuda=True was specified, but the rosa_soft CUDA extension is not available/built."
            )
        out_rs = rosa_soft.rosa_soft(
            q_rs, k_rs, v_rs, max_suffix_length=max_suffix_length
        )
    else:
        out_rs = rosa_soft.rosa_soft_reference(
            q_rs, k_rs, v_rs, max_suffix_length=max_suffix_length
        )

    return out_rs.reshape(B, T, C)


class ROSAQKV(nn.Module):
    """Submodule holding ROSA embedding parameter for state dict matching (rosa_qkv.emb)."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.emb = nn.Parameter(torch.full((1, 1, n_embd), 1.0))


class ROSALayerCompat(nn.Module):
    """Full ROSA layer compatibility wrapper combining projections, ROSA routing, and output projection."""

    def __init__(self, n_embd: int = 768, max_suffix_length: int = 512):
        super().__init__()
        self.n_embd = n_embd
        self.max_suffix_length = max_suffix_length
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_q = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.x_v = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
        self.rosa_qkv = ROSAQKV(n_embd)
        self.o = nn.Linear(n_embd, n_embd)

    @property
    def emb(self) -> nn.Parameter:
        return self.rosa_qkv.emb

    def forward(
        self,
        x: torch.Tensor,
        use_cuda: bool = False,
        use_blinkdl_ref: bool = False,
    ) -> torch.Tensor:
        xx = self.time_shift(x) - x
        q = self.q(x + xx * self.x_q)
        k = self.k(x + xx * self.x_k)
        v = self.v(x + xx * self.x_v)

        if use_blinkdl_ref:
            y = blinkdl_rosa_4bit_reference(q, k, v, emb=self.rosa_qkv.emb)
        else:
            signed_rosa = rosa_4bit_forward(
                q, k, v, max_suffix_length=self.max_suffix_length, use_cuda=use_cuda
            )
            y = apply_blinkdl_embedding(signed_rosa, self.rosa_qkv.emb)

        return self.o(y)
