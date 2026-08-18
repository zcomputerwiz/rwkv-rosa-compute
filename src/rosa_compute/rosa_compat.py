import os
import sys

import torch
import torch.nn as nn

_rosa_soft_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/rosa_soft"))
if _rosa_soft_path not in sys.path:
    sys.path.insert(0, _rosa_soft_path)

import rosa_soft


def apply_blinkdl_embedding(symbols: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """
    Applies BlinkDL transformation to pure ROSA symbol results:
    matched bit = 1 -> +embedding
    matched bit = 0 -> -embedding
    unmatched -> 0
    symbols: [B, T, C] where matched bits are 1.0 or 0.0, and unmatched are 0.0 (or masked)
    emb: [1, 1, C]
    """
    signs = torch.where(symbols == 1.0, 1.0, -1.0)
    unmatched = (symbols == 0.0) | torch.isnan(symbols)
    out = signs * emb
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
    """
    Adapter converting BlinkDL [B, T, 768] input layout to rosa_soft [B, T, H, D] layout (H=192, D=4),
    executing rosa_soft operator with max_suffix_length (default 512), and converting output back to [B, T, 768].
    """
    B, T, C = query.shape
    bits = 4
    assert C % bits == 0, f"C ({C}) must be divisible by {bits}"
    H = C // bits
    D = bits

    q_rs = query.view(B, T, H, D)
    k_rs = key.view(B, T, H, D)
    v_rs = value.view(B, T, H, D)

    if use_cuda and rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
        out_rs = rosa_soft.rosa_soft(
            q_rs, k_rs, v_rs, max_suffix_length=max_suffix_length
        )
    else:
        out_rs = rosa_soft.rosa_soft_reference(
            q_rs, k_rs, v_rs, max_suffix_length=max_suffix_length
        )

    return out_rs.reshape(B, T, C)

class ROSALayerCompat(nn.Module):
    """
    Full ROSA layer compatibility wrapper combining projections, ROSA routing, and output projection.
    """
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
        self.emb = nn.Parameter(torch.full((1, 1, n_embd), 1.0))
        self.o = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor, use_cuda: bool = False, use_blinkdl_ref: bool = False) -> torch.Tensor:
        xx = self.time_shift(x) - x
        q = self.q(x + xx * self.x_q)
        k = self.k(x + xx * self.x_k)
        v = self.v(x + xx * self.x_v)

        if use_blinkdl_ref:
            from .blinkdl_reference import blinkdl_rosa_4bit_reference
            y = blinkdl_rosa_4bit_reference(q, k, v, emb=self.emb)
        else:
            sym_out = rosa_4bit_forward(q, k, v, max_suffix_length=self.max_suffix_length, use_cuda=use_cuda)
            y = apply_blinkdl_embedding(sym_out, self.emb)

        return self.o(y)
