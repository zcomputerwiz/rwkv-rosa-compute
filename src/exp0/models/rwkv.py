"""0B Backbone: RWKV-7 model adapter supporting inputs_embeds."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKV7TimeMix(nn.Module):
    """Reference implementation for RWKV-7 Time-Mixing block."""

    def __init__(self, hidden_size: int = 384, head_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim

        self.r_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.num_heads
        N = self.head_dim

        r = self.r_proj(x).view(B, T, H, N)
        w = torch.exp(-torch.exp(self.w_proj(x).view(B, T, H, N)))
        k = self.k_proj(x).view(B, T, H, N)
        v = self.v_proj(x).view(B, T, H, N)
        a = self.a_proj(x).view(B, T, H, N)
        b = self.b_proj(x).view(B, T, H, N)

        out = torch.zeros((B, T, H, N), device=x.device, dtype=x.dtype)
        state = torch.zeros((B, H, N, N), device=x.device, dtype=x.dtype)

        for t in range(T):
            kk, rr, vv, aa, bb = k[:, t, :], r[:, t, :], v[:, t, :], a[:, t, :], b[:, t, :]
            sab = torch.einsum("bhik,bhk,bhj->bhij", state, aa, bb)
            state = state * w[:, t, :, None, :] + sab + torch.einsum("bhj,bhi->bhij", kk, vv)
            out[:, t, :] = torch.einsum("bhj,bhij->bhi", rr, state)

        out = out.view(B, T, C)
        return self.out_proj(out)


class RWKV7ChannelMix(nn.Module):
    """RWKV-7 Channel-Mixing (FFN) block."""

    def __init__(self, hidden_size: int = 384, intermediate_size: int = 1536):
        super().__init__()
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = F.relu(self.key(x)).pow(2)
        return self.value(k)


class RWKV7Layer(nn.Module):
    def __init__(self, hidden_size: int = 384, intermediate_size: int = 1536, head_dim: int = 64):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.time_mix = RWKV7TimeMix(hidden_size=hidden_size, head_dim=head_dim)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.channel_mix = RWKV7ChannelMix(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class RWKV7Backbone(nn.Module):
    """RWKV-7 backbone accepting inputs_embeds."""

    def __init__(
        self,
        hidden_size: int = 384,
        num_layers: int = 4,
        intermediate_size: int = 1536,
        head_dim: int = 64,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RWKV7Layer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
            )
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        return x
