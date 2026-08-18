"""0B Backbone: RWKV-7 model adapter supporting inputs_embeds and complete RWKV-7 parameterization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def RWKV7_OP(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RWKV-7 CPU sequence recurrence op matching reference RWKV7_OP."""
    B, T, C = r.size()
    HEAD_SIZE = 64
    H = C // HEAD_SIZE
    N = HEAD_SIZE

    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))

    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float32)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float32)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)

    return out.view(B, T, C)


class RWKV7TimeMix(nn.Module):
    """Complete RWKV-7 Time-Mixing block ported from reference rwkv_v7_demo.py."""

    def __init__(self, hidden_size: int = 384, layer_id: int = 0, head_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim

        H = self.num_heads
        N = self.head_dim
        C = hidden_size

        D_DECAY_LORA = max(32, C // 12)
        D_AAA_LORA = max(32, C // 12)
        D_MV_LORA = max(16, C // 24)
        D_GATE_LORA = max(64, C // 6)

        self.x_r = nn.Parameter(torch.zeros(1, 1, C))
        self.x_w = nn.Parameter(torch.zeros(1, 1, C))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.x_v = nn.Parameter(torch.zeros(1, 1, C))
        self.x_a = nn.Parameter(torch.zeros(1, 1, C))
        self.x_g = nn.Parameter(torch.zeros(1, 1, C))

        self.w0 = nn.Parameter(torch.zeros(1, 1, C))
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.zeros(1, 1, C))
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.zeros(D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.zeros(1, 1, C))
        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.zeros(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.zeros(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C))
        self.k_a = nn.Parameter(torch.zeros(1, 1, C))
        self.r_k = nn.Parameter(torch.zeros(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        H = self.num_heads

        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        out = RWKV7_OP(r, w, k, v, -kk, kk * a)
        out = self.ln_x(out.view(B * T, C)).view(B, T, C)

        out = out + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        out = self.output(out * g)

        return out, v_first


class RWKV7ChannelMix(nn.Module):
    """Complete RWKV-7 Channel-Mixing (FFN) block with token shift."""

    def __init__(self, hidden_size: int = 384, intermediate_size: int = 1536):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)).pow(2)
        return self.value(k)


class RWKV7Layer(nn.Module):
    def __init__(self, hidden_size: int = 384, layer_id: int = 0, intermediate_size: int = 1536, head_dim: int = 64):
        super().__init__()
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size)
        else:
            self.ln0 = None

        self.ln1 = nn.LayerNorm(hidden_size)
        self.time_mix = RWKV7TimeMix(hidden_size=hidden_size, layer_id=layer_id, head_dim=head_dim)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.channel_mix = RWKV7ChannelMix(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_id == 0:
            x = self.ln0(x)

        tm_out, v_first = self.time_mix(self.ln1(x), v_first)
        x = x + tm_out
        x = x + self.channel_mix(self.ln2(x))
        return x, v_first


class RWKV7Backbone(nn.Module):
    """RWKV-7 backbone accepting inputs_embeds and threading v_first."""

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
                layer_id=i,
                intermediate_size=intermediate_size,
                head_dim=head_dim,
            )
            for i in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        x = inputs_embeds
        v_first = torch.empty_like(x)

        for layer in self.layers:
            x, v_first = layer(x, v_first)

        x = self.ln_out(x)
        return x
