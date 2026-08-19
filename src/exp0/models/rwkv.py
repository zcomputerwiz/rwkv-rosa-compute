"""0B Backbone: RWKV-7 model adapter supporting inputs_embeds."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp0.models.rwkv_cuda import rwkv7_cuda_recurrence


def RWKV7_OP(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_dim: int = 64,
) -> torch.Tensor:
    """Reference FP32 RWKV-7 sequence recurrence.

    ``w`` is the already transformed log-decay used by the historical
    Experiment 0 implementation. Keeping this function unchanged provides a
    stable forward/backward oracle for the optional fused CUDA backend.
    """
    batch, timesteps, channels = r.size()
    head_size = head_dim
    heads = channels // head_size

    r = r.view(batch, timesteps, heads, head_size).float()
    k = k.view(batch, timesteps, heads, head_size).float()
    v = v.view(batch, timesteps, heads, head_size).float()
    a = a.view(batch, timesteps, heads, head_size).float()
    b = b.view(batch, timesteps, heads, head_size).float()
    w = torch.exp(-torch.exp(w.view(batch, timesteps, heads, head_size).float()))

    out = torch.zeros(
        (batch, timesteps, heads, head_size),
        device=r.device,
        dtype=torch.float32,
    )
    state = torch.zeros(
        (batch, heads, head_size, head_size),
        device=r.device,
        dtype=torch.float32,
    )

    for timestep in range(timesteps):
        kk = k[:, timestep, :].view(batch, heads, 1, head_size)
        rr = r[:, timestep, :].view(batch, heads, head_size, 1)
        vv = v[:, timestep, :].view(batch, heads, head_size, 1)
        aa = a[:, timestep, :].view(batch, heads, head_size, 1)
        bb = b[:, timestep, :].view(batch, heads, 1, head_size)
        state = (
            state * w[:, timestep, :, None, :]
            + state @ aa @ bb
            + vv @ kk
        )
        out[:, timestep, :] = (state @ rr).view(batch, heads, head_size)

    return out.view(batch, timesteps, channels)


def RWKV7_OP_step(
    r_t: torch.Tensor,
    w_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    state: torch.Tensor,
    head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step reference recurrent forward pass for RWKV7_OP."""
    batch, channels = r_t.size()
    head_size = head_dim
    heads = channels // head_size

    rr = r_t.view(batch, heads, head_size, 1).float()
    kk = k_t.view(batch, heads, 1, head_size).float()
    vv = v_t.view(batch, heads, head_size, 1).float()
    aa = a_t.view(batch, heads, head_size, 1).float()
    bb = b_t.view(batch, heads, 1, head_size).float()
    ww = torch.exp(-torch.exp(w_t.view(batch, heads, head_size, 1).float()))

    next_state = state * ww.swapaxes(-1, -2) + state @ aa @ bb + vv @ kk
    out_t = (next_state @ rr).view(batch, channels)

    return out_t, next_state


class RWKV7TimeMix(nn.Module):
    """Complete RWKV-7 Time-Mixing block with reference initialization."""

    def __init__(
        self,
        hidden_size: int = 384,
        layer_id: int = 0,
        num_layers: int = 4,
        head_dim: int = 64,
        rwkv_kernel: str = "reference",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.rwkv_kernel = rwkv_kernel

        heads = self.num_heads
        head_size = self.head_dim
        channels = hidden_size

        assert head_size > 1, (
            f"head_dim must be > 1 for zigzag computation, got {head_size}"
        )

        decay_lora = max(
            32,
            int(round((2.5 * (channels**0.5)) / 32) * 32),
        )
        aaa_lora = max(
            32,
            int(round((2.5 * (channels**0.5)) / 32) * 32),
        )
        mv_lora = max(
            32,
            int(round((1.7 * (channels**0.5)) / 32) * 32),
        )
        gate_lora = max(
            32,
            int(round((5.0 * (channels**0.5)) / 32) * 32),
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.ln_x = nn.GroupNorm(heads, channels, eps=64e-5)

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(1, num_layers - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)

            ddd = torch.ones(1, 1, channels)
            for i in range(channels):
                ddd[0, 0, i] = i / channels

            self.x_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)
            )
            self.x_w = nn.Parameter(
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
            )
            self.x_k = nn.Parameter(
                1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)
            )
            self.x_v = nn.Parameter(
                1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)
            )
            self.x_a = nn.Parameter(
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
            )
            self.x_g = nn.Parameter(
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)
            )

            def ortho_init(x, scale):
                shape = x.shape
                assert len(shape) == 2
                gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                nn.init.orthogonal_(x, gain=gain * scale)
                return x

            www = torch.zeros(channels)
            zigzag = torch.zeros(channels)
            linear = torch.zeros(channels)
            for n in range(channels):
                linear[n] = n / (channels - 1) - 0.5
                zigzag[n] = (
                    (n % head_size) - ((head_size - 1) / 2)
                ) / ((head_size - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (channels - 1)) ** (
                    1 + ratio_0_to_1**0.3
                )

            self.w1 = nn.Parameter(torch.zeros(channels, decay_lora))
            self.w2 = nn.Parameter(
                ortho_init(torch.zeros(decay_lora, channels), 0.1)
            )
            self.w0 = nn.Parameter(
                www.reshape(1, 1, channels) + 0.5 + zigzag * 2.5
            )

            self.a1 = nn.Parameter(torch.zeros(channels, aaa_lora))
            self.a2 = nn.Parameter(
                ortho_init(torch.zeros(aaa_lora, channels), 0.1)
            )
            self.a0 = nn.Parameter(
                torch.zeros(1, 1, channels) - 0.19 + zigzag * 0.3 + linear * 0.4
            )

            self.v1 = nn.Parameter(torch.zeros(channels, mv_lora))
            self.v2 = nn.Parameter(
                ortho_init(torch.zeros(mv_lora, channels), 0.1)
            )
            self.v0 = nn.Parameter(
                torch.zeros(1, 1, channels) + 0.73 - linear * 0.4
            )

            self.g1 = nn.Parameter(torch.zeros(channels, gate_lora))
            self.g2 = nn.Parameter(
                ortho_init(torch.zeros(gate_lora, channels), 0.1)
            )

            self.k_k = nn.Parameter(
                torch.zeros(1, 1, channels) + 0.71 - linear * 0.1
            )
            self.k_a = nn.Parameter(torch.zeros(1, 1, channels) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(heads, head_size) - 0.04)

            nn.init.zeros_(self.output.weight)

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, timesteps, channels = x.size()
        heads = self.num_heads

        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        raw_w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0 or v_first is None:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(
            kk.view(batch, timesteps, heads, -1),
            dim=-1,
            p=2.0,
        ).view(batch, timesteps, channels)
        k = k * (1 + (a - 1) * self.k_a)

        if self.rwkv_kernel == "cuda":
            out = rwkv7_cuda_recurrence(
                r,
                raw_w,
                k,
                v,
                -kk,
                kk * a,
                head_dim=self.head_dim,
            )
        else:
            w = -F.softplus(-raw_w) - 0.5
            out = RWKV7_OP(
                r,
                w,
                k,
                v,
                -kk,
                kk * a,
                head_dim=self.head_dim,
            )

        out = self.ln_x(out.view(batch * timesteps, channels)).view(
            batch,
            timesteps,
            channels,
        )
        rkv = (
            r.view(batch, timesteps, heads, -1)
            * k.view(batch, timesteps, heads, -1)
            * self.r_k
        ).sum(dim=-1, keepdim=True)
        out = out + (rkv * v.view(batch, timesteps, heads, -1)).view(
            batch,
            timesteps,
            channels,
        )
        out = self.output(out * g)

        return out, v_first


class RWKV7ChannelMix(nn.Module):
    """Complete RWKV-7 Channel-Mixing (FFN) block with token shift."""

    def __init__(
        self,
        hidden_size: int = 384,
        layer_id: int = 0,
        num_layers: int = 4,
        intermediate_size: int = 1536,
    ):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)
            ddd = torch.ones(1, 1, hidden_size)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size
            self.x_k = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0**4)
            )

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)).pow(2)
        return self.value(k)


class RWKV7Layer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        layer_id: int = 0,
        num_layers: int = 4,
        intermediate_size: int = 1536,
        head_dim: int = 64,
        rwkv_kernel: str = "reference",
    ):
        super().__init__()
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size)
        else:
            self.ln0 = None

        self.ln1 = nn.LayerNorm(hidden_size)
        self.time_mix = RWKV7TimeMix(
            hidden_size=hidden_size,
            layer_id=layer_id,
            num_layers=num_layers,
            head_dim=head_dim,
            rwkv_kernel=rwkv_kernel,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.channel_mix = RWKV7ChannelMix(
            hidden_size=hidden_size,
            layer_id=layer_id,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_id == 0 and self.ln0 is not None:
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
        rwkv_kernel: str = "reference",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RWKV7Layer(
                    hidden_size=hidden_size,
                    layer_id=i,
                    num_layers=num_layers,
                    intermediate_size=intermediate_size,
                    head_dim=head_dim,
                    rwkv_kernel=rwkv_kernel,
                )
                for i in range(num_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        x = inputs_embeds
        v_first = None

        for layer in self.layers:
            x, v_first = layer(x, v_first)

        return self.ln_out(x)
