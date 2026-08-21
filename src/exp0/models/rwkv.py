"""0B Backbone: RWKV-7 model adapter supporting inputs_embeds."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp0.models.rwkv_cuda import rwkv7_cuda_recurrence, rwkv7_cuda_step


@dataclass
class RWKV7LayerState:
    """Caller-owned persistent state for one RWKV-7 layer."""

    time_mix_previous: torch.Tensor
    channel_mix_previous: torch.Tensor
    recurrence: torch.Tensor


def RWKV7_OP(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_dim: int = 64,
    *,
    initial_state: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    if initial_state is None:
        state = torch.zeros(
            (batch, heads, head_size, head_size),
            device=r.device,
            dtype=torch.float32,
        )
    else:
        expected = (batch, heads, head_size, head_size)
        if initial_state.shape != expected:
            raise ValueError(
                f"RWKV-7 state must have shape {expected}, "
                f"got {tuple(initial_state.shape)}"
            )
        if initial_state.dtype != torch.float32:
            raise TypeError("RWKV-7 state must use torch.float32")
        if initial_state.device != r.device:
            raise ValueError("RWKV-7 state and recurrence inputs must share a device")
        state = initial_state

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

    result = out.view(batch, timesteps, channels)
    if return_state:
        return result, state
    return result


def rwkv7_reference_step(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one logical RWKV-7 transition with explicit recurrent state.

    The six recurrence inputs have shape ``[B, C]`` and may use any floating
    dtype. ``raw_w`` is the pre-softplus decay parameter used by the CUDA
    training kernel. The state has shape ``[B, H, N, N]`` where ``N`` is
    ``head_dim`` and ``H = C / N``. It must be contiguous FP32 on the input
    device; rows and columns are the value and key/receptance dimensions,
    respectively.

    This reference function does not mutate ``state``. It returns an FP32
    output of shape ``[B, C]`` and a newly allocated FP32 next state. Callers
    own both the input and returned state tensors.
    """
    batch, channels = r.shape
    heads = channels // head_dim
    expected_input = (batch, channels)
    tensors = (raw_w, k, v, a, b)
    if channels % head_dim:
        raise ValueError("RWKV channels must be divisible by head_dim")
    if any(tensor.shape != expected_input for tensor in tensors):
        raise ValueError("RWKV-7 step inputs must all have shape [B, C]")
    expected_state = (batch, heads, head_dim, head_dim)
    if state.shape != expected_state:
        raise ValueError(
            f"RWKV-7 state must have shape {expected_state}, "
            f"got {tuple(state.shape)}"
        )
    if state.dtype != torch.float32:
        raise TypeError("RWKV-7 state must use torch.float32")
    if not state.is_contiguous():
        raise ValueError("RWKV-7 state must be contiguous")
    if any(tensor.device != r.device for tensor in (*tensors, state)):
        raise ValueError("RWKV-7 state and recurrence inputs must share a device")

    transformed_w = -F.softplus(-raw_w.float()) - 0.5
    return RWKV7_OP_step(
        r,
        transformed_w,
        k,
        v,
        a,
        b,
        state,
        head_dim=head_dim,
    )


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
    """Legacy one-step helper accepting RWKV7_OP's transformed ``w``."""
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
        # torch.compile treats integer nn.Module attributes as static, so
        # branching on layer_id specializes this frame once per layer. With
        # 12 layers that exceeds dynamo's recompile limit of 8 and the rest
        # fall back to eager silently. A bool has two values, so the guard
        # collapses to two specializations with identical semantics.
        self.is_first_layer = layer_id == 0
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

        if self.is_first_layer or v_first is None:
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

    def forward_step(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        previous_x: torch.Tensor,
        recurrence_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one token with caller-owned shift and recurrent state."""
        batch, channels = x.shape
        heads = self.num_heads
        if channels != self.hidden_size:
            raise ValueError(
                f"RWKV-7 step expected hidden size {self.hidden_size}, "
                f"got {channels}"
            )
        if previous_x.shape != x.shape:
            raise ValueError("RWKV-7 TimeMix previous state must match x")

        xx = previous_x - x
        xr = x + xx * self.x_r.view(1, channels)
        xw = x + xx * self.x_w.view(1, channels)
        xk = x + xx * self.x_k.view(1, channels)
        xv = x + xx * self.x_v.view(1, channels)
        xa = x + xx * self.x_a.view(1, channels)
        xg = x + xx * self.x_g.view(1, channels)

        r = self.receptance(xr)
        raw_w = self.w0.view(1, channels) + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)

        if self.is_first_layer or v_first is None:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0.view(1, channels) + (xv @ self.v1) @ self.v2
            )

        a = torch.sigmoid(
            self.a0.view(1, channels) + (xa @ self.a1) @ self.a2
        )
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k.view(1, channels)
        kk = F.normalize(
            kk.view(batch, heads, self.head_dim),
            dim=-1,
            p=2.0,
        ).view(batch, channels)
        k = k * (1 + (a - 1) * self.k_a.view(1, channels))

        if self.rwkv_kernel == "cuda":
            out, next_recurrence = rwkv7_cuda_step(
                r,
                raw_w,
                k,
                v,
                -kk,
                kk * a,
                recurrence_state,
                head_dim=self.head_dim,
            )
        else:
            out, next_recurrence = rwkv7_reference_step(
                r,
                raw_w,
                k,
                v,
                -kk,
                kk * a,
                recurrence_state,
                head_dim=self.head_dim,
            )

        out = self.ln_x(out)
        rkv = (
            r.view(batch, heads, self.head_dim)
            * k.view(batch, heads, self.head_dim)
            * self.r_k
        ).sum(dim=-1, keepdim=True)
        out = out + (rkv * v.view(batch, heads, self.head_dim)).view(
            batch,
            channels,
        )
        out = self.output(out * g)
        previous_x.copy_(x)
        return out, v_first, next_recurrence


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

    def forward_step(
        self,
        x: torch.Tensor,
        previous_x: torch.Tensor,
    ) -> torch.Tensor:
        """Run one token and update the caller-owned token-shift state."""
        if previous_x.shape != x.shape:
            raise ValueError("RWKV-7 ChannelMix previous state must match x")
        xx = previous_x - x
        k = x + xx * self.x_k.view(1, x.shape[-1])
        k = torch.relu(self.key(k)).pow(2)
        out = self.value(k)
        previous_x.copy_(x)
        return out


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
        # Guard on ln0 alone, not layer_id: ln0 is constructed if and only
        # if layer_id == 0, so the comparison is redundant - but torch.compile
        # treats integer nn.Module attributes as static, so branching on
        # layer_id specializes this frame once per layer. With 12 layers that
        # exceeds dynamo's recompile limit of 8 and the remaining layers fall
        # back to eager silently.
        if self.ln0 is not None:
            x = self.ln0(x)

        tm_out, v_first = self.time_mix(self.ln1(x), v_first)
        x = x + tm_out
        x = x + self.channel_mix(self.ln2(x))
        return x, v_first

    def forward_step(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        state: RWKV7LayerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one forward-only token and mutate this layer's state."""
        # Guard on ln0 alone, not layer_id: ln0 is constructed if and only
        # if layer_id == 0, so the comparison is redundant - but torch.compile
        # treats integer nn.Module attributes as static, so branching on
        # layer_id specializes this frame once per layer. With 12 layers that
        # exceeds dynamo's recompile limit of 8 and the remaining layers fall
        # back to eager silently.
        if self.ln0 is not None:
            x = self.ln0(x)

        tm_out, v_first, next_recurrence = self.time_mix.forward_step(
            self.ln1(x),
            v_first,
            state.time_mix_previous,
            state.recurrence,
        )
        state.recurrence = next_recurrence
        x = x + tm_out
        x = x + self.channel_mix.forward_step(
            self.ln2(x),
            state.channel_mix_previous,
        )
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

    def init_step_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        activation_dtype: torch.dtype | None = None,
    ) -> list[RWKV7LayerState]:
        """Allocate zeroed caller-owned state for forward-only stepping."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        state_device = parameter.device if device is None else torch.device(device)
        state_dtype = parameter.dtype if activation_dtype is None else activation_dtype
        hidden_size = self.layers[0].time_mix.hidden_size
        head_dim = self.layers[0].time_mix.head_dim
        heads = hidden_size // head_dim
        if self.layers[0].time_mix.rwkv_kernel == "cuda" and (
            batch_size != 1
            or hidden_size != 768
            or heads != 12
            or head_dim != 64
            or state_dtype != torch.bfloat16
        ):
            raise ValueError(
                "CUDA step state requires B=1, hidden=768, heads=12, "
                "head_dim=64, and BF16 activations"
            )
        return [
            RWKV7LayerState(
                time_mix_previous=torch.zeros(
                    (batch_size, hidden_size),
                    device=state_device,
                    dtype=state_dtype,
                ),
                channel_mix_previous=torch.zeros(
                    (batch_size, hidden_size),
                    device=state_device,
                    dtype=state_dtype,
                ),
                recurrence=torch.zeros(
                    (batch_size, heads, head_dim, head_dim),
                    device=state_device,
                    dtype=torch.float32,
                ),
            )
            for _ in self.layers
        ]

    @torch.no_grad()
    def forward_step(
        self,
        inputs_embeds: torch.Tensor,
        state: list[RWKV7LayerState],
    ) -> tuple[torch.Tensor, list[RWKV7LayerState]]:
        """Run one token through all layers with persistent state."""
        if inputs_embeds.ndim != 2:
            raise ValueError("RWKV-7 step inputs_embeds must have shape [B, C]")
        if len(state) != len(self.layers):
            raise ValueError("RWKV-7 step state must contain one entry per layer")
        x = inputs_embeds
        v_first = None
        for layer, layer_state in zip(self.layers, state):
            x, v_first = layer.forward_step(x, v_first, layer_state)
        return self.ln_out(x), state
