"""0A Backbone: Small Llama transformer supporting inputs_embeds."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply pairwise rotary position encoding to the final dimension."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        (
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ),
        dim=-1,
    )
    return rotated.flatten(-2)


class RotaryEmbedding(nn.Module):
    """Standard Llama-style rotary position encoding for Q/K projections."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        if base <= 0:
            raise ValueError("RoPE base must be greater than zero.")

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )
        frequencies = torch.outer(positions, self.inv_freq.to(device=device))
        # [1, 1, T, D/2] broadcasts over batch and attention heads.
        cos = frequencies.cos().to(dtype=dtype)[None, None, :, :]
        sin = frequencies.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape:
            raise ValueError("RoPE Q/K tensors must have identical shapes.")
        cos, sin = self.cos_sin(
            q.shape[-2],
            device=q.device,
            dtype=q.dtype,
        )
        return apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Llama attention head_dim must be even for RoPE.")

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, base=rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, channels = x.shape
        q = self.q_proj(x).view(
            batch,
            timesteps,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch,
            timesteps,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch,
            timesteps,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k = self.rotary(q, k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, timesteps, channels)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int = 384, intermediate_size: int = 1536):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
        intermediate_size: int = 1536,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rope_theta=rope_theta,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaBackbone(nn.Module):
    """Small Llama backbone with causal attention and rotary position encoding."""

    def __init__(
        self,
        hidden_size: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        intermediate_size: int = 1536,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    rope_theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
