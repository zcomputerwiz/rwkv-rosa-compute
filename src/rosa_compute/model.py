import torch
import torch.nn as nn

from .config import DEFAULT_CONFIG, ROSAConfig
from .rosa_compat import ROSALayerCompat


class RWKV_CMix_x070(nn.Module):
    """ChannelMix / FFN submodule matching BlinkDL RWKV-7/RWKV-8 architecture."""

    def __init__(self, n_embd: int, dim_ffn: int | None = None):
        super().__init__()
        if dim_ffn is None:
            dim_ffn = n_embd * 4
        self.n_embd = n_embd
        self.dim_ffn = dim_ffn
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class ROSABlock(nn.Module):
    """Single RWKV-8 ROSA Block containing optional ln0, ln3, ROSA layer, ln2, and FFN layer."""

    def __init__(self, config: ROSAConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)

        self.rosa = ROSALayerCompat(
            n_embd=config.n_embd, max_suffix_length=config.context_length
        )
        self.ffn = RWKV_CMix_x070(n_embd=config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        use_cuda: bool = False,
        use_blinkdl_ref: bool = False,
    ) -> torch.Tensor:
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.rosa(
            self.ln3(x), use_cuda=use_cuda, use_blinkdl_ref=use_blinkdl_ref
        )
        x = x + self.ffn(self.ln2(x))

        return x


class ROSAModelSkeleton(nn.Module):
    """Skeleton compatibility scaffold matching BlinkDL RWKV-8 ROSA-4bit block structure."""

    def __init__(self, config: ROSAConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [ROSABlock(config, i) for i in range(config.n_layer)]
        )
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.dtype != torch.float32:
            self.to(dtype=config.dtype)

    def forward(
        self,
        idx: torch.Tensor,
        use_cuda: bool = False,
        use_blinkdl_ref: bool = False,
    ) -> torch.Tensor:
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, use_cuda=use_cuda, use_blinkdl_ref=use_blinkdl_ref)
        x = self.ln_out(x)
        return self.head(x)
