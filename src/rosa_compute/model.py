import torch
import torch.nn as nn

from .config import DEFAULT_CONFIG, ROSAConfig
from .rosa_compat import ROSALayerCompat


class ROSAModelSkeleton(nn.Module):
    """
    Skeleton model wrapper for 0.1B ROSA-4bit model architecture.
    """
    def __init__(self, config: ROSAConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([
            ROSALayerCompat(n_embd=config.n_embd, max_suffix_length=config.context_length)
            for _ in range(config.n_layer)
        ])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, use_cuda: bool = False) -> torch.Tensor:
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x, use_cuda=use_cuda)
        x = self.ln_out(x)
        return self.head(x)
