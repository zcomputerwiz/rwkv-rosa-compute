from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ROSAConfig:
    n_layer: int = 12
    n_embd: int = 768
    vocab_size: int = 65536
    rosa_bits: int = 4
    rosa_groups: int = 192
    context_length: int = 512
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.rosa_bits != 4:
            raise ValueError(f"rosa_bits must be 4 for ROSA-4bit target, got {self.rosa_bits}")
        if self.n_embd % self.rosa_bits != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by rosa_bits ({self.rosa_bits})")
        if self.rosa_groups != self.n_embd // self.rosa_bits:
            raise ValueError(f"rosa_groups ({self.rosa_groups}) must equal n_embd / rosa_bits ({self.n_embd // self.rosa_bits})")
        if self.context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {self.context_length}")

DEFAULT_CONFIG = ROSAConfig()
