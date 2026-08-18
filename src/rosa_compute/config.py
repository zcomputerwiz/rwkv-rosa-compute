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
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if self.n_embd < 4:
            raise ValueError(f"n_embd must be >= 4, got {self.n_embd}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {self.context_length}")
        if self.rosa_bits != 4:
            raise ValueError(f"rosa_bits must be 4 for ROSA-4bit target, got {self.rosa_bits}")
        if self.n_embd % self.rosa_bits != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by rosa_bits ({self.rosa_bits})")
        if self.rosa_groups != self.n_embd // self.rosa_bits:
            raise ValueError(f"rosa_groups ({self.rosa_groups}) must equal n_embd / rosa_bits ({self.n_embd // self.rosa_bits})")
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point PyTorch dtype, got {self.dtype}")


DEFAULT_CONFIG = ROSAConfig()
