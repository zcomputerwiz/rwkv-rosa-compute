"""Configuration dataclasses for Experiment 0."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Task3SumConfig:
    """Configuration for 3SUM task data generation."""

    length: int = 12
    dimension: int = 3
    mod: int = 10
    num_filler: Optional[int] = None  # Defaults to length**2 if None
    true_rate: float = 0.5
    vocab_reduction: bool = True
    seed: int = 42
    num_samples: int = 10000

    def __post_init__(self):
        if self.mod != 10:
            raise ValueError(
                f"Modulus other than 10 is not supported in Experiment 0, "
                f"got mod={self.mod}"
            )


@dataclass
class ModelConfig:
    """Configuration for model backbones and initialization."""

    architecture: str = "llama"  # "llama" or "rwkv"
    init_mode: str = "random"  # "random" or "pretrained"
    rwkv_checkpoint: Optional[str] = None
    rwkv_checkpoint_sha256: Optional[str] = None
    hidden_size: int = 384
    num_hidden_layers: int = 4
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    head_dim: int = 64  # Relevant for RWKV-7 where heads = hidden_size // head_dim
    vocab_size: int = 256  # Actual token-set size optimization
    device: str = "cpu"

    def __post_init__(self):
        if self.architecture not in {"llama", "rwkv"}:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if self.init_mode not in {"random", "pretrained"}:
            raise ValueError(f"Unknown initialization mode: {self.init_mode}")


@dataclass
class TrainConfig:
    """Configuration for training and optimization."""

    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixture: str = "parallel_cot_filler"  # "parallel_cot_filler", "filler_only", "serial_cot_filler"
    parallel_ratio: float = 0.5
    filler_ratio: float = 0.5
    serial_ratio: float = 0.0
    immediate_ratio: float = 0.0
