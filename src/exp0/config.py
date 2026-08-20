"""Configuration dataclasses for Experiment 0."""

from dataclasses import dataclass
from typing import Optional

from exp0.task3sum import (
    DEFAULT_CORRUPTION_RATE,
    GENERATOR_MODES,
    SOURCE_GENERATOR,
)


@dataclass
class Task3SumConfig:
    """Configuration for 3SUM task data generation."""

    length: int = 12
    dimension: int = 3
    mod: int = 10
    num_filler: Optional[int] = None
    true_rate: float = 0.5
    vocab_reduction: bool = True
    include_separator_token: bool = True
    generator_mode: str = SOURCE_GENERATOR
    corruption_rate: float = DEFAULT_CORRUPTION_RATE
    seed: int = 42
    num_samples: int = 10000

    def __post_init__(self):
        if self.mod != 10:
            raise ValueError(
                f"Modulus other than 10 is not supported in Experiment 0, "
                f"got mod={self.mod}"
            )
        if not 0.0 <= self.true_rate <= 1.0:
            raise ValueError("true_rate must be in [0, 1].")
        if not self.include_separator_token:
            raise ValueError(
                "Experiment 0 requires the supervised continuation separator. "
                "The pre-repair separator-dropping protocol is not supported."
            )
        if self.generator_mode not in GENERATOR_MODES:
            raise ValueError(
                f"generator_mode must be one of {GENERATOR_MODES}; "
                f"got {self.generator_mode!r}"
            )
        if self.corruption_rate < 1.0:
            raise ValueError("corruption_rate must be >= 1.0.")


@dataclass
class ModelConfig:
    """Configuration for model backbones and initialization."""

    architecture: str = "llama"
    init_mode: str = "random"
    rwkv_checkpoint: Optional[str] = None
    rwkv_checkpoint_sha256: Optional[str] = None
    rwkv_kernel: str = "reference"
    hidden_size: int = 384
    num_hidden_layers: int = 4
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    head_dim: int = 64
    llama_rope_theta: float = 10000.0
    llama_initializer_range: float = 0.02
    match3_shared_input_features: bool = True
    vocab_size: int = 256
    device: str = "cpu"

    def __post_init__(self):
        if self.architecture not in {"llama", "rwkv"}:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if self.init_mode not in {"random", "pretrained"}:
            raise ValueError(f"Unknown initialization mode: {self.init_mode}")
        if self.rwkv_kernel not in {"reference", "cuda"}:
            raise ValueError(
                "rwkv_kernel must be one of: reference, cuda; "
                f"got {self.rwkv_kernel!r}"
            )
        if self.architecture != "rwkv" and self.rwkv_kernel != "reference":
            raise ValueError("rwkv_kernel='cuda' is only valid for RWKV models.")
        if self.rwkv_kernel == "cuda" and self.head_dim != 64:
            raise ValueError(
                "The pinned RWKV-7 CUDA kernel currently requires head_dim=64."
            )
        if self.llama_rope_theta <= 0:
            raise ValueError("llama_rope_theta must be greater than zero.")
        if self.llama_initializer_range <= 0:
            raise ValueError("llama_initializer_range must be greater than zero.")
        if not self.match3_shared_input_features:
            raise ValueError(
                "Experiment 0 requires shared Match-3 tuple/CoT input features. "
                "The pre-repair separate tuple/token embedding protocol is not "
                "supported."
            )


@dataclass
class TrainConfig:
    """Configuration for training, optimization, and data loading."""

    seed: int = 42
    num_workers: int = 0
    val_num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    lr_schedule: str = "linear_warmup_decay"
    warmup_fraction: float = 0.05
    precision: str = "fp32"
    fused_adamw: bool = False
    mixture: str = "parallel_cot_filler"
    parallel_ratio: float = 0.5
    filler_ratio: float = 0.5
    serial_ratio: float = 0.0
    immediate_ratio: float = 0.0

    def __post_init__(self):
        if self.num_workers < 0 or self.val_num_workers < 0:
            raise ValueError("DataLoader worker counts must be non-negative.")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be greater than zero.")
        if not 0.0 < self.adam_beta1 < 1.0 or not 0.0 < self.adam_beta2 < 1.0:
            raise ValueError("Adam beta values must be strictly between zero and one.")
        if self.lr_schedule not in {"constant", "linear_warmup_decay"}:
            raise ValueError(
                "lr_schedule must be one of: constant, linear_warmup_decay; "
                f"got {self.lr_schedule!r}"
            )
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1).")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(
                "precision must be one of: fp32, bf16, fp16; "
                f"got {self.precision!r}"
            )
