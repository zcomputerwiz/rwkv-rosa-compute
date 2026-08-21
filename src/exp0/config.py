"""Configuration dataclasses for Experiment 0."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    include_eos_target: bool = True
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
        if not self.include_eos_target:
            raise ValueError(
                "Experiment 0 source-fidelity protocol requires a supervised "
                "EOS target after the final True/False token."
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
    output_vocab_size: Optional[int] = None
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
        if self.output_vocab_size is not None and self.output_vocab_size <= 0:
            raise ValueError("output_vocab_size must be greater than zero when set.")
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


# Supported early-stopping targets, mapped to the direction of improvement.
# "filler_accuracy" is validation answer accuracy; its theoretical target is 1.0.
# "cot_result_nll" is the teacher-forced result-slot NLL; its theoretical target
# is the measured cot_result_nll_floor, the irreducible uncertainty that
# randomized coordinate selection imposes. A model at the floor is computing the
# result, not guessing it, so the floor is the correct stop target rather than 0.
EARLY_STOP_METRICS = {
    "none": None,
    "filler_accuracy": "max",
    "cot_result_nll": "min",
}

EARLY_STOP_FIELDS = (
    "early_stop_metric",
    "early_stop_target",
    "early_stop_tolerance",
    "early_stop_patience",
)


# DataLoader plumbing is execution, not protocol: it cannot change what a run
# computes, so it must not change run identity.
#
# Task3SumDataset.__getitem__ derives every item from (seed, idx) alone - the
# format code is precomputed per index and the per-item RNG is seeded
# random.Random(f"{seed}_{idx}") - so an index yields the same example whichever
# worker builds it, and DataLoader preserves batch order for a fixed sampler.
#
# Normalized to a canonical value rather than popped: popping would change the
# key set of the canonical config, whereas substituting a fixed value keeps the
# shape identical. Either way the run_id of a run that TUNED these fields does
# change - see CHECKPOINT_TOLERATED_FIELDS in exp0.checkpointing for how
# checkpoints written before this became neutral are still accepted.
DATALOADER_NEUTRAL_FIELDS: Dict[str, Any] = {
    "num_workers": 0,
    "val_num_workers": 0,
    "pin_memory": True,
    "prefetch_factor": 2,
}


def drop_identity_neutral_fields(train_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip or normalize options that do not change what a run computes.

    A feature left at its default must not change the fingerprint of runs that
    do not use it, otherwise adding the option invalidates every existing run_id
    and every checkpoint written before it. A run with early stopping off and
    the immediate-answer protocol enabled behaves exactly as it did before
    either option existed, so its identity must match too.

    DataLoader settings are normalized for the same reason: see
    DATALOADER_NEUTRAL_FIELDS.
    """
    if train_dict.get("early_stop_metric", "none") == "none":
        for key in EARLY_STOP_FIELDS:
            train_dict.pop(key, None)
    if train_dict.get("immediate_protocol", True):
        train_dict.pop("immediate_protocol", None)
    for key in ("tf32_matmul", "torch_compile", "grouped_execution"):
        if not train_dict.get(key, False):
            train_dict.pop(key, None)
    for key, canonical in DATALOADER_NEUTRAL_FIELDS.items():
        if key in train_dict:
            train_dict[key] = canonical
    return train_dict


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
    neutral_ratio: float = 0.0
    # Early stopping. "none" keeps the fixed-budget protocol, in which `epochs`
    # is exactly the number of epochs trained. Any other value makes `epochs` a
    # ceiling instead, which is a different experimental protocol: see
    # EARLY_STOP_METRICS for the supported targets.
    early_stop_metric: str = "none"
    early_stop_target: Optional[float] = None
    early_stop_tolerance: float = 0.0
    early_stop_patience: int = 1
    # The published immediate-answer protocol multiplies epochs and substitutes
    # its own weight decay and gradient clip whenever num_filler is 0 or the
    # mixture is "immediate". Leave this True to reproduce the source protocol.
    # Set it False to hold epochs, weight decay, and gradient clip exactly as
    # requested, which is what an N=0 arm needs to be compute-matched against an
    # N>0 arm. Doing so is a different protocol and changes the run_id.
    immediate_protocol: bool = True
    # Execution protocols, both off by default so "fp32" keeps meaning strict
    # FP32 and an uncompiled graph. Enabling either is a deliberate protocol
    # choice, is recorded in the report, and changes the run_id: TF32 lowers
    # the internal precision of FP32 matmuls, and compilation fuses and
    # reorders operations. Neither may be switched on partway through a sweep.
    tf32_matmul: bool = False
    torch_compile: bool = False
    # Length-aware execution: run one optimizer batch as length-homogeneous
    # subgroups so filler examples are not carried through a CoT-sized
    # rectangle. One optimizer update, one scheduler step, and one global
    # gradient clip are preserved, and the loss stays token-weighted, so the
    # objective is unchanged. It is opt-in and changes the run_id because
    # summation order differs, which moves logits by float32 epsilon.
    #
    # This is an implementation-efficiency option. It is NOT compute matching:
    # the scientific budget is still the requested filler transition count N.
    grouped_execution: bool = False

    def __post_init__(self):
        format_ratios = {
            "parallel_ratio": self.parallel_ratio,
            "filler_ratio": self.filler_ratio,
            "serial_ratio": self.serial_ratio,
            "immediate_ratio": self.immediate_ratio,
            "neutral_ratio": self.neutral_ratio,
        }
        for name, value in format_ratios.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative; got {value!r}")
        if self.early_stop_metric not in EARLY_STOP_METRICS:
            raise ValueError(
                "early_stop_metric must be one of: "
                f"{', '.join(sorted(EARLY_STOP_METRICS))}; "
                f"got {self.early_stop_metric!r}"
            )
        if self.early_stop_metric == "none" and self.early_stop_target is not None:
            raise ValueError(
                "early_stop_target requires early_stop_metric to be enabled."
            )
        if self.early_stop_target is not None and not math.isfinite(
            self.early_stop_target
        ):
            raise ValueError("early_stop_target must be finite when set.")
        if not math.isfinite(self.early_stop_tolerance):
            raise ValueError("early_stop_tolerance must be finite.")
        if self.early_stop_tolerance < 0.0:
            raise ValueError("early_stop_tolerance must be non-negative.")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be at least one epoch.")
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
