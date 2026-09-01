"""Official Qwen4-Exp backbone adapted to the Experiment 1 task seam."""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn

from exp0.models.base import InputEmbedWrapper

TRANSFORMERS_VERSION = "5.16.0"
QWEN4_VARIANTS = ("hybrid", "all-gdn")
QWEN4_MAX_POSITION_EMBEDDINGS = 128


@dataclass(frozen=True)
class Qwen4MicroConfig:
    """The registered Qwen4-Exp micro architecture for the first pilot."""

    vocab_size: int
    hidden_size: int = 128
    num_hidden_layers: int = 4
    variant: str = "hybrid"
    architecture: str = field(default="qwen4_exp", init=False)

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one.")
        if self.hidden_size != 128 or self.num_hidden_layers != 4:
            raise ValueError(
                "The registered Qwen4-Exp pilot requires hidden_size=128 and "
                "num_hidden_layers=4."
            )
        if self.variant not in QWEN4_VARIANTS:
            raise ValueError(
                f"variant must be one of {QWEN4_VARIANTS}; got {self.variant!r}"
            )

    @property
    def layer_types(self) -> tuple[str, ...]:
        if self.variant == "all-gdn":
            return ("linear_attention",) * 4
        return (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "qwen_sparse_attention",
        )

    def resolved(self) -> Dict[str, Any]:
        """Return the complete JSON-safe architecture identity."""
        return {
            "architecture": self.architecture,
            "transformers_version": TRANSFORMERS_VERSION,
            "variant": self.variant,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "num_experts": 4,
            "num_experts_per_tok": 1,
            "hc_count": 4,
            "hc_lowrank": 32,
            "layer_types": list(self.layer_types),
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 32,
            "indexer_budget": 8,
            "indexer_compress_ratio": 2,
            "ple_layer_ids": [],
            "max_position_embeddings": QWEN4_MAX_POSITION_EMBEDDINGS,
            "use_cache": False,
        }


class Qwen4ExpBackbone(nn.Module):
    """Convert ``Qwen4ExpTextModel`` output to the tensor backbone contract."""

    def __init__(self, config: Qwen4MicroConfig):
        super().__init__()
        import transformers
        from transformers import Qwen4ExpTextConfig, Qwen4ExpTextModel

        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(
                "Qwen4-Exp execution identity requires transformers=="
                f"{TRANSFORMERS_VERSION}; found {transformers.__version__}."
            )

        values = config.resolved()
        values.pop("architecture")
        values.pop("transformers_version")
        values.pop("variant")
        self.model = Qwen4ExpTextModel(Qwen4ExpTextConfig(**values))

    def forward(self, *, inputs_embeds: torch.Tensor) -> torch.Tensor:
        output = self.model(inputs_embeds=inputs_embeds, return_dict=True)
        return output.last_hidden_state


def create_qwen4_micro_model(
    config: Qwen4MicroConfig,
    *,
    d_input: int,
) -> InputEmbedWrapper:
    """Build the official Qwen4-Exp micro backbone with the existing task head."""
    return InputEmbedWrapper(
        backbone=Qwen4ExpBackbone(config),
        d_input=d_input,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
    )
