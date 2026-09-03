"""Official Qwen4-Exp backbone adapted to the Experiment 1 task seam."""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn

from exp0.models.base import InputEmbedWrapper

TRANSFORMERS_VERSION = "5.16.0"
QWEN4_VARIANTS = ("hybrid", "all-gdn")
# Which QSA index selector a run installs. "causal-exact" reproduces
# upstream's masks exactly; "batched-stable-v1" is a separate apparatus
# whose batching can change near-tie block choices. Defined here rather
# than in exp1.qsa_indexer so that reading the identity never imports
# Transformers ahead of the version guard in Qwen4ExpBackbone.
QSA_IMPLEMENTATIONS = ("causal-exact", "batched-stable-v1")
DEFAULT_QSA_IMPLEMENTATION = "causal-exact"
# Which chunk size the Gated DeltaNet chunk rule uses. "fixed-64" is
# upstream's own default and is numerically unchanged. "min-sequence-64-v1"
# passes chunk_size=min(64, T): identical to fixed-64 for T >= 64, and for
# shorter T it removes the padding upstream's reference loop pays for --
# measured to change gradients by O(1e-8) on CUDA, so it is a separate
# numerics-affecting identity, not a transparent optimization.
GDN_CHUNK_POLICIES = ("fixed-64", "min-sequence-64-v1")
DEFAULT_GDN_CHUNK_POLICY = "fixed-64"
QWEN4_MAX_POSITION_EMBEDDINGS = 128


@dataclass(frozen=True)
class Qwen4MicroConfig:
    """The registered Qwen4-Exp micro architecture for the first pilot."""

    vocab_size: int
    hidden_size: int = 128
    num_hidden_layers: int = 4
    variant: str = "hybrid"
    # Which QSA index selector to install. A repository-level choice, not a
    # Transformers config field, so it is never passed to Qwen4ExpTextConfig.
    qsa_implementation: str = DEFAULT_QSA_IMPLEMENTATION
    # Which GDN chunk-size policy to use. Also repository-level: passed as a
    # per-forward keyword argument to the pinned model call, never into
    # Qwen4ExpTextConfig.
    gdn_chunk_policy: str = DEFAULT_GDN_CHUNK_POLICY
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
        if self.qsa_implementation not in QSA_IMPLEMENTATIONS:
            raise ValueError(
                f"qsa_implementation must be one of {QSA_IMPLEMENTATIONS}; "
                f"got {self.qsa_implementation!r}"
            )
        if self.gdn_chunk_policy not in GDN_CHUNK_POLICIES:
            raise ValueError(
                f"gdn_chunk_policy must be one of {GDN_CHUNK_POLICIES}; "
                f"got {self.gdn_chunk_policy!r}"
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
        """Return the complete JSON-safe architecture identity.

        ``qsa_implementation`` and ``gdn_chunk_policy`` each appear only when
        they are not their default. Both defaults reproduce upstream's own
        numerics exactly, so omitting them keeps this identity byte-identical
        to the one already written into existing checkpoints: they stay
        loadable, and they can only resume under the same, matching,
        non-default selection. A non-default value adds its key, which makes
        the identity differ and the resume signature reject a mismatch.
        """
        identity: Dict[str, Any] = {
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
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "linear_conv_kernel_dim": 4,
            "output_router_logits": False,
            "router_aux_loss_coef": 0.001,
            "norm_topk_prob": True,
            "tie_word_embeddings": False,
            "use_cache": False,
        }
        if self.qsa_implementation != DEFAULT_QSA_IMPLEMENTATION:
            identity["qsa_implementation"] = self.qsa_implementation
        if self.gdn_chunk_policy != DEFAULT_GDN_CHUNK_POLICY:
            identity["gdn_chunk_policy"] = self.gdn_chunk_policy
        return identity


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
        # Repository-level selectors, not Transformers config fields.
        values.pop("qsa_implementation", None)
        values.pop("gdn_chunk_policy", None)

        # The registered QSA forwards read no tensor contents, so they cannot
        # verify the mask per step. They rely on this boundary instead: this
        # wrapper's forward supplies only inputs_embeds and never an external
        # mask, and use_cache is off, so Transformers builds a plain unpadded
        # full-causal mask with key length equal to query length. Asserted
        # here so a future config change trips rather than silently voiding
        # the contract the indexer depends on.
        if values["use_cache"] is not False:
            raise RuntimeError(
                "The registered Qwen4-Exp identity requires use_cache=False; "
                "the QSA index selectors assume the no-cache full-causal mask."
            )

        if config.gdn_chunk_policy != DEFAULT_GDN_CHUNK_POLICY:
            # A non-default policy passes chunk_size as a per-forward keyword,
            # which reaches the GDN chunk rule through **kwargs. That argument
            # is only honoured by the pinned Torch fallback: the hub-kernel
            # wrapper filters kwargs down to whatever the *installed*
            # implementation accepts, and an installed `fla` kernel is not
            # audited to accept -- or obey -- chunk_size. Silently ignoring it
            # would make this policy a no-op rather than a numerics change, so
            # it fails closed instead.
            import importlib.util

            if importlib.util.find_spec("fla") is not None:
                raise RuntimeError(
                    f"gdn_chunk_policy={config.gdn_chunk_policy!r} requires the "
                    "Torch fallback chunk_gated_delta_rule implementation, "
                    "whose signature this policy is measured against. The "
                    "'fla' package is importable in this environment, so the "
                    "hub-kernel wrapper may install its kernel instead, which "
                    "is not known to accept or honour chunk_size. Uninstall "
                    "'fla' or use the default gdn_chunk_policy."
                )

        # Imported here, not at module scope, so the version guard above runs
        # before anything reaches into the Transformers internals this binds to.
        from exp1.qsa_indexer import install_qsa_indexer

        self.model = Qwen4ExpTextModel(Qwen4ExpTextConfig(**values))
        self.model.embed_tokens.weight.requires_grad_(False)
        install_qsa_indexer(self.model, config.qsa_implementation)
        self._gdn_chunk_policy = config.gdn_chunk_policy

    def forward(self, *, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if self._gdn_chunk_policy == "min-sequence-64-v1":
            # Identical to fixed-64 for T >= 64 -- min(64, T) is then 64, the
            # same value the default call leaves implicit. Only T < 64 changes
            # behaviour, by removing the padding upstream's chunk rule pays
            # for. Computed per call, not cached, because T can vary between
            # the train and eval banks.
            chunk_size = min(64, inputs_embeds.shape[1])
            output = self.model(inputs_embeds=inputs_embeds, chunk_size=chunk_size)
        else:
            output = self.model(inputs_embeds=inputs_embeds)
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
