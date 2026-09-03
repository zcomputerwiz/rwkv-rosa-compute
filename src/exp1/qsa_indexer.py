"""A batched replacement for the Qwen4-Exp QSA indexer's per-query loop.

Upstream ``Qwen4ExpTextQSAIndexer.forward`` loops over every ``(batch, query)``
pair and calls ``torch.nonzero`` once per pair. ``torch.nonzero`` has a
data-dependent output shape, so each call is a host-device synchronization: at
the registered D=1 shape that is 64 x 18 = 1152 synchronizations per forward,
measured against 9 for the all-GDN variant which has no indexer.

The loop is only necessary if the visible set is arbitrary. Under a plain
unpadded causal mask with no cache it is not: query ``q`` sees exactly
``[0..q]``, so block membership and the incomplete tail are closed-form in
``q``, and every query can be computed at once.

This module supplies that batched path and a predicate that admits it only for
inputs it can prove upstream would treat identically. The predicate is checked
before any projection or cache update, and a false predicate calls the
preserved upstream method exactly once. Nothing here is installed globally:
``install_batched_qsa_indexer`` binds the replacement to the indexer instances
of one model.
"""

import math
import types

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextQSAIndexer,
    apply_rotary_pos_emb,
)

# Captured at import, before anything is installed, so a fallback can never
# recurse into the replacement.
_UPSTREAM_FORWARD = Qwen4ExpTextQSAIndexer.forward


def _visible(attention_mask: torch.Tensor) -> torch.Tensor:
    """Upstream's own reading of the mask: True where a key is visible."""
    if attention_mask.dtype == torch.bool:
        return attention_mask
    return attention_mask == 0


def optimized_path_applies(indexer, hidden_states, attention_mask,
                           past_key_values) -> bool:
    """Whether the batched path provably reproduces upstream for this input.

    Reads no parameters and updates no state, so it is safe to evaluate before
    the projections and before ``update_indexer``.
    """
    if past_key_values is not None:
        return False
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 4:
        return False

    batch_size, seq_length, _ = hidden_states.shape
    if attention_mask.shape[0] != batch_size or attention_mask.shape[1] != 1:
        return False
    # The no-cache path has key length equal to query length. Anything else is
    # a cached or cross-length layout the closed form does not describe.
    if attention_mask.shape[-2] != seq_length or attention_mask.shape[-1] != seq_length:
        return False

    ratio = indexer.compress_ratio
    # block_topk * ratio must exhaust the budget, or the output width
    # (budget + ratio - 1) is not the bound the batched writer assumes.
    if ratio < 1 or indexer.token_budget % ratio != 0:
        return False
    if indexer.block_topk != indexer.token_budget // ratio:
        return False

    expected = torch.tril(
        torch.ones(seq_length, seq_length, dtype=torch.bool,
                   device=attention_mask.device)
    )
    # One device-side reduction, against the seq_length * batch_size
    # synchronizations the upstream loop performs.
    return bool((_visible(attention_mask)[:, 0] == expected).all())


def batched_qsa_forward(self, hidden_states, position_embeddings, attention_mask,
                        past_key_values):
    """Instance ``forward`` for ``Qwen4ExpTextQSAIndexer``.

    Falls back to the preserved upstream method, exactly once, whenever the
    predicate does not hold. Failures inside the batched path are not caught:
    an implementation bug must stay visible rather than degrade into a silent
    slow path.
    """
    if not optimized_path_applies(self, hidden_states, attention_mask,
                                  past_key_values):
        return _UPSTREAM_FORWARD(self, hidden_states, position_embeddings,
                                 attention_mask, past_key_values)

    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

    qk = self.index_qk_proj(hidden_states)
    query, token_k = torch.split(
        qk,
        [self.index_n_heads * self.index_head_dim,
         self.index_kv_heads * self.index_head_dim],
        dim=-1,
    )
    query = self.q_layernorm(query.reshape(*hidden_shape))
    query = apply_rotary_pos_emb(query, cos=current_cos, sin=current_sin,
                                 unsqueeze_dim=2)
    raw_keys = token_k.reshape(*hidden_shape).squeeze(2)

    ratio = self.compress_ratio
    device = hidden_states.device
    n_blocks = seq_length // ratio

    # Every complete block pooled once, rather than once per (batch, query).
    block_tokens = torch.arange(n_blocks * ratio, device=device).view(n_blocks, ratio)
    pooled = raw_keys[:, : n_blocks * ratio].view(
        batch_size, n_blocks, ratio, self.index_head_dim
    )
    pooled = self.k_layernorm(pooled.float().mean(dim=2).to(raw_keys.dtype))
    block_keys = apply_rotary_pos_emb(
        pooled.unsqueeze(1),
        cos=full_cos[:, block_tokens[:, 0]],
        sin=full_sin[:, block_tokens[:, 0]],
    ).squeeze(1)

    # scores[b, q, k] = sum_h relu(query[b, q, h] . block_keys[b, k]) / sqrt(d),
    # which is upstream's per-query matmul, relu, head-sum and scale.
    scores = torch.einsum("bqhd,bkd->bqkh", query.float(), block_keys.float())
    scores = torch.relu(scores).sum(-1) / math.sqrt(self.index_head_dim)

    # Query q has (q + 1) // ratio complete blocks; later blocks are invisible
    # to it and upstream never offers them to topk.
    n_complete = (torch.arange(seq_length, device=device) + 1) // ratio
    available = (torch.arange(n_blocks, device=device).unsqueeze(0)
                 < n_complete.unsqueeze(1))
    masked = scores.masked_fill(~available.unsqueeze(0), float("-inf"))

    top = min(self.block_topk, n_blocks)
    order = masked.topk(top, dim=-1).indices if top else masked.new_zeros(
        (batch_size, seq_length, 0), dtype=torch.long
    )
    n_taken = n_complete.clamp(max=self.block_topk)
    taken = torch.arange(top, device=device).unsqueeze(0) < n_taken.unsqueeze(1)

    selected = block_tokens[order] if top else order.new_zeros(
        (batch_size, seq_length, 0, ratio)
    )
    selected = selected.masked_fill(~taken.unsqueeze(0).unsqueeze(-1), -1)
    selected = selected.reshape(batch_size, seq_length, top * ratio)

    width = self.token_budget + ratio - 1
    slot = torch.arange(width, device=device)
    indices = torch.full((batch_size, seq_length, width), -1, dtype=torch.long,
                         device=device)
    indices[..., : top * ratio] = selected

    # The tail is the trailing incomplete block: tokens [n_complete*ratio .. q].
    n_selected = n_taken * ratio
    tail_offset = slot.unsqueeze(0) - n_selected.unsqueeze(1)
    tail_length = (torch.arange(seq_length, device=device) + 1) - n_complete * ratio
    in_tail = (tail_offset >= 0) & (tail_offset < tail_length.unsqueeze(1))
    indices = torch.where(
        in_tail.unsqueeze(0),
        ((n_complete * ratio).unsqueeze(1) + tail_offset).unsqueeze(0),
        indices,
    )
    # Slots after the selected blocks that the tail does not reach stay unset.
    beyond = (slot.unsqueeze(0) >= n_selected.unsqueeze(1)) & ~in_tail
    indices = indices.masked_fill(beyond.unsqueeze(0), -1)

    # Upstream's mask construction, unchanged. It scatters into a boolean, so
    # only the selected set is observable and the order topk returned is not.
    kv_length = attention_mask.shape[-1]
    selected_token_mask = torch.zeros(
        (*indices.shape[:-1], kv_length + 1), device=attention_mask.device,
        dtype=torch.bool,
    )
    scatter_indices = torch.where(indices >= 0, indices, kv_length)
    selected_token_mask = selected_token_mask.scatter(
        -1, scatter_indices, True
    )[..., :kv_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(
            selected_token_mask, attention_mask.new_zeros(()), min_dtype
        )
    return selected_token_mask


def install_batched_qsa_indexer(model: torch.nn.Module) -> int:
    """Bind the batched forward to this model's indexer instances only.

    Binds to instances rather than patching the class, so no other model in the
    process is affected. Only ``forward`` is rebound: parameters, buffers and
    state-dict keys are untouched.
    """
    installed = 0
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextQSAIndexer):
            module.forward = types.MethodType(batched_qsa_forward, module)
            installed += 1
    return installed
