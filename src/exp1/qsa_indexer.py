"""Remove the data-dependent synchronization from the Qwen4-Exp QSA indexer.

Upstream ``Qwen4ExpTextQSAIndexer.forward`` calls ``torch.nonzero`` once per
``(batch, query)`` pair to find which keys a query can see. ``torch.nonzero``
has a data-dependent output shape, so it is itself a host-device
synchronization: at the registered D=1 shape that is 64 x 18 = 1152
synchronizations per forward, measured against 9 for the all-GDN variant, which
has no indexer.

Under a plain unpadded causal mask with no cache the answer is known without
looking: query ``q`` sees exactly ``[0..q]``. This module supplies a forward
that is upstream's, with the ``torch.nonzero`` call replaced by a prefix of a
single ``torch.arange`` and the block count derived from the Python query
index. **Nothing else changes.** The per-query pooling, normalization, RoPE,
``torch.matmul``, one-dimensional ``topk``, tail concatenation, dtypes and
final scatter are upstream's own operations on upstream's own shapes, so the
selected mask is exact on every platform rather than on the one it was
developed on.

That constraint is deliberate and was learned the hard way. An earlier version
of this module computed every query at once with a batched ``einsum`` and a
batched ``topk``. It was exact on sm_89/Windows across 1600 cases and produced
eleven exact-equality failures on Python 3.11/Linux at batch 64: changing the
reduction shape changes the floating-point result, and a batched ``topk`` does
not promise the tie order a one-dimensional ``topk`` gives. The chosen mask is
discrete, so a tolerance is not available as a remedy. Removing the
synchronization is worth less than batching the whole loop and is portable;
batching is not.

A predicate admits this path only for inputs it can prove upstream treats
identically. It is checked before any projection or cache update, and a false
predicate calls the preserved upstream method exactly once. Nothing here is
installed globally: ``install_causal_qsa_indexer`` binds the replacement to the
indexer instances of one model.
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
    """Whether ``[0..q]`` is provably the visible set for every ``(b, q)``.

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


def causal_qsa_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values):
    """Instance ``forward`` for ``Qwen4ExpTextQSAIndexer``.

    Falls back to the preserved upstream method, exactly once, whenever the
    predicate does not hold. Failures inside this path are not caught: an
    implementation bug must stay visible rather than degrade into a silent slow
    path.
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
    q, token_k = torch.split(
        qk,
        [self.index_n_heads * self.index_head_dim,
         self.index_kv_heads * self.index_head_dim],
        dim=-1,
    )
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = self.q_layernorm(q)
    q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

    selected_token_indices = torch.full(
        (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )

    # The only departure from upstream: one arange, whose prefix through
    # query_idx + 1 is what torch.nonzero would have returned for a full-causal
    # row, and a block count computed from the Python index rather than from a
    # device-side shape. Everything below is upstream's, unchanged.
    positions = torch.arange(seq_length, device=hidden_states.device)

    for batch_idx in range(batch_size):
        for query_idx in range(seq_length):
            local_visible_indices = positions[: query_idx + 1]
            num_complete_blocks = (query_idx + 1) // self.compress_ratio
            # Compute selected tokens
            if num_complete_blocks > 0:
                block_token_indices = local_visible_indices[
                    : num_complete_blocks * self.compress_ratio
                ].view(num_complete_blocks, self.compress_ratio)

                key_groups = raw_keys[batch_idx].index_select(
                    0, block_token_indices.flatten())
                key_groups = key_groups.view(*block_token_indices.shape,
                                             self.index_head_dim)
                pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                pooled_keys = self.k_layernorm(pooled_keys)
                group_starts = block_token_indices[:, 0]
                block_key_states = apply_rotary_pos_emb(
                    pooled_keys.unsqueeze(1),
                    cos=full_cos[batch_idx].index_select(0, group_starts),
                    sin=full_sin[batch_idx].index_select(0, group_starts),
                ).squeeze(1)

                scores = torch.matmul(
                    q[batch_idx, query_idx].float(),
                    block_key_states.float().transpose(-1, -2),
                ).transpose(-1, -2)
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(
                    self.index_head_dim)

                selected_block_indices = scores.topk(
                    min(self.block_topk, num_complete_blocks), dim=0).indices
                # Remap the indices of the blocks to the indices of individual tokens
                selected_tokens = block_token_indices.index_select(
                    0, selected_block_indices).flatten()
            else:
                selected_tokens = torch.tensor([], device=hidden_states.device)
            tail = local_visible_indices[num_complete_blocks * self.compress_ratio:]
            selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
            selected_token_indices[
                batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

    # Create the additive mask to be added to the main causal mask
    kv_length = attention_mask.shape[-1]
    selected_token_mask = torch.zeros(
        (*selected_token_indices.shape[:-1], kv_length + 1),
        device=attention_mask.device, dtype=torch.bool,
    )
    # We absorb all the -1 by scattering them to the last index that we will drop
    scatter_indices = torch.where(selected_token_indices >= 0,
                                  selected_token_indices, kv_length)
    selected_token_mask = selected_token_mask.scatter(
        -1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
    # if using eager, convert to float mask
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(
            selected_token_mask, attention_mask.new_zeros(()), min_dtype)

    return selected_token_mask


def install_causal_qsa_indexer(model: torch.nn.Module) -> int:
    """Bind the no-nonzero forward to this model's indexer instances only.

    Binds to instances rather than patching the class, so no other model in the
    process is affected. Only ``forward`` is rebound: parameters, buffers and
    state-dict keys are untouched.
    """
    installed = 0
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextQSAIndexer):
            module.forward = types.MethodType(causal_qsa_forward, module)
            installed += 1
    return installed
