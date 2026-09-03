"""QSA index selection without the per-query ``torch.nonzero``.

Upstream ``Qwen4ExpTextQSAIndexer.forward`` calls ``torch.nonzero`` once per
``(batch, query)`` pair to find which keys a query can see. ``torch.nonzero``
has a data-dependent output shape, so it is itself a host-device
synchronization: at the registered D=1 shape that is 64 x 18 = 1152
synchronizations per forward, against 9 for the all-GDN variant, which has no
indexer.

Under a plain unpadded causal mask with no cache the answer is known without
looking: query ``q`` sees exactly ``[0..q]``. Two implementations use that.

``causal-exact`` keeps upstream's loop and replaces only the ``nonzero`` call
with a prefix of one ``torch.arange``. Every other operation, shape and dtype is
upstream's, so its selected mask is exact on every platform by construction. It
is the default, the oracle for the other mode, and the fallback for inputs the
contract does not cover. It removes the synchronizations but keeps roughly
59,000 kernel launches per forward, and measured 1.17x.

``batched-stable-v1`` computes every query at once. It is a **new apparatus,
not upstream reproduced**: batching changes the floating-point reduction order,
so scores that are near-ties upstream can rank differently here, and a
different block can be selected. That is why it is versioned, why it is opt-in,
and why it carries its own model identity. An earlier revision of this module
claimed the batched path equalled upstream; it was exact on sm_89/Windows over
1600 cases and produced eleven exact-equality failures on Python 3.11/Linux at
batch 64. The claim was withdrawn, not the code.

Its tie rule is defined here rather than inherited: blocks are ranked by
descending score with a **stable** ordering, so equal scores select lower
original block indices first. ``torch.topk`` does not specify a tie order and
must not be relied on for one.

Neither registered forward performs a device-to-host read. The mask contract is
established at the wrapper boundary -- ``Qwen4ExpBackbone.forward`` accepts only
``inputs_embeds`` and its config sets ``use_cache=False`` -- rather than
re-proved from CUDA on every forward, because that costs one synchronization per
step and prevents CUDA Graph capture. ``guarded_qsa_forward`` keeps the content
check for generic direct calls.
"""

import math
import types

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextQSAIndexer,
    apply_rotary_pos_emb,
)

from exp1.qwen4_micro import (
    DEFAULT_QSA_IMPLEMENTATION,
    QSA_IMPLEMENTATIONS,
)

# Captured at import, before anything is installed, so a fallback can never
# recurse into a replacement.
_UPSTREAM_FORWARD = Qwen4ExpTextQSAIndexer.forward


def structural_contract_holds(indexer, hidden_states, attention_mask,
                              past_key_values) -> bool:
    """Host-side checks only: no tensor contents are read, so no sync.

    Shapes, dtypes and ``is None`` live on the host already. What this cannot
    check is whether the mask's *contents* are full-causal; that is the
    wrapper-boundary contract, not a per-forward proof.
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
    return indexer.block_topk == indexer.token_budget // ratio


def mask_is_full_causal(attention_mask: torch.Tensor, seq_length: int) -> bool:
    """Read the mask's contents and confirm every row is ``[0..q]``.

    **One device-to-host synchronization.** Deliberately not called by either
    registered forward; used by ``guarded_qsa_forward`` and by tests.
    """
    visible = (attention_mask if attention_mask.dtype == torch.bool
               else attention_mask == 0)
    expected = torch.tril(
        torch.ones(seq_length, seq_length, dtype=torch.bool,
                   device=attention_mask.device)
    )
    return bool((visible[:, 0] == expected).all())


def _projected(indexer, hidden_states, position_embeddings):
    """Upstream's projection, layernorm and query RoPE, shared by both modes."""
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, -1, indexer.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos = full_cos[:, -seq_length:, :]
    current_sin = full_sin[:, -seq_length:, :]

    qk = indexer.index_qk_proj(hidden_states)
    query, token_k = torch.split(
        qk,
        [indexer.index_n_heads * indexer.index_head_dim,
         indexer.index_kv_heads * indexer.index_head_dim],
        dim=-1,
    )
    query = indexer.q_layernorm(query.reshape(*hidden_shape))
    query = apply_rotary_pos_emb(query, cos=current_cos, sin=current_sin,
                                 unsqueeze_dim=2)
    raw_keys = token_k.reshape(*hidden_shape).squeeze(2)
    return query, raw_keys, full_cos, full_sin


def _selected_mask(indices, attention_mask):
    """Upstream's mask construction, unchanged, shared by both modes."""
    kv_length = attention_mask.shape[-1]
    selected_token_mask = torch.zeros(
        (*indices.shape[:-1], kv_length + 1),
        device=attention_mask.device, dtype=torch.bool,
    )
    scatter_indices = torch.where(indices >= 0, indices, kv_length)
    selected_token_mask = selected_token_mask.scatter(
        -1, scatter_indices.long(), True)[..., :kv_length].unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(
            selected_token_mask, attention_mask.new_zeros(()), min_dtype)
    return selected_token_mask


def causal_qsa_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values):
    """``causal-exact``: upstream's loop with the ``nonzero`` call removed.

    Falls back to the preserved upstream method, exactly once, when the
    structural contract does not hold. Failures inside this path are not
    caught: an implementation bug must stay visible rather than degrade into a
    silent slow path.
    """
    if not structural_contract_holds(self, hidden_states, attention_mask,
                                     past_key_values):
        return _UPSTREAM_FORWARD(self, hidden_states, position_embeddings,
                                 attention_mask, past_key_values)

    batch_size, seq_length, _ = hidden_states.shape
    q, raw_keys, full_cos, full_sin = _projected(self, hidden_states,
                                                 position_embeddings)

    selected_token_indices = torch.full(
        (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
        -1, dtype=torch.int32, device=hidden_states.device,
    )
    # The only departure from upstream: one arange, whose prefix through
    # query_idx + 1 is what torch.nonzero would have returned for a full-causal
    # row, and a block count from the Python index rather than a device shape.
    positions = torch.arange(seq_length, device=hidden_states.device)

    for batch_idx in range(batch_size):
        for query_idx in range(seq_length):
            local_visible_indices = positions[: query_idx + 1]
            num_complete_blocks = (query_idx + 1) // self.compress_ratio
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
                selected_tokens = block_token_indices.index_select(
                    0, selected_block_indices).flatten()
            else:
                selected_tokens = torch.tensor([], device=hidden_states.device)
            tail = local_visible_indices[num_complete_blocks * self.compress_ratio:]
            selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
            selected_token_indices[
                batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

    return _selected_mask(selected_token_indices, attention_mask)


def batched_stable_qsa_forward(self, hidden_states, position_embeddings,
                               attention_mask, past_key_values):
    """``batched-stable-v1``: every query at once, with a defined tie rule.

    Semantically the same selector as upstream -- pool complete aligned blocks,
    score them against the query, keep the top ``block_topk``, append the
    incomplete causal tail -- but batched, so the floating-point reduction order
    differs and near-ties can rank differently. Not upstream-exact, by design.
    """
    if not structural_contract_holds(self, hidden_states, attention_mask,
                                     past_key_values):
        return _UPSTREAM_FORWARD(self, hidden_states, position_embeddings,
                                 attention_mask, past_key_values)

    batch_size, seq_length, _ = hidden_states.shape
    q, raw_keys, full_cos, full_sin = _projected(self, hidden_states,
                                                 position_embeddings)

    ratio = self.compress_ratio
    device = hidden_states.device
    n_blocks = seq_length // ratio

    # Every complete block pooled once, rather than once per (batch, query).
    block_tokens = torch.arange(n_blocks * ratio, device=device).view(n_blocks, ratio)
    pooled = raw_keys[:, : n_blocks * ratio].view(
        batch_size, n_blocks, ratio, self.index_head_dim)
    pooled = self.k_layernorm(pooled.float().mean(dim=2).to(raw_keys.dtype))
    block_keys = apply_rotary_pos_emb(
        pooled.unsqueeze(1),
        cos=full_cos[:, block_tokens[:, 0]],
        sin=full_sin[:, block_tokens[:, 0]],
    ).squeeze(1)

    # scores[b, q, k] = sum_h relu(q[b, q, h] . block_keys[b, k]) / sqrt(d)
    scores = torch.einsum("bqhd,bkd->bqkh", q.float(), block_keys.float())
    scores = torch.relu(scores).sum(-1) / math.sqrt(self.index_head_dim)

    # Query q has (q + 1) // ratio complete blocks; later blocks are invisible
    # to it and upstream never offers them for selection.
    n_complete = (torch.arange(seq_length, device=device) + 1) // ratio
    available = (torch.arange(n_blocks, device=device).unsqueeze(0)
                 < n_complete.unsqueeze(1))
    masked = scores.masked_fill(~available.unsqueeze(0), float("-inf"))

    top = min(self.block_topk, n_blocks)
    # Stable descending sort, not topk: torch.topk leaves its tie order
    # unspecified, and this mode promises that equal scores select the lower
    # original block index. -inf entries sort last, so the first n_complete
    # positions are exactly the available blocks.
    order = torch.argsort(masked, dim=-1, descending=True,
                          stable=True)[..., :top]
    n_taken = n_complete.clamp(max=self.block_topk)
    taken = torch.arange(top, device=device).unsqueeze(0) < n_taken.unsqueeze(1)

    selected = block_tokens[order]
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
    beyond = (slot.unsqueeze(0) >= n_selected.unsqueeze(1)) & ~in_tail
    indices = indices.masked_fill(beyond.unsqueeze(0), -1)

    return _selected_mask(indices, attention_mask)


def guarded_qsa_forward(self, hidden_states, position_embeddings, attention_mask,
                        past_key_values):
    """``causal-exact`` for a generic caller, with the mask contents proved.

    Costs one synchronization per call, which is why the registered path does
    not use it. Provided so a direct caller outside ``Qwen4ExpBackbone`` -- with
    padding, a sliding window, or a cache -- still gets a correct answer.
    """
    if not structural_contract_holds(self, hidden_states, attention_mask,
                                     past_key_values):
        return _UPSTREAM_FORWARD(self, hidden_states, position_embeddings,
                                 attention_mask, past_key_values)
    if not mask_is_full_causal(attention_mask, hidden_states.shape[1]):
        return _UPSTREAM_FORWARD(self, hidden_states, position_embeddings,
                                 attention_mask, past_key_values)
    return causal_qsa_forward(self, hidden_states, position_embeddings,
                              attention_mask, past_key_values)


_FORWARDS = {
    "causal-exact": causal_qsa_forward,
    "batched-stable-v1": batched_stable_qsa_forward,
}


def install_qsa_indexer(model: torch.nn.Module,
                        implementation: str = DEFAULT_QSA_IMPLEMENTATION) -> int:
    """Bind the chosen forward to this model's indexer instances only.

    Binds to instances rather than patching the class, so no other model in the
    process is affected. Only ``forward`` is rebound: parameters, buffers and
    state-dict keys are untouched.
    """
    if implementation not in _FORWARDS:
        raise ValueError(
            f"qsa_implementation must be one of {QSA_IMPLEMENTATIONS}; "
            f"got {implementation!r}")
    forward = _FORWARDS[implementation]
    installed = 0
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextQSAIndexer):
            module.forward = types.MethodType(forward, module)
            installed += 1
    return installed
