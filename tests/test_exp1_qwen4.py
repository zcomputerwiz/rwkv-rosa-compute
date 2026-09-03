import hashlib

import pytest
import torch

from exp0.config import TrainConfig
from exp0.train import set_seed
from exp1.dataset import PointerChaseDataset
from exp1.pointer_chase import ChaseSpec, generate_dataset
from exp1.qwen4_micro import Qwen4MicroConfig, create_qwen4_micro_model
from exp1.train import _model_signature, forward_logits, train_model
from exp1.workspace import Workspace


def _model(config: Qwen4MicroConfig, spec: ChaseSpec):
    return create_qwen4_micro_model(config, d_input=spec.d_input)


def _nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in module.parameters()
    )


def test_registered_qwen4_micro_config_is_complete_and_fixed():
    hybrid = Qwen4MicroConfig(vocab_size=16)
    recurrent = Qwen4MicroConfig(vocab_size=16, variant="all-gdn")

    assert hybrid.layer_types == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    )
    assert recurrent.layer_types == ("linear_attention",) * 4
    assert hybrid.resolved()["ple_layer_ids"] == []
    assert hybrid.resolved()["rope_parameters"] == {
        "rope_type": "default",
        "rope_theta": 10000.0,
    }
    assert hybrid.resolved()["linear_conv_kernel_dim"] == 4
    assert hybrid.resolved()["norm_topk_prob"] is True
    assert hybrid.resolved()["use_cache"] is False

    with pytest.raises(ValueError, match="registered Qwen4-Exp pilot"):
        Qwen4MicroConfig(vocab_size=16, hidden_size=64)


def test_qwen4_micro_forward_backward_reaches_every_component():
    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    set_seed(7)
    model = _model(Qwen4MicroConfig(vocab_size=16), spec)
    workspace = Workspace(128, num_slots=8, num_steps=8)
    inputs = torch.randn(2, spec.seq_len(0), spec.d_input)

    logits = forward_logits(model, inputs, workspace)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2]))
    loss.backward()

    assert logits.shape == (2, 16)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert _nonzero_gradient(model.input_proj)
    assert _nonzero_gradient(model.head)
    layers = model.backbone.model.layers
    assert _nonzero_gradient(layers[0].linear_attn)
    assert _nonzero_gradient(layers[3].self_attn)
    assert _nonzero_gradient(layers[0].mlp.experts)
    assert _nonzero_gradient(layers[0].attn_hyper_connection)
    assert _nonzero_gradient(workspace)
    assert any("ple" in name for name, _ in model.named_modules()) is False
    assert model.backbone.model.embed_tokens.weight.requires_grad is False
    assert model.backbone.model.embed_tokens.weight.grad is None

    model_parameters = sum(p.numel() for p in model.parameters())
    totals = {
        model_parameters + sum(p.numel() for p in Workspace(128, num_slots=m, num_steps=k).parameters())
        for m, k in ((1, 1), (1, 8), (8, 1), (8, 8))
    }
    assert len(totals) == 1


def test_qwen4_micro_initialization_is_seeded():
    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    config = Qwen4MicroConfig(vocab_size=16, variant="all-gdn")

    def digest(seed: int) -> str:
        set_seed(seed)
        model = _model(config, spec)
        sha = hashlib.sha256()
        for name, parameter in sorted(model.state_dict().items()):
            sha.update(name.encode())
            sha.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return sha.hexdigest()

    assert digest(41) == digest(41)
    assert digest(41) != digest(42)


def test_qwen4_signature_records_every_resolved_field():
    hybrid = Qwen4MicroConfig(vocab_size=16)
    recurrent = Qwen4MicroConfig(vocab_size=16, variant="all-gdn")

    signature = _model_signature(hybrid)
    assert signature == {"qwen4_exp": hybrid.resolved()}
    assert _model_signature(recurrent) != signature


def test_qwen4_cpu_checkpoint_resume_is_exact(tmp_path):
    spec = ChaseSpec(num_nodes=4, num_maps=2, max_depth=2)
    train_ds = PointerChaseDataset(
        generate_dataset(1, 1, depth=1, seed=1, num_nodes=4, num_maps=2),
        spec,
    )
    val_ds = PointerChaseDataset(
        generate_dataset(1, 1, depth=1, seed=2, num_nodes=4, num_maps=2),
        spec,
    )
    config = Qwen4MicroConfig(vocab_size=4, variant="all-gdn")
    train_config = TrainConfig(
        seed=42,
        batch_size=1,
        precision="fp32",
        num_workers=0,
        epochs=2,
        learning_rate=1e-3,
    )
    common = dict(
        depth=1,
        train_data_seed=1,
        val_data_seed=2,
        train_size=1,
        val_size=1,
        queries_per_memory=1,
    )
    device = torch.device("cpu")

    set_seed(42)
    full, full_history = train_model(
        _model(config, spec), train_ds, val_ds, spec, config, train_config,
        device, **common,
    )

    checkpoint_dir = tmp_path / "checkpoints"
    set_seed(42)
    train_model(
        _model(config, spec), train_ds, val_ds, spec, config, train_config,
        device, checkpoint_path=checkpoint_dir, max_epochs=1, **common,
    )

    set_seed(999)
    resumed, resumed_history = train_model(
        _model(config, spec), train_ds, val_ds, spec, config, train_config,
        device, resume_from_checkpoint=checkpoint_dir / "latest.pt", **common,
    )

    assert (
        resumed_history["epoch_train_losses"]
        == full_history["epoch_train_losses"]
    )
    assert (
        resumed_history["epoch_val_accuracies"]
        == full_history["epoch_val_accuracies"]
    )
    assert resumed_history["best_val_accuracy"] == full_history["best_val_accuracy"]
    for (full_name, full_value), (resumed_name, resumed_value) in zip(
        full.state_dict().items(), resumed.state_dict().items()
    ):
        assert resumed_name == full_name
        assert torch.equal(resumed_value, full_value), full_name

    wrong = Qwen4MicroConfig(vocab_size=4, variant="hybrid")
    with pytest.raises(ValueError, match="qwen4_exp"):
        train_model(
            _model(wrong, spec), train_ds, val_ds, spec, wrong, train_config,
            device, resume_from_checkpoint=checkpoint_dir / "latest.pt",
            max_epochs=0, **common,
        )



# ---------------------------------------------------------------------------
# QSA index selection (exp1.qsa_indexer).
#
# Two implementations share one contract and differ in one promise:
#   causal-exact       reproduces upstream's selected mask exactly
#   batched-stable-v1  a separate apparatus; batching changes the reduction
#                      order, so near-ties may rank differently. It is NOT
#                      required to equal upstream.
#
# Invariants that hold for both are parametrized over the mode. Only the
# exact-mode oracle suite compares against upstream, whose forward is captured
# at import before any instance is rebound, so a fallback cannot recurse.
# ---------------------------------------------------------------------------

QSA_SEQ_LENGTHS = (1, 2, 3, 7, 17, 18, 32, 64, 128)
QSA_MODES = ("causal-exact", "batched-stable-v1")


def _indexer(device="cpu", seed=0, **overrides):
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextQSAIndexer,
    )

    values = Qwen4MicroConfig(vocab_size=16).resolved()
    for key in ("architecture", "transformers_version", "variant"):
        values.pop(key)
    values.pop("qsa_implementation", None)
    values.update(overrides)
    torch.manual_seed(seed)
    return Qwen4ExpTextQSAIndexer(Qwen4ExpTextConfig(**values), layer_idx=0).to(
        device
    ).eval()


def _rope(batch, seq, head_dim, device):
    position = torch.arange(seq, device=device).float().unsqueeze(-1)
    inverse = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    angle = position * inverse
    embedding = torch.cat([angle, angle], dim=-1)
    return (
        embedding.cos().unsqueeze(0).expand(batch, -1, -1).contiguous(),
        embedding.sin().unsqueeze(0).expand(batch, -1, -1).contiguous(),
    )


def _causal_mask(batch, seq, device, dtype=torch.bool, kv=None):
    kv = seq if kv is None else kv
    mask = torch.tril(torch.ones(seq, kv, dtype=torch.bool, device=device))
    mask = mask.view(1, 1, seq, kv).expand(batch, 1, seq, kv).contiguous()
    if dtype is torch.bool:
        return mask
    return torch.where(
        mask, torch.zeros((), dtype=dtype, device=device),
        torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device),
    )


def _run(mode, indexer, hidden, mask, past_key_values=None):
    from exp1 import qsa_indexer

    forward = {"causal-exact": qsa_indexer.causal_qsa_forward,
               "batched-stable-v1": qsa_indexer.batched_stable_qsa_forward}[mode]
    embeddings = _rope(hidden.shape[0], hidden.shape[1], indexer.index_head_dim,
                       hidden.device)
    with torch.no_grad():
        return forward(indexer, hidden, embeddings, mask, past_key_values)


def _run_upstream(indexer, hidden, mask, past_key_values=None):
    from exp1.qsa_indexer import _UPSTREAM_FORWARD

    embeddings = _rope(hidden.shape[0], hidden.shape[1], indexer.index_head_dim,
                       hidden.device)
    with torch.no_grad():
        return _UPSTREAM_FORWARD(indexer, hidden, embeddings, mask,
                                 past_key_values)


def _selected_sets(mask_out, seq):
    """Rows of the returned mask as sets of selected key indices."""
    boolean = mask_out if mask_out.dtype == torch.bool else mask_out == 0
    return [[set(torch.nonzero(boolean[b, 0, q]).flatten().tolist())
             for q in range(seq)]
            for b in range(boolean.shape[0])]


# --- exact-mode oracle suite (compact) --------------------------------------

@pytest.mark.parametrize("seq", QSA_SEQ_LENGTHS)
@pytest.mark.parametrize("batch", (1, 2, 64))
@pytest.mark.parametrize("dtype", (torch.bool, torch.float32))
def test_causal_exact_mask_equals_upstream_on_cpu(seq, batch, dtype):
    indexer = _indexer(seed=seq * 31 + batch)
    hidden = torch.randn(batch, seq, 128)
    mask = _causal_mask(batch, seq, "cpu", dtype)
    upstream = _run_upstream(indexer, hidden, mask)
    got = _run("causal-exact", indexer, hidden, mask)
    assert upstream.dtype == got.dtype
    assert torch.equal(upstream, got)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq", QSA_SEQ_LENGTHS)
@pytest.mark.parametrize("dtype", (torch.bool, torch.float32))
def test_causal_exact_mask_equals_upstream_on_cuda(seq, dtype):
    indexer = _indexer(device="cuda", seed=seq)
    hidden = torch.randn(4, seq, 128, device="cuda")
    mask = _causal_mask(4, seq, "cuda", dtype)
    assert torch.equal(_run_upstream(indexer, hidden, mask),
                       _run("causal-exact", indexer, hidden, mask))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_causal_exact_equals_upstream_under_exact_ties(device):
    """Ties are constructed, not approximated; scaling inputs down is not one."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    seq, batch = 18, 4
    indexer = _indexer(device=device)
    mask = _causal_mask(batch, seq, device)

    for hidden in _tied_inputs(indexer, batch, seq, device):
        assert torch.equal(_run_upstream(indexer, hidden, mask),
                           _run("causal-exact", indexer, hidden, mask))


def _tied_inputs(indexer, batch, seq, device):
    """Three constructions giving bitwise-equal candidate scores."""
    # 1. Zero hidden states: every score is exactly 0.0.
    yield torch.zeros(batch, seq, 128, device=device)
    # 2. A repeated pattern: distinct blocks pool to identical keys.
    unit = torch.randn(1, indexer.compress_ratio, 128, device=device)
    yield unit.repeat(batch, seq // indexer.compress_ratio + 1, 1)[:, :seq].contiguous()
    # 3. Zero the query half of the projection: keys vary, all scores are 0.
    with torch.no_grad():
        rows = indexer.index_n_heads * indexer.index_head_dim
        indexer.index_qk_proj.weight[:rows].zero_()
    yield torch.randn(batch, seq, 128, device=device)


# --- contracts both modes must satisfy --------------------------------------

@pytest.mark.parametrize("mode", QSA_MODES)
@pytest.mark.parametrize("seq", QSA_SEQ_LENGTHS)
@pytest.mark.parametrize("dtype", (torch.bool, torch.float32))
def test_qsa_selection_invariants(mode, seq, dtype):
    """Causal visibility, budget, complete blocks, and the incomplete tail.

    Covers short lengths (no complete block, fewer blocks than the budget) and
    batch 64 through the parametrization, for both mask dtypes.
    """
    batch = 64 if seq == 18 else 3
    indexer = _indexer(seed=seq)
    hidden = torch.randn(batch, seq, 128)
    out = _run(mode, indexer, hidden, _causal_mask(batch, seq, "cpu", dtype))
    assert out.shape == (batch, 1, seq, seq)

    ratio, budget = indexer.compress_ratio, indexer.token_budget
    for rows in _selected_sets(out, seq):
        for query, chosen in enumerate(rows):
            # never selects a key the causal mask hides
            assert all(key <= query for key in chosen), (mode, seq, query)
            complete = (query + 1) // ratio
            tail = set(range(complete * ratio, query + 1))
            # the incomplete trailing block is always kept whole
            assert tail <= chosen, (mode, seq, query)
            block_tokens = chosen - tail
            # selections are whole aligned blocks, never partial ones
            assert len(block_tokens) % ratio == 0
            for token in block_tokens:
                partner = token + 1 if token % ratio == 0 else token - 1
                assert partner in block_tokens, (mode, seq, query, token)
            # budget respected, and everything available is taken below it
            assert len(block_tokens) <= budget
            assert len(block_tokens) == min(complete, budget // ratio) * ratio


@pytest.mark.parametrize("mode", QSA_MODES)
def test_qsa_repeats_exactly_within_a_mode(mode):
    """Determinism within a mode, which is what a rerun of a cell relies on."""
    indexer = _indexer(seed=5)
    hidden = torch.randn(64, 18, 128)
    mask = _causal_mask(64, 18, "cpu")
    first = _run(mode, indexer, hidden, mask)
    for _ in range(3):
        assert torch.equal(first, _run(mode, indexer, hidden, mask))


def test_batched_stable_breaks_exact_ties_toward_lower_block_indices():
    """The tie rule is defined here, not inherited from torch.topk.

    With every score exactly equal, the budget must go to blocks 0, 1, 2, 3 --
    the lowest original indices -- for every query that has more complete
    blocks than the budget allows.
    """
    seq, batch = 18, 2
    indexer = _indexer()
    ratio, block_topk = indexer.compress_ratio, indexer.block_topk
    # Zero hidden states make every block score exactly 0.0.
    out = _run("batched-stable-v1", indexer, torch.zeros(batch, seq, 128),
               _causal_mask(batch, seq, "cpu"))

    for rows in _selected_sets(out, seq):
        for query, chosen in enumerate(rows):
            complete = (query + 1) // ratio
            if complete <= block_topk:
                continue
            tail = set(range(complete * ratio, query + 1))
            expected = set(range(block_topk * ratio))
            assert chosen - tail == expected, (query, sorted(chosen - tail))


def test_qsa_modes_disagree_only_by_selection_not_by_shape():
    """The batched mode is not required to equal upstream, but it must be a
    selector of the same kind: same output shape, dtype and tail behaviour."""
    indexer = _indexer(seed=11)
    hidden = torch.randn(8, 18, 128)
    mask = _causal_mask(8, 18, "cpu")
    exact = _run("causal-exact", indexer, hidden, mask)
    batched = _run("batched-stable-v1", indexer, hidden, mask)
    assert exact.shape == batched.shape
    assert exact.dtype == batched.dtype
    # Same number of selected keys per query, whichever blocks were chosen.
    assert torch.equal(exact.sum(-1), batched.sum(-1))


# --- fallback for inputs the contract does not cover -------------------------

def _fallback_cases(device="cpu"):
    seq, batch = 18, 4
    causal = _causal_mask(batch, seq, device)
    left = causal.clone()
    left[1, 0, :, 0] = False
    right = causal.clone()
    right[1, 0, -1, :] = False
    window = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))
    window &= ~torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device), -4)
    window = window.view(1, 1, seq, seq).expand(batch, 1, seq, seq).contiguous()
    return {
        "left padding": (left, False),
        "right padding": (right, False),
        "sliding window": (window, False),
        "key length exceeds query length": (_causal_mask(batch, seq, device,
                                                         kv=seq + 4), True),
        "mask rank is not four": (causal[:, 0], True),
        "mask head dimension is not one": (causal.expand(batch, 2, seq, seq), True),
    }


@pytest.mark.parametrize("name", list(_fallback_cases()))
def test_structural_contract_refuses_layouts_it_cannot_describe(name):
    """Structural refusals are host-side and cost no synchronization.

    Content-only violations (padding, a sliding window) pass the structural
    check by design: proving them needs a device read, which the registered
    path must not do. The guarded helper is what catches those.
    """
    from exp1.qsa_indexer import mask_is_full_causal, structural_contract_holds

    indexer = _indexer()
    hidden = torch.randn(4, 18, 128)
    mask, structural = _fallback_cases()[name]
    assert structural_contract_holds(indexer, hidden, mask, None) is not structural
    if not structural and mask.dim() == 4 and mask.shape[-1] == mask.shape[-2]:
        assert mask_is_full_causal(mask, 18) is False


def test_guarded_helper_falls_back_on_content_violations():
    """A generic direct caller still gets upstream's answer for a padded mask."""
    from exp1.qsa_indexer import guarded_qsa_forward

    indexer = _indexer(seed=3)
    hidden = torch.randn(4, 18, 128)
    padded = _causal_mask(4, 18, "cpu").clone()
    padded[1, 0, :, 0] = False
    embeddings = _rope(4, 18, indexer.index_head_dim, "cpu")
    with torch.no_grad():
        guarded = guarded_qsa_forward(indexer, hidden, embeddings, padded, None)
    assert torch.equal(guarded, _run_upstream(indexer, hidden, padded))


class _CountingCache:
    def __init__(self):
        self.calls = 0

    def update_indexer(self, indexer_key_states, *args):
        self.calls += 1
        return indexer_key_states


@pytest.mark.parametrize("mode", QSA_MODES)
def test_qsa_falls_back_for_a_cache_and_updates_it_once(mode):
    from exp1.qsa_indexer import structural_contract_holds

    indexer = _indexer()
    hidden = torch.randn(2, 18, 128)
    mask = _causal_mask(2, 18, "cpu")
    cache = _CountingCache()
    assert structural_contract_holds(indexer, hidden, mask, cache) is False
    _run(mode, indexer, hidden, mask, past_key_values=cache)
    # One update, from the single upstream call. Deciding the contract before
    # the projections is what makes that hold; a later check would write twice.
    assert cache.calls == 1


def test_unsupported_compression_and_budget_are_refused():
    from exp1.qsa_indexer import structural_contract_holds

    hidden = torch.randn(2, 18, 128)
    mask = _causal_mask(2, 18, "cpu")
    assert structural_contract_holds(_indexer(), hidden, mask, None) is True
    # Transformers' config validator already rejects a budget the ratio does
    # not divide, so this is exercised on the module; the check is defence in
    # depth if that validator is relaxed.
    ragged = _indexer()
    ragged.token_budget = 7
    assert structural_contract_holds(ragged, hidden, mask, None) is False
    skewed = _indexer()
    skewed.block_topk = skewed.token_budget // skewed.compress_ratio + 1
    assert structural_contract_holds(skewed, hidden, mask, None) is False


# --- model identity, installation, and resume -------------------------------

def _hybrid(mode, seed=7):
    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    set_seed(seed)
    config = Qwen4MicroConfig(vocab_size=16, qsa_implementation=mode)
    return _model(config, spec), config, spec


def test_default_identity_is_exact_and_omits_the_selector_key():
    """Existing checkpoints must stay loadable, so the default identity is
    byte-identical to the one written before this field existed."""
    exact = Qwen4MicroConfig(vocab_size=16)
    assert exact.qsa_implementation == "causal-exact"
    assert "qsa_implementation" not in exact.resolved()

    batched = Qwen4MicroConfig(vocab_size=16,
                               qsa_implementation="batched-stable-v1")
    assert batched.resolved()["qsa_implementation"] == "batched-stable-v1"
    assert _model_signature(exact) != _model_signature(batched)

    with pytest.raises(ValueError, match="qsa_implementation"):
        Qwen4MicroConfig(vocab_size=16, qsa_implementation="batched")


def test_selector_is_not_passed_into_the_transformers_config():
    """It is a repository-level field; Qwen4ExpTextConfig must never see it."""
    model, _, _ = _hybrid("batched-stable-v1")
    assert not hasattr(model.backbone.model.config, "qsa_implementation")


@pytest.mark.parametrize("mode", QSA_MODES)
def test_installed_forward_matches_the_requested_mode(mode):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextQSAIndexer,
    )

    from exp1 import qsa_indexer

    model, _, _ = _hybrid(mode)
    indexers = [m for m in model.modules()
                if isinstance(m, Qwen4ExpTextQSAIndexer)]
    assert len(indexers) == 1
    expected = {"causal-exact": qsa_indexer.causal_qsa_forward,
                "batched-stable-v1": qsa_indexer.batched_stable_qsa_forward}[mode]
    assert indexers[0].forward.__func__ is expected


def test_all_gdn_variant_is_untouched_by_either_mode():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextQSAIndexer,
    )

    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    for mode in QSA_MODES:
        set_seed(3)
        model = _model(Qwen4MicroConfig(vocab_size=16, variant="all-gdn",
                                        qsa_implementation=mode), spec)
        assert not [m for m in model.modules()
                    if isinstance(m, Qwen4ExpTextQSAIndexer)]
    set_seed(3)
    plain = _model(Qwen4MicroConfig(vocab_size=16, variant="all-gdn"), spec)
    set_seed(3)
    batched = _model(Qwen4MicroConfig(vocab_size=16, variant="all-gdn",
                                      qsa_implementation="batched-stable-v1"), spec)
    inputs = torch.randn(2, 18, spec.d_input)
    with torch.no_grad():
        assert torch.equal(forward_logits(plain, inputs),
                           forward_logits(batched, inputs))


def test_state_dict_names_and_shapes_match_across_modes(tmp_path):
    exact, _, _ = _hybrid("causal-exact")
    batched, _, _ = _hybrid("batched-stable-v1")
    assert list(exact.state_dict()) == list(batched.state_dict())
    for name, value in exact.state_dict().items():
        assert value.shape == batched.state_dict()[name].shape, name

    # A checkpoint from either loads strictly into the other: the selector
    # changes behaviour, never parameter layout.
    path = tmp_path / "exact.pt"
    torch.save(exact.state_dict(), path)
    report = batched.load_state_dict(torch.load(path, weights_only=True),
                                     strict=True)
    assert not report.missing_keys and not report.unexpected_keys


def _tiny_training(mode, tmp_path, **kwargs):
    spec = ChaseSpec(num_nodes=4, num_maps=2, max_depth=2)
    train_ds = PointerChaseDataset(
        generate_dataset(1, 1, depth=1, seed=1, num_nodes=4, num_maps=2), spec)
    val_ds = PointerChaseDataset(
        generate_dataset(1, 1, depth=1, seed=2, num_nodes=4, num_maps=2), spec)
    config = Qwen4MicroConfig(vocab_size=4, variant="hybrid",
                              qsa_implementation=mode)
    train_config = TrainConfig(seed=42, batch_size=1, precision="fp32",
                               num_workers=0, epochs=2, learning_rate=1e-3)
    common = dict(depth=1, train_data_seed=1, val_data_seed=2, train_size=1,
                  val_size=1, queries_per_memory=1)
    return train_model(_model(config, spec), train_ds, val_ds, spec, config,
                       train_config, torch.device("cpu"), **common, **kwargs)


@pytest.mark.parametrize("mode", QSA_MODES)
def test_same_mode_checkpoint_resume_is_exact(mode, tmp_path):
    set_seed(42)
    full, full_history = _tiny_training(mode, tmp_path)
    checkpoints = tmp_path / f"ckpt-{mode}"
    set_seed(42)
    _tiny_training(mode, tmp_path, checkpoint_path=checkpoints, max_epochs=1)
    set_seed(999)
    resumed, resumed_history = _tiny_training(
        mode, tmp_path, resume_from_checkpoint=checkpoints / "latest.pt")

    assert resumed_history["epoch_train_losses"] == full_history["epoch_train_losses"]
    assert (resumed_history["epoch_val_accuracies"]
            == full_history["epoch_val_accuracies"])
    for name, value in full.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], value), name


def test_resume_is_rejected_across_qsa_implementations(tmp_path):
    """A batched run must never silently resume an exact checkpoint."""
    checkpoints = tmp_path / "exact-ckpt"
    set_seed(42)
    _tiny_training("causal-exact", tmp_path, checkpoint_path=checkpoints,
                   max_epochs=1)
    with pytest.raises(ValueError, match="qwen4_exp"):
        _tiny_training("batched-stable-v1", tmp_path,
                       resume_from_checkpoint=checkpoints / "latest.pt",
                       max_epochs=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", QSA_MODES)
def test_registered_forward_adds_no_host_synchronization(mode):
    """The indexer must cost no extra sync over the variant that has none.

    A device-to-host read here is one synchronization per training step and
    prevents CUDA Graph capture, which is the reason the mask contract lives at
    the wrapper boundary instead of in the forward.
    """
    import warnings

    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)

    def syncs(model, inputs):
        with torch.no_grad():
            forward_logits(model, inputs)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                forward_logits(model, inputs)
        torch.cuda.set_sync_debug_mode("default")
        return sum(1 for w in caught if "synchroniz" in str(w.message))

    inputs = torch.randn(8, 18, spec.d_input, device="cuda")
    set_seed(3)
    reference = _model(Qwen4MicroConfig(vocab_size=16, variant="all-gdn"),
                       spec).cuda()
    set_seed(3)
    hybrid = _model(Qwen4MicroConfig(vocab_size=16, qsa_implementation=mode),
                    spec).cuda()
    assert syncs(hybrid, inputs) <= syncs(reference, inputs)
