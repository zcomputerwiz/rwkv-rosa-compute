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
# QSA indexer without the per-query torch.nonzero (exp1.qsa_indexer): exact
# equality against the installed upstream method, and explicit fallback
# everywhere the predicate is false.
#
# The oracle is the upstream method captured at import, before any instance is
# rebound, so a fallback cannot recurse into the replacement under test.
# ---------------------------------------------------------------------------

QSA_SEQ_LENGTHS = (1, 2, 3, 7, 17, 18, 32, 64, 128)


def _indexer(device="cpu", seed=0, **overrides):
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextQSAIndexer,
    )

    values = Qwen4MicroConfig(vocab_size=16).resolved()
    for key in ("architecture", "transformers_version", "variant"):
        values.pop(key)
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


def _both_paths(indexer, hidden, mask, past_key_values=None):
    """Return (upstream, fast) for the same input, without installing."""
    from exp1.qsa_indexer import _UPSTREAM_FORWARD, causal_qsa_forward

    batch, seq, _ = hidden.shape
    embeddings = _rope(batch, seq, indexer.index_head_dim, hidden.device)
    with torch.no_grad():
        upstream = _UPSTREAM_FORWARD(
            indexer, hidden, embeddings, mask, past_key_values
        )
        fast = causal_qsa_forward(
            indexer, hidden, embeddings, mask, past_key_values
        )
    return upstream, fast


def _use_upstream_indexer(model):
    """Drop the instance override so the class method is used again."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextQSAIndexer,
    )

    removed = 0
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextQSAIndexer) and "forward" in module.__dict__:
            del module.__dict__["forward"]
            removed += 1
    return removed


@pytest.mark.parametrize("seq", QSA_SEQ_LENGTHS)
@pytest.mark.parametrize("batch", (1, 2, 64))
@pytest.mark.parametrize("dtype", (torch.bool, torch.float32))
def test_no_nonzero_qsa_mask_is_exactly_upstream_on_cpu(seq, batch, dtype):
    indexer = _indexer(seed=seq * 31 + batch)
    hidden = torch.randn(batch, seq, 128)
    upstream, fast = _both_paths(indexer, hidden, _causal_mask(batch, seq,
                                                                 "cpu", dtype))
    assert upstream.dtype == fast.dtype
    assert torch.equal(upstream, fast)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq", QSA_SEQ_LENGTHS)
@pytest.mark.parametrize("dtype", (torch.bool, torch.float32))
def test_no_nonzero_qsa_mask_is_exactly_upstream_on_cuda(seq, dtype):
    indexer = _indexer(device="cuda", seed=seq)
    hidden = torch.randn(4, seq, 128, device="cuda")
    upstream, fast = _both_paths(indexer, hidden, _causal_mask(4, seq, "cuda",
                                                                 dtype))
    assert torch.equal(upstream, fast)


def test_no_nonzero_qsa_covers_lengths_below_a_block_and_below_the_budget():
    """seq 1 has no complete block; seq 7 has fewer selectable than the budget."""
    indexer = _indexer()
    assert indexer.compress_ratio == 2 and indexer.token_budget == 8
    for seq in (1, 7):
        hidden = torch.randn(3, seq, 128)
        upstream, fast = _both_paths(indexer, hidden,
                                        _causal_mask(3, seq, "cpu"))
        assert torch.equal(upstream, fast), seq


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_no_nonzero_qsa_agrees_under_exact_score_ties(device):
    """Ties are constructed, not approximated: scaling inputs down is not a tie.

    Three constructions, each producing bitwise-equal candidate scores so the
    two topk call shapes must break the tie the same way for the masks to match.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    seq, batch = 18, 4
    indexer = _indexer(device=device)
    mask = _causal_mask(batch, seq, device)

    # 1. Zero hidden states: every score is exactly 0.0.
    upstream, fast = _both_paths(
        indexer, torch.zeros(batch, seq, 128, device=device), mask
    )
    assert torch.equal(upstream, fast)

    # 2. A repeated two-token pattern: distinct blocks pool to identical keys.
    unit = torch.randn(1, indexer.compress_ratio, 128, device=device)
    repeated = unit.repeat(batch, seq // indexer.compress_ratio + 1, 1)[:, :seq]
    upstream, fast = _both_paths(indexer, repeated.contiguous(), mask)
    assert torch.equal(upstream, fast)

    # 3. Zero the query half of the projection: keys vary, all scores are 0.
    with torch.no_grad():
        rows = indexer.index_n_heads * indexer.index_head_dim
        indexer.index_qk_proj.weight[:rows].zero_()
    upstream, fast = _both_paths(
        indexer, torch.randn(batch, seq, 128, device=device), mask
    )
    assert torch.equal(upstream, fast)


class _CountingCache:
    """Minimal stand-in that records how often the indexer cache is updated."""

    def __init__(self):
        self.calls = 0

    def update_indexer(self, indexer_key_states, *args):
        self.calls += 1
        return indexer_key_states


def _fallback_cases(device="cpu"):
    seq, batch = 18, 4
    causal = _causal_mask(batch, seq, device)

    left = causal.clone()
    left[1, 0, :, 0] = False
    right = causal.clone()
    right[1, 0, -1, :] = False

    window = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))
    window &= ~torch.tril(
        torch.ones(seq, seq, dtype=torch.bool, device=device), -4
    )
    window = window.view(1, 1, seq, seq).expand(batch, 1, seq, seq).contiguous()

    return {
        "left padding": left,
        "right padding": right,
        "sliding window": window,
        "key length exceeds query length": _causal_mask(batch, seq, device,
                                                        kv=seq + 4),
        "mask rank is not four": causal[:, 0],
        "mask head dimension is not one": causal.expand(batch, 2, seq, seq),
    }


@pytest.mark.parametrize("name", list(_fallback_cases()))
def test_no_nonzero_qsa_refuses_masks_it_cannot_prove(name):
    from exp1.qsa_indexer import optimized_path_applies

    indexer = _indexer()
    hidden = torch.randn(4, 18, 128)
    mask = _fallback_cases()[name]
    assert optimized_path_applies(indexer, hidden, mask, None) is False


def test_no_nonzero_qsa_refuses_unsupported_compression_and_budget():
    from exp1.qsa_indexer import optimized_path_applies

    hidden = torch.randn(2, 18, 128)
    mask = _causal_mask(2, 18, "cpu")
    assert optimized_path_applies(_indexer(), hidden, mask, None) is True

    # Transformers' own config validator already rejects a budget the ratio
    # does not divide, so these assumptions are exercised on the constructed
    # module rather than through a config that cannot be built. The predicate
    # is defence in depth if that validator is ever relaxed.
    ragged = _indexer()
    ragged.token_budget = 7
    assert ragged.token_budget % ragged.compress_ratio != 0
    assert optimized_path_applies(ragged, hidden, mask, None) is False

    # block_topk out of step with budget // ratio breaks the output-width bound
    # upstream's own writer assumes, even when the budget divides cleanly.
    skewed = _indexer()
    skewed.block_topk = skewed.token_budget // skewed.compress_ratio + 1
    assert optimized_path_applies(skewed, hidden, mask, None) is False


def test_no_nonzero_qsa_falls_back_for_a_cache_and_updates_it_once():
    from exp1.qsa_indexer import causal_qsa_forward, optimized_path_applies

    indexer = _indexer()
    hidden = torch.randn(2, 18, 128)
    mask = _causal_mask(2, 18, "cpu")
    cache = _CountingCache()

    assert optimized_path_applies(indexer, hidden, mask, cache) is False
    embeddings = _rope(2, 18, indexer.index_head_dim, hidden.device)
    with torch.no_grad():
        causal_qsa_forward(indexer, hidden, embeddings, mask, cache)
    # One update, from the single upstream call. The predicate must be decided
    # before any projection or cache write, or this would be two.
    assert cache.calls == 1


def _hybrid_pair(seq=18, batch=4, seed=7):
    """Two identically seeded hybrid models: one fast-path, one upstream."""
    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    config = Qwen4MicroConfig(vocab_size=16)
    set_seed(seed)
    fast = _model(config, spec)
    set_seed(seed)
    upstream = _model(config, spec)
    assert _use_upstream_indexer(upstream) == 1
    inputs = torch.randn(batch, seq, spec.d_input)
    targets = torch.randint(0, 16, (batch,))
    return fast, upstream, inputs, targets


def test_no_nonzero_qsa_gives_identical_logits_and_loss():
    fast, upstream, inputs, targets = _hybrid_pair()
    with torch.no_grad():
        want = forward_logits(upstream, inputs)
        got = forward_logits(fast, inputs)
    assert torch.equal(want, got)
    assert torch.equal(
        torch.nn.functional.cross_entropy(want, targets),
        torch.nn.functional.cross_entropy(got, targets),
    )


def test_no_nonzero_qsa_gives_identical_gradients_after_one_backward():
    fast, upstream, inputs, targets = _hybrid_pair()
    for model in (upstream, fast):
        torch.nn.functional.cross_entropy(
            forward_logits(model, inputs), targets
        ).backward()

    want = dict(upstream.named_parameters())
    got = dict(fast.named_parameters())
    assert want.keys() == got.keys()
    assert any(p.grad is not None for p in want.values())
    for name, parameter in want.items():
        if parameter.grad is None:
            assert got[name].grad is None, name
        else:
            assert torch.equal(parameter.grad, got[name].grad), name


def test_no_nonzero_qsa_gives_identical_state_after_one_optimizer_step():
    fast, upstream, inputs, targets = _hybrid_pair()
    states = []
    for model in (upstream, fast):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(
            forward_logits(model, inputs), targets
        ).backward()
        optimizer.step()
        states.append((model.state_dict(), optimizer.state_dict()))

    (want_model, want_optimizer), (got_model, got_optimizer) = states
    assert list(want_model) == list(got_model)
    for name, value in want_model.items():
        assert value.shape == got_model[name].shape, name
        assert torch.equal(value, got_model[name]), name

    assert want_optimizer["param_groups"] == got_optimizer["param_groups"]
    assert want_optimizer["state"].keys() == got_optimizer["state"].keys()
    for key, entry in want_optimizer["state"].items():
        for field, value in entry.items():
            other = got_optimizer["state"][key][field]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), (key, field)
            else:
                assert value == other, (key, field)


def test_no_nonzero_qsa_preserves_state_dict_keys_and_loads_both_ways(tmp_path):
    fast, upstream, _, _ = _hybrid_pair()
    assert list(fast.state_dict()) == list(upstream.state_dict())
    for name, value in fast.state_dict().items():
        assert value.shape == upstream.state_dict()[name].shape, name

    # A checkpoint written by either path must load strictly into the other.
    path = tmp_path / "upstream.pt"
    torch.save(upstream.state_dict(), path)
    report = fast.load_state_dict(torch.load(path, weights_only=True),
                                     strict=True)
    assert not report.missing_keys and not report.unexpected_keys

    path = tmp_path / "fast.pt"
    torch.save(fast.state_dict(), path)
    report = upstream.load_state_dict(torch.load(path, weights_only=True),
                                      strict=True)
    assert not report.missing_keys and not report.unexpected_keys


def test_hybrid_checkpoint_resume_is_exact_with_the_fast_indexer(tmp_path):
    """The existing resume identity test, on the variant that has an indexer."""
    spec = ChaseSpec(num_nodes=4, num_maps=2, max_depth=2)
    train_ds = PointerChaseDataset(
        generate_dataset(1, 1, depth=1, seed=1, num_nodes=4, num_maps=2), spec,
    )
    val_ds = PointerChaseDataset(
        generate_dataset(1, 1, depth=1, seed=2, num_nodes=4, num_maps=2), spec,
    )
    config = Qwen4MicroConfig(vocab_size=4, variant="hybrid")
    train_config = TrainConfig(
        seed=42, batch_size=1, precision="fp32", num_workers=0, epochs=2,
        learning_rate=1e-3,
    )
    common = dict(depth=1, train_data_seed=1, val_data_seed=2, train_size=1,
                  val_size=1, queries_per_memory=1)
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

    assert resumed_history["epoch_train_losses"] == full_history["epoch_train_losses"]
    assert (
        resumed_history["epoch_val_accuracies"]
        == full_history["epoch_val_accuracies"]
    )
    for name, value in full.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], value), name


def test_no_nonzero_qsa_is_installed_on_hybrid_and_absent_from_all_gdn():
    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextQSAIndexer,
    )

    hybrid = _model(Qwen4MicroConfig(vocab_size=16), spec)
    indexers = [
        module for module in hybrid.modules()
        if isinstance(module, Qwen4ExpTextQSAIndexer)
    ]
    assert len(indexers) == 1
    assert "forward" in indexers[0].__dict__

    recurrent = _model(Qwen4MicroConfig(vocab_size=16, variant="all-gdn"), spec)
    assert not [
        module for module in recurrent.modules()
        if isinstance(module, Qwen4ExpTextQSAIndexer)
    ]


def test_no_nonzero_qsa_path_is_actually_taken_by_a_model_forward(monkeypatch):
    """Without this, every equality test above could pass vacuously.

    Equality against upstream proves nothing if the fast path is never
    entered, so this proves the model's forward reaches it and takes the
    optimized branch rather than the fallback.
    """
    import exp1.qsa_indexer as qsa

    calls = {"fast": 0, "fallback": 0}
    real = qsa.causal_qsa_forward

    def counting(self, hidden_states, position_embeddings, attention_mask,
                 past_key_values):
        if qsa.optimized_path_applies(self, hidden_states, attention_mask,
                                      past_key_values):
            calls["fast"] += 1
        else:
            calls["fallback"] += 1
        return real(self, hidden_states, position_embeddings, attention_mask,
                    past_key_values)

    monkeypatch.setattr(qsa, "causal_qsa_forward", counting)
    spec = ChaseSpec(num_nodes=16, num_maps=4, max_depth=32)
    model = _model(Qwen4MicroConfig(vocab_size=16), spec)
    assert qsa.install_causal_qsa_indexer(model) == 1

    with torch.no_grad():
        forward_logits(model, torch.randn(4, 18, spec.d_input))

    assert calls == {"fast": 1, "fallback": 0}
