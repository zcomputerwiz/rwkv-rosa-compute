import pytest
import torch

from exp0.config import ModelConfig, TrainConfig
from exp0.train import create_model, set_seed
from exp1.dataset import PointerChaseDataset, exp1_collate_fn
from exp1.pointer_chase import ChaseSpec, generate_dataset
from exp1.train import evaluate_vway_accuracy, train_model

M, K = 16, 4

def test_dataset_adapter_yields_correct_shapes():
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=16)
    data = generate_dataset(2, 4, depth=8, seed=0, num_nodes=M, num_maps=K)
    ds = PointerChaseDataset(data, spec, num_silent=0)

    assert len(ds) == 8
    item = ds[0]
    assert "input_tuples" in item
    assert "targets" in item
    assert item["input_tuples"].shape == (spec.seq_len(0), spec.d_input)

    batch = exp1_collate_fn([ds[i] for i in range(4)])
    assert batch["input_tuples"].shape == (4, spec.seq_len(0), spec.d_input)
    assert batch["targets"].shape == (4,)

def test_encoded_input_width_identical_across_depths():
    """D must not change the architecture."""
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=32)
    widths = set()
    for d in (1, 4, 16, 32):
        data = generate_dataset(1, 2, depth=d, seed=d, num_nodes=M, num_maps=K)
        ds = PointerChaseDataset(data, spec)
        widths.add(ds[0]["input_tuples"].shape[-1])
    assert widths == {spec.d_input}

def test_untrained_model_scores_near_chance():
    """An untrained model on a balanced evaluation set scores near 1/V, not near 1/2."""
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    data = generate_dataset(250, 4, depth=4, seed=42, num_nodes=M, num_maps=K)
    ds = PointerChaseDataset(data, spec)

    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=64,
        vocab_size=M,
        rwkv_kernel="reference",
    )
    model = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    train_cfg = TrainConfig(seed=42, batch_size=128, precision="fp32", num_workers=0)
    device = torch.device("cpu")

    acc = evaluate_vway_accuracy(model, ds, train_cfg, device)

    # Chance is 1/M = 1/16 = 0.0625
    # Should definitely not be near 0.5
    assert acc < 0.15
    assert abs(acc - 1/M) < 0.08

def test_tiny_e2e_training_smoke():
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    train_data = generate_dataset(2, 4, depth=4, seed=1, num_nodes=M, num_maps=K)
    val_data = generate_dataset(2, 4, depth=4, seed=2, num_nodes=M, num_maps=K)

    train_ds = PointerChaseDataset(train_data, spec)
    val_ds = PointerChaseDataset(val_data, spec)

    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=64,
        vocab_size=M,
        rwkv_kernel="reference",
    )
    model = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    train_cfg = TrainConfig(
        seed=42,
        batch_size=4,
        precision="fp32",
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
    )
    device = torch.device("cpu")

    model, history = train_model(
        model,
        train_ds,
        val_ds,
        spec,
        model_cfg,
        train_cfg,
        device,
        depth=4,
        train_data_seed=1,
        val_data_seed=2,
        train_size=2,
        val_size=2,
    )

    assert len(history["epoch_train_losses"]) == 1
    assert len(history["epoch_val_accuracies"]) == 1
    assert "best_val_accuracy" in history

def test_training_resumes_from_checkpoint(tmp_path):
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    train_data = generate_dataset(4, 4, depth=4, seed=1, num_nodes=M, num_maps=K)
    val_data = generate_dataset(2, 4, depth=4, seed=2, num_nodes=M, num_maps=K)

    train_ds = PointerChaseDataset(train_data, spec)
    val_ds = PointerChaseDataset(val_data, spec)

    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=64,
        vocab_size=M,
        rwkv_kernel="reference",
    )
    model = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    train_cfg = TrainConfig(
        seed=42,
        batch_size=4,
        precision="fp32",
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
    )
    device = torch.device("cpu")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    model, history_1 = train_model(
        model,
        train_ds,
        val_ds,
        spec,
        model_cfg,
        train_cfg,
        device,
        checkpoint_path=checkpoint_dir,
        depth=4,
        train_data_seed=1,
        val_data_seed=2,
        train_size=4,
        val_size=2,
    )

    # Assert a checkpoint was created
    latest_ckpt = checkpoint_dir / "latest.pt"
    assert latest_ckpt.exists()

    # Resume training for 1 more epoch, but the epochs field is part of the signature!
    # So we must use max_epochs=1 with the original train_cfg (which had epochs=1)
    # to test resumption, OR we construct the test differently.
    # Let's adjust epochs=2 and use max_epochs=1 in the first run.
    train_cfg_total = TrainConfig(
        seed=42,
        batch_size=4,
        precision="fp32",
        num_workers=0,
        epochs=2,
        learning_rate=1e-3,
    )

    # We re-run the first part to ensure the signature recorded epochs=2
    checkpoint_dir_2 = tmp_path / "checkpoints_2"
    checkpoint_dir_2.mkdir()

    model2 = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)
    model2, history_1 = train_model(
        model2, train_ds, val_ds, spec, model_cfg, train_cfg_total, device,
        checkpoint_path=checkpoint_dir_2, depth=4, train_data_seed=1, val_data_seed=2, train_size=4, val_size=2, max_epochs=1
    )

    latest_ckpt_2 = checkpoint_dir_2 / "latest.pt"

    model_resumed = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    model_resumed, history_2 = train_model(
        model_resumed,
        train_ds,
        val_ds,
        spec,
        model_cfg,
        train_cfg_total,
        device,
        checkpoint_path=checkpoint_dir_2,
        resume_from_checkpoint=latest_ckpt_2,
        depth=4,
        train_data_seed=1,
        val_data_seed=2,
        train_size=4,
        val_size=2,
    )

    # history_1 ran 1 epoch, history_2 ran 1 epoch (from epoch 1 to 2)
    assert history_1["epochs_trained"] == 1
    assert history_2["epochs_trained"] == 1

    # Redefine checkpoint_dir for the asserts below
    checkpoint_dir = checkpoint_dir_2


    # Verify optimizer state was actually restored
    from exp0.checkpointing import load_training_checkpoint
    ckpt_data_1 = load_training_checkpoint(checkpoint_dir / "epoch_001.pt")
    ckpt_data_2 = load_training_checkpoint(checkpoint_dir / "epoch_002.pt")

    # Optimizer steps should have incremented from epoch 1 to epoch 2
    assert ckpt_data_1["progress"]["optimizer_steps"] > 0
    assert ckpt_data_2["progress"]["optimizer_steps"] > ckpt_data_1["progress"]["optimizer_steps"]

    # Ensure state dictionary holds actual parameter states
    assert "state" in ckpt_data_2["optimizer_state_dict"]
    assert len(ckpt_data_2["optimizer_state_dict"]["state"]) > 0

    # Check that another checkpoint was saved
    assert (checkpoint_dir / "epoch_002.pt").exists()

def test_model_initialization_seeding():
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=64,
        vocab_size=M,
        rwkv_kernel="reference",
    )

    import hashlib

    from exp0.train import set_seed

    def hash_model(seed):
        set_seed(seed)
        m = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)
        sha = hashlib.sha256()
        for name, p in sorted(m.state_dict().items()):
            sha.update(name.encode())
            sha.update(p.detach().cpu().contiguous().numpy().tobytes())
        return sha.hexdigest()

    h1 = hash_model(42)
    h2 = hash_model(42)
    h3 = hash_model(43)

    assert h1 == h2, "Model parameters are not identical for the same seed"
    assert h1 != h3, "Model parameters identical across different seeds"

def test_exact_resume_and_history_preservation(tmp_path):
    """Defect 3: Resume must preserve history and be functionally exact to an uninterrupted run."""
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    train_data = generate_dataset(4, 4, depth=4, seed=1, num_nodes=M, num_maps=K)
    val_data = generate_dataset(2, 4, depth=4, seed=2, num_nodes=M, num_maps=K)

    train_ds = PointerChaseDataset(train_data, spec)
    val_ds = PointerChaseDataset(val_data, spec)

    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=64,
        vocab_size=M,
        rwkv_kernel="reference",
    )

    train_cfg_full = TrainConfig(seed=42, batch_size=4, precision="fp32", num_workers=0, epochs=2, learning_rate=1e-3)
    device = torch.device("cpu")

    # 1. Train 2 epochs uninterrupted
    from exp0.train import set_seed
    set_seed(42)
    model_full = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    model_full, history_full = train_model(
        model_full, train_ds, val_ds, spec, model_cfg, train_cfg_full, device,
        depth=4, train_data_seed=1, val_data_seed=2, train_size=4, val_size=2
    )

    # 2. Train 1 epoch, save, resume for 1 more epoch
    set_seed(42)
    model_part = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    checkpoint_dir = tmp_path / "checkpoints_exact"
    checkpoint_dir.mkdir()

    model_part, history_part1 = train_model(
        model_part, train_ds, val_ds, spec, model_cfg, train_cfg_full, device, checkpoint_path=checkpoint_dir, max_epochs=1,
        depth=4, train_data_seed=1, val_data_seed=2, train_size=4, val_size=2
    )

    latest_ckpt = checkpoint_dir / "latest.pt"

    # Notice we pass `epochs=2` here because `train_model` handles start_epoch up to epochs.
    # We must construct a fresh model to prove resume works from disk.
    set_seed(999) # different seed to ensure we rely on loaded weights
    model_resumed = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    model_resumed, history_resumed = train_model(
        model_resumed, train_ds, val_ds, spec, model_cfg, train_cfg_full, device,
        checkpoint_path=checkpoint_dir, resume_from_checkpoint=latest_ckpt,
        depth=4, train_data_seed=1, val_data_seed=2, train_size=4, val_size=2
    )

    # Assert histories match
    assert history_full["epoch_train_losses"] == history_resumed["epoch_train_losses"], "Train losses diverge after resume"
    assert history_full["epoch_val_accuracies"] == history_resumed["epoch_val_accuracies"], "Val accuracies diverge after resume"
    assert history_full["best_val_accuracy"] == history_resumed["best_val_accuracy"], "Best val acc differs"

    # Assert parameters match exactly
    for (name1, p1), (name2, p2) in zip(model_full.state_dict().items(), model_resumed.state_dict().items()):
        assert name1 == name2
        assert torch.equal(p1, p2), f"Parameter {name1} diverges after resume"

def test_step_periodic_checkpointing(tmp_path):
    """Defect 4: Ensure checkpointing works per-step with a sample offset."""
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    train_data = generate_dataset(4, 4, depth=4, seed=1, num_nodes=M, num_maps=K)
    val_data = generate_dataset(2, 4, depth=4, seed=2, num_nodes=M, num_maps=K)

    train_ds = PointerChaseDataset(train_data, spec)
    val_ds = PointerChaseDataset(val_data, spec)

    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=64,
        vocab_size=M,
        rwkv_kernel="reference",
    )

    train_cfg = TrainConfig(seed=42, batch_size=4, precision="fp32", num_workers=0, epochs=1, learning_rate=1e-3)
    device = torch.device("cpu")

    from exp0.train import set_seed
    set_seed(42)
    model = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    checkpoint_dir = tmp_path / "checkpoints_steps"
    checkpoint_dir.mkdir()

    # Train and checkpoint every 2 steps
    train_model(
        model, train_ds, val_ds, spec, model_cfg, train_cfg, device,
        checkpoint_path=checkpoint_dir, checkpoint_every_steps=2,
        depth=4, train_data_seed=1, val_data_seed=2, train_size=4, val_size=2
    )

    assert (checkpoint_dir / "step_000002.pt").exists()
    assert (checkpoint_dir / "step_000004.pt").exists()
    assert (checkpoint_dir / "epoch_001.pt").exists()


def test_training_resumes_refused_on_signature_mismatch(tmp_path):
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    train_data = generate_dataset(4, 4, depth=4, seed=1, num_nodes=M, num_maps=K)
    val_data = generate_dataset(2, 4, depth=4, seed=2, num_nodes=M, num_maps=K)

    train_ds = PointerChaseDataset(train_data, spec)
    val_ds = PointerChaseDataset(val_data, spec)

    model_cfg = ModelConfig(
        architecture="rwkv",
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=64,
        vocab_size=M,
        rwkv_kernel="reference",
    )
    model = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    train_cfg = TrainConfig(
        seed=42,
        batch_size=4,
        precision="fp32",
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
    )
    device = torch.device("cpu")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    train_model(
        model,
        train_ds,
        val_ds,
        spec,
        model_cfg,
        train_cfg,
        device,
        checkpoint_path=checkpoint_dir,
        depth=4,
        train_data_seed=1,
        val_data_seed=2,
        train_size=4,
        val_size=2,
    )

    latest_ckpt = checkpoint_dir / "latest.pt"
    assert latest_ckpt.exists()

    model_resumed = create_model(model_cfg, d_input=spec.d_input, compact_reduced_features=False)

    # Change depth
    with pytest.raises(ValueError, match="Training checkpoint does not match the requested run. Differing signature sections: depth"):
        train_model(
            model_resumed, train_ds, val_ds, spec, model_cfg, train_cfg, device,
            resume_from_checkpoint=latest_ckpt,
            depth=5, train_data_seed=1, val_data_seed=2, train_size=4, val_size=2,
        )

    # Change model_seed (train_cfg.seed)
    train_cfg_diff_seed = TrainConfig(seed=43, batch_size=4, precision="fp32", num_workers=0, epochs=1, learning_rate=1e-3)
    with pytest.raises(ValueError, match="Training checkpoint does not match the requested run. Differing signature sections: model_seed"):
        train_model(
            model_resumed, train_ds, val_ds, spec, model_cfg, train_cfg_diff_seed, device,
            resume_from_checkpoint=latest_ckpt,
            depth=4, train_data_seed=1, val_data_seed=2, train_size=4, val_size=2,
        )

    # Change train_data_seed
    with pytest.raises(ValueError, match="Training checkpoint does not match the requested run. Differing signature sections: train_data_seed"):
        train_model(
            model_resumed, train_ds, val_ds, spec, model_cfg, train_cfg, device,
            resume_from_checkpoint=latest_ckpt,
            depth=4, train_data_seed=2, val_data_seed=2, train_size=4, val_size=2,
        )


def test_identity_arguments_cannot_be_omitted():
    """A default here would be written into the signature as if it were real.

    Two callers that both forgot the same argument would then produce
    checkpoints that validate against each other, which is exactly the
    mislabeled-result failure the signature exists to prevent.
    """
    import inspect

    params = inspect.signature(train_model).parameters
    for name in ("depth", "train_data_seed", "val_data_seed",
                 "train_size", "val_size"):
        assert params[name].default is inspect.Parameter.empty, (
            f"{name} has a default; an omitted caller would silently record it")


def _tiny_setup():
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    train_ds = PointerChaseDataset(
        generate_dataset(2, 4, depth=4, seed=1, num_nodes=M, num_maps=K), spec)
    val_ds = PointerChaseDataset(
        generate_dataset(2, 4, depth=4, seed=2, num_nodes=M, num_maps=K), spec)
    model_cfg = ModelConfig(
        architecture="rwkv", hidden_size=64, num_hidden_layers=1,
        num_attention_heads=1, head_dim=64, vocab_size=M,
        rwkv_kernel="reference",
    )
    model = create_model(model_cfg, d_input=spec.d_input,
                         compact_reduced_features=False)
    train_cfg = TrainConfig(seed=42, batch_size=4, precision="fp32",
                            num_workers=0, epochs=1, learning_rate=1e-3)
    return model, train_ds, val_ds, spec, model_cfg, train_cfg


def _train(model, train_ds, val_ds, spec, model_cfg, train_cfg, **kwargs):
    return train_model(
        model, train_ds, val_ds, spec, model_cfg, train_cfg,
        torch.device("cpu"),
        depth=4, train_data_seed=1, val_data_seed=2, train_size=2, val_size=2,
        **kwargs,
    )


def test_compile_backend_is_rejected_when_the_model_is_not_compiled():
    """Naming a backend for an eager model would write a false resume signature.

    The signature would then claim the run used that backend, and a genuinely
    compiled resume against it would be accepted.
    """
    parts = _tiny_setup()
    with pytest.raises(ValueError, match="not compiled"):
        _train(*parts, compile_backend="cudagraphs")


def test_a_compiled_backbone_without_a_named_backend_is_rejected():
    """Compiling and saying nothing would leave the signature indistinguishable.

    inductor and cudagraphs do not produce the same trajectory, and the backend
    is not recoverable from the wrapper, so it has to be declared.
    """
    model, train_ds, val_ds, spec, model_cfg, train_cfg = _tiny_setup()
    model.backbone = torch.compile(model.backbone)
    with pytest.raises(ValueError, match="compile_backend was not passed"):
        _train(model, train_ds, val_ds, spec, model_cfg, train_cfg)


def test_the_resume_signature_records_the_compile_backend(tmp_path):
    """An eager run must not resume from a compiled checkpoint, or the reverse."""
    parts = _tiny_setup()
    ckpt = tmp_path / "ckpt"
    _train(*parts, checkpoint_path=ckpt)
    saved = torch.load(ckpt / "latest.pt", map_location="cpu",
                       weights_only=False)
    assert "compile_backend" in saved["signature"]
    assert saved["signature"]["compile_backend"] is None


def test_a_midepoch_resume_reports_the_whole_epoch_loss(tmp_path):
    """A resumed epoch must report the mean over all its batches, not the tail.

    The epoch loss accumulator was reinitialised on resume and the partial
    checkpoint did not carry it, so an epoch interrupted after most of its work
    reported the mean of only the batches that ran afterwards. That is silently
    wrong: the number is plausible, it is stored in the report, and nothing
    about it looks truncated.
    """
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    train_ds = PointerChaseDataset(
        generate_dataset(8, 4, depth=4, seed=1, num_nodes=M, num_maps=K), spec)
    val_ds = PointerChaseDataset(
        generate_dataset(2, 4, depth=4, seed=2, num_nodes=M, num_maps=K), spec)
    model_cfg = ModelConfig(
        architecture="rwkv", hidden_size=64, num_hidden_layers=1,
        num_attention_heads=1, head_dim=64, vocab_size=M,
        rwkv_kernel="reference")
    device = torch.device("cpu")

    def build():
        set_seed(42)
        return create_model(model_cfg, d_input=spec.d_input,
                            compact_reduced_features=False)

    def cfg(epochs):
        return TrainConfig(seed=42, batch_size=4, precision="fp32",
                           num_workers=0, epochs=epochs, learning_rate=1e-3)

    common = dict(depth=4, train_data_seed=1, val_data_seed=2,
                  train_size=8, val_size=2)

    # Uninterrupted reference.
    _, whole = train_model(build(), train_ds, val_ds, spec, model_cfg,
                           cfg(1), device, **common)

    # Same epoch, checkpointed partway and resumed.
    # 32 instances at batch 4 is 8 steps, so a checkpoint every 4 steps puts one
    # exactly halfway through the epoch. The last is the epoch boundary, which
    # is not the case under test.
    ckpt = tmp_path / "ck"
    train_model(build(), train_ds, val_ds, spec, model_cfg, cfg(1), device,
                checkpoint_path=ckpt, checkpoint_every_steps=4, **common)
    steps = sorted(ckpt.glob("step_*.pt"))
    assert len(steps) >= 2, f"expected a mid-epoch checkpoint, got {steps}"

    _, resumed = train_model(build(), train_ds, val_ds, spec, model_cfg,
                             cfg(1), device,
                             resume_from_checkpoint=steps[0], **common)

    got = resumed["epoch_train_losses"][-1]
    want = whole["epoch_train_losses"][-1]
    # The tail-only mean differs from the whole-epoch mean by far more than
    # accumulation order does; this is checking that the whole epoch is covered.
    assert abs(got - want) < 1e-3, (
        f"resumed epoch reported {got:.6f} against {want:.6f} for the same "
        f"epoch of work")


def test_train_model_refuses_a_ragged_bank_under_cudagraphs():
    """The runner is not the only caller, so the guard cannot live only there.

    A ragged final batch raises from inside the graph replay partway through
    the first evaluation, after training has already run. Refusing before the
    first step turns a wasted run into a message.
    """
    spec = ChaseSpec(num_nodes=M, num_maps=K, max_depth=8)
    # 3 memories x 4 queries = 12 instances, which is not a multiple of 8.
    train_ds = PointerChaseDataset(
        generate_dataset(3, 4, depth=4, seed=1, num_nodes=M, num_maps=K), spec)
    val_ds = PointerChaseDataset(
        generate_dataset(2, 4, depth=4, seed=2, num_nodes=M, num_maps=K), spec)
    model_cfg = ModelConfig(
        architecture="rwkv", hidden_size=64, num_hidden_layers=1,
        num_attention_heads=1, head_dim=64, vocab_size=M,
        rwkv_kernel="reference")
    model = create_model(model_cfg, d_input=spec.d_input,
                         compact_reduced_features=False)
    model.backbone = torch.compile(model.backbone)
    train_cfg = TrainConfig(seed=42, batch_size=8, precision="fp32",
                            num_workers=0, epochs=1, learning_rate=1e-3)

    with pytest.raises(ValueError, match="same shape"):
        train_model(model, train_ds, val_ds, spec, model_cfg, train_cfg,
                    torch.device("cpu"), depth=4, train_data_seed=1,
                    val_data_seed=2, train_size=3, val_size=2,
                    compile_backend="cudagraphs")
