import torch

from exp0.config import ModelConfig, TrainConfig
from exp0.train import create_model
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
    )

    assert len(history["epoch_train_losses"]) == 1
    assert len(history["epoch_val_accuracies"]) == 1
    assert "best_val_accuracy" in history
