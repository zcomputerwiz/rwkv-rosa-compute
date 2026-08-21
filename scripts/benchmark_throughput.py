import random
import sys
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exp0.config import ModelConfig, Task3SumConfig, TrainConfig
from exp0.dataset import Task3SumDataset, build_default_vocab
from exp0.generation import generate_protocol_packed_instances
from exp0.train import train_model

def benchmark(n, samples=7680):
    task_cfg = Task3SumConfig(length=6, dimension=3, num_filler=n, num_samples=samples, vocab_reduction=True)
    model_cfg = ModelConfig(
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        intermediate_size=1536,
        output_vocab_size=32000,
        device="cuda",
    )
    train_cfg = TrainConfig(
        epochs=1,
        batch_size=384,
        learning_rate=1e-4,
        precision="bf16",
        immediate_protocol=False,
        early_stop_metric="none",
        seed=43,
    )
    vocab = build_default_vocab(6, 3)
    train_inst = generate_protocol_packed_instances(
        num_samples=samples,
        length=6,
        dimension=3,
        mod=10,
        true_rate=0.5,
        rng=random.Random(43),
        generator_mode="source_corrupted",
        corruption_rate=1.33333,
    )
    val_inst = generate_protocol_packed_instances(
        num_samples=384,
        length=6,
        dimension=3,
        mod=10,
        true_rate=0.5,
        rng=random.Random(9999),
        generator_mode="source_corrupted",
        corruption_rate=1.33333,
    )
    train_ds = Task3SumDataset(train_inst, None, n, vocab, True, parallel_ratio=0.5, filler_ratio=0.5)
    val_ds = Task3SumDataset(val_inst, "filler", n, vocab, True)

    t0 = time.perf_counter()
    metrics = train_model(model_cfg, train_cfg, task_cfg, train_ds, val_ds)
    t1 = time.perf_counter()
    sps = samples / (t1 - t0)
    print(f"N={n:2d}: {sps:8.1f} samples/sec (took {t1-t0:.2f}s for {samples} samples)")

benchmark(0)
benchmark(36)
