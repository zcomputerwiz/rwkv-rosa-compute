#!/usr/bin/env python3
"""Run Experiment 0 single configuration across seeds."""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from exp0.challenge_set import (  # noqa: E402
    ChallengeSpec,
    challenge_set_report,
    generate_challenge_set,
)
from exp0.config import ModelConfig, Task3SumConfig, TrainConfig  # noqa: E402
from exp0.construction_strata import (  # noqa: E402
    build_records,
    diagnose_packed,
    error_records,
    summarize_strata,
)
from exp0.dataset import (  # noqa: E402
    Task3SumDataset,
    build_default_vocab,
    pad_collate_fn,
)
from exp0.evaluate import (  # noqa: E402
    canonical_run_config,
    compile_experiment_report,
    compute_run_id,
)
from exp0.generation import generate_protocol_packed_instances  # noqa: E402
from exp0.rwkv_checkpoint import sha256_file  # noqa: E402
from exp0.task3sum import (  # noqa: E402
    DEFAULT_CORRUPTION_RATE,
    GENERATOR_MODES,
    SOURCE_GENERATOR,
)
from exp0.train import evaluate_accuracy, train_model  # noqa: E402


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Experiment 0 single configuration across seeds"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="llama",
        choices=["llama", "rwkv"],
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        choices=["random", "pretrained"],
        help=(
            "Initialization mode. Llama defaults to random. RWKV requires an "
            "explicit checkpoint for pretrained 0B, or --init random for a "
            "debug-only random run."
        ),
    )
    parser.add_argument(
        "--rwkv_checkpoint",
        type=str,
        default=None,
        help="Explicit path to a stock pretrained RWKV-7 x070 checkpoint",
    )
    parser.add_argument(
        "--rwkv_kernel",
        type=str,
        default="reference",
        choices=["reference", "cuda"],
        help=(
            "RWKV recurrence implementation. 'reference' is the FP32 PyTorch "
            "oracle. 'cuda' uses the pinned upstream BF16 x070 CUDA kernel and "
            "requires --architecture rwkv --precision bf16 --head_dim 64."
        ),
    )
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument(
        "--output_vocab_size",
        type=int,
        default=32000,
        help=(
            "Classifier width. 32000 matches the Llama config used by the "
            "published Match-3 positive control; task input token IDs remain compact."
        ),
    )
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--num_filler", type=int, default=None)
    parser.add_argument(
        "--true_rate",
        type=float,
        default=0.5,
        help="Requested fraction of planted-positive Match-3 examples.",
    )
    parser.add_argument(
        "--generator_mode",
        type=str,
        default=SOURCE_GENERATOR,
        choices=list(GENERATOR_MODES),
        help=(
            "Match-3 data distribution. source_corrupted reproduces the "
            "published planted-positive/geometrically-corrupted-negative setup; "
            "uniform_conditioned preserves the pre-fidelity repository generator."
        ),
    )
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=DEFAULT_CORRUPTION_RATE,
        help="Mean geometric corruption count parameter used by source_corrupted.",
    )
    parser.add_argument(
        "--format_type",
        type=str,
        default=None,
        choices=[
            "parallel_cot",
            "filler",
            "immediate",
            "serial_cot",
            "neutral",
        ],
    )
    parser.add_argument("--parallel_ratio", type=float, default=0.5)
    parser.add_argument("--filler_ratio", type=float, default=0.5)
    parser.add_argument("--serial_ratio", type=float, default=0.0)
    parser.add_argument(
        "--immediate_ratio",
        type=float,
        default=0.0,
        help="Share of training examples using the immediate (N=0) format.",
    )
    parser.add_argument(
        "--neutral_ratio",
        type=float,
        default=0.0,
        help=(
            "Share of training examples using the Experiment 2 neutral-token "
            "control format."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=2000)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=9999,
        help="Fixed eval seed for validation set",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="none",
        choices=["none", "filler_accuracy", "cot_result_nll"],
        help=(
            "Stop once the metric reaches its target. With 'none' (default) "
            "--epochs is the exact budget; otherwise --epochs becomes a ceiling "
            "and the run is no longer fixed-budget if training actually stops early."
        ),
    )
    parser.add_argument(
        "--early_stop_target",
        type=float,
        default=None,
        help=(
            "Override the stop target; valid only when --early_stop_metric is "
            "enabled. Defaults to 1.0 for filler_accuracy or the measured "
            "cot_result_nll_floor for cot_result_nll."
        ),
    )
    parser.add_argument(
        "--early_stop_tolerance",
        type=float,
        default=0.0,
        help=(
            "Absolute slack around the target, e.g. 0.005 stops at >=0.995 "
            "accuracy or at <=floor+0.005 NLL. A small positive tolerance is "
            "recommended for cot_result_nll because measured NLL need not equal "
            "the theoretical floor exactly."
        ),
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=1,
        help=(
            "Consecutive epochs that must meet the target before stopping. "
            "Use 2 to avoid stopping on a single lucky epoch."
        ),
    )
    parser.add_argument("--out_dir", type=str, default="results/exp0")
    parser.add_argument(
        "--checkpoint_every_steps",
        type=int,
        default=5000,
        help=(
            "Write rolling latest.pt checkpoints every N optimizer steps plus "
            "a permanent checkpoint after each completed epoch. Use 0 to disable "
            "automatic checkpoint creation."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Checkpoint root. Defaults to <out_dir>/checkpoints/<run_id>; each "
            "training seed receives its own subdirectory."
        ),
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help=(
            "Resume one training seed from an Experiment 0 latest.pt or "
            "epoch_NNN.pt checkpoint. Requires exactly one --seeds value."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help=(
            "Training/evaluation autocast precision. fp32 preserves the prior "
            "protocol; bf16 is recommended for supported CUDA GPUs."
        ),
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow TF32 for FP32 matmuls. Off by default so --precision fp32 "
            "keeps meaning strict FP32. This is a distinct numerical protocol, "
            "not a free speedup: it is recorded in the report and changes the "
            "run_id. Do not enable it partway through a sweep."
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="torch_compile",
        help=(
            "Compile the training forward with torch.compile. An execution "
            "protocol, recorded in the report and part of the run_id. Requires "
            "a working Triton; see docs/experiment0_precision_and_compile.md."
        ),
    )
    parser.add_argument(
        "--grouped_execution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run each optimizer batch as length-homogeneous subgroups so filler "
            "examples are not carried through a CoT-sized rectangle. One "
            "optimizer update, one scheduler step, and one global gradient clip "
            "are preserved and the loss stays token-weighted, so the objective "
            "is unchanged. This is implementation efficiency, NOT compute "
            "matching: the scientific budget is still the requested N. Recorded "
            "in the report and part of the run_id."
        ),
    )
    parser.add_argument(
        "--construction_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emit supplementary construction-stratum diagnostics for the "
            "canonical validation pass: per-stratum accuracy plus per-instance "
            "error records. Adds no forward pass and does not change "
            "filler_accuracy, the validation distribution, or the run_id."
        ),
    )
    parser.add_argument(
        "--challenge_per_class",
        type=int,
        default=0,
        help=(
            "Also evaluate a deliberately rebalanced diagnostic challenge set "
            "with this many instances per construction stratum (0 disables). "
            "Reported separately from canonical validation and never averaged "
            "with it."
        ),
    )
    parser.add_argument(
        "--challenge_seed",
        type=int,
        default=20260820,
        help="Deterministic seed for the diagnostic challenge set.",
    )
    parser.add_argument(
        "--immediate_protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the published immediate-answer protocol when num_filler is 0 "
            "or the mixture is 'immediate': five times the requested epochs, "
            "weight decay 0.1, grad clip 0.5. Pass --no-immediate_protocol to "
            "train exactly the requested epochs, weight decay, and grad clip, "
            "which is what an N=0 arm needs to be compute-matched against an "
            "N>0 arm. That is a different protocol and changes the run_id."
        ),
    )
    parser.add_argument(
        "--fused_adamw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PyTorch fused AdamW on CUDA (opt-in numerical/kernel change).",
    )
    parser.add_argument(
        "--vocab_reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable vocab reduction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Training DataLoader workers.",
    )
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=0,
        help="Validation workers; zero avoids retaining extra validation workers.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Batches prefetched per training worker when num_workers > 0.",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin DataLoader batches for asynchronous CUDA transfer.",
    )
    return parser


def _resolve_initialization(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None]:
    requested_init = getattr(args, "init", None)
    checkpoint_arg = getattr(args, "rwkv_checkpoint", None)

    if args.architecture == "llama":
        if requested_init not in {None, "random"}:
            raise ValueError(
                "Experiment 0 Llama runs support only random initialization."
            )
        if checkpoint_arg is not None:
            raise ValueError(
                "--rwkv_checkpoint is only valid with --architecture rwkv."
            )
        return "random", None, None

    if args.hidden_size % args.head_dim != 0:
        raise ValueError(
            f"RWKV hidden_size={args.hidden_size} must be divisible by "
            f"head_dim={args.head_dim}."
        )

    if requested_init is None:
        if checkpoint_arg is None:
            raise ValueError(
                "RWKV Experiment 0B requires a stock pretrained checkpoint. "
                "Provide --rwkv_checkpoint PATH (pretrained is inferred), or "
                "use --init random explicitly for a debug-only random run."
            )
        requested_init = "pretrained"

    if requested_init == "random":
        if checkpoint_arg is not None:
            raise ValueError(
                "--init random must not be combined with --rwkv_checkpoint."
            )
        return "random", None, None

    if checkpoint_arg is None:
        raise ValueError(
            "--init pretrained requires --rwkv_checkpoint PATH for RWKV."
        )

    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RWKV-7 checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    return "pretrained", str(checkpoint_path), checkpoint_sha256


def build_configs(
    args: argparse.Namespace,
) -> tuple[Task3SumConfig, ModelConfig, TrainConfig]:
    init_mode, checkpoint_path, checkpoint_sha256 = _resolve_initialization(args)

    task_cfg = Task3SumConfig(
        length=args.length,
        dimension=args.dimension,
        num_filler=(
            args.num_filler if args.num_filler is not None else args.length**2
        ),
        true_rate=args.true_rate,
        num_samples=args.num_samples,
        vocab_reduction=args.vocab_reduction,
        generator_mode=args.generator_mode,
        corruption_rate=args.corruption_rate,
    )

    num_heads = args.num_attention_heads
    if args.architecture == "rwkv":
        num_heads = args.hidden_size // args.head_dim

    model_cfg = ModelConfig(
        architecture=args.architecture,
        init_mode=init_mode,
        rwkv_checkpoint=checkpoint_path,
        rwkv_checkpoint_sha256=checkpoint_sha256,
        rwkv_kernel=args.rwkv_kernel,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=num_heads,
        intermediate_size=args.intermediate_size,
        head_dim=args.head_dim,
        output_vocab_size=args.output_vocab_size,
        device=args.device,
    )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        mixture=(args.format_type if args.format_type else "50_50_cot_filler"),
        parallel_ratio=args.parallel_ratio,
        filler_ratio=args.filler_ratio,
        serial_ratio=args.serial_ratio,
        immediate_ratio=args.immediate_ratio,
        neutral_ratio=args.neutral_ratio,
        early_stop_metric=args.early_stop_metric,
        early_stop_target=args.early_stop_target,
        early_stop_tolerance=args.early_stop_tolerance,
        early_stop_patience=args.early_stop_patience,
        immediate_protocol=args.immediate_protocol,
        tf32_matmul=args.tf32,
        torch_compile=args.torch_compile,
        grouped_execution=args.grouped_execution,
        num_workers=args.num_workers,
        val_num_workers=args.val_num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        precision=args.precision,
        fused_adamw=args.fused_adamw,
    )
    return task_cfg, model_cfg, train_cfg


def get_report_path(
    args: argparse.Namespace,
    task_cfg: Task3SumConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Path:
    run_id = compute_run_id(
        model_cfg,
        train_cfg,
        task_cfg,
        args.eval_seed,
        args.val_samples,
        args.seeds,
    )
    fmt_tag = args.format_type if args.format_type else "mix_50_50"
    filename = (
        f"{args.architecture}_len{args.length}_N{task_cfg.num_filler}_"
        f"fmt_{fmt_tag}_{run_id}.json"
    )
    return Path(args.out_dir) / filename


def _check_existing_report(
    report_path: Path,
    current_run_config: dict,
) -> bool:
    """Return True when an existing report exactly matches the requested run."""
    if not report_path.exists():
        return False

    with open(report_path, encoding="utf-8") as f:
        existing_report = json.load(f)

    existing_config = existing_report.get("run_config")
    if existing_config == current_run_config:
        print(
            f"Report {report_path} already exists and its full run_config "
            "matches. Skipping run."
        )
        return True

    raise ValueError(
        f"Report {report_path} exists but its full run_config does not match "
        "the requested configuration. Will not overwrite."
    )


def _predicted_labels(predicted_ids, vocab):
    """Map answer-position token ids to booleans; None for anything else."""
    true_id = vocab.token2id["True"]
    false_id = vocab.token2id["False"]
    return [
        True if token == true_id else False if token == false_id else None
        for token in predicted_ids
    ]


def _canonical_diagnostics(details, val_instances, vocab, task_cfg):
    """Construction strata for the canonical validation pass.

    Uses per-example data captured during the run's existing final validation
    pass, so no additional forward pass is performed. Supplementary only:
    filler_accuracy remains the canonical metric.
    """
    if not details:
        return None
    diagnostics = diagnose_packed(val_instances, mod=task_cfg.mod)
    records = build_records(
        diagnostics,
        _predicted_labels(details["predicted_ids"], vocab),
        details.get("true_logits"),
        details.get("false_logits"),
    )
    return {
        "distribution": "canonical_source_faithful",
        "distribution_note": (
            "Source-faithful validation distribution. This is the same pass "
            "that produced filler_accuracy; the strata below partition it and "
            "do not replace it."
        ),
        "stratified": summarize_strata(records),
        "errors": error_records(records),
    }


def _challenge_diagnostics(args, task_cfg, vocab, model, train_cfg, model_cfg):
    """Evaluate a rebalanced diagnostic challenge set, reported separately."""
    spec = ChallengeSpec(
        seed=args.challenge_seed,
        per_stratum=args.challenge_per_class,
        length=args.length,
        dimension=args.dimension,
        mod=task_cfg.mod,
        generator_mode=task_cfg.generator_mode,
        corruption_rate=task_cfg.corruption_rate,
    )
    challenge = generate_challenge_set(spec)
    dataset = Task3SumDataset(
        challenge.instances,
        format_type="filler",
        num_filler=task_cfg.num_filler,
        vocab=vocab,
        seed=args.challenge_seed,
        vocab_reduction=args.vocab_reduction,
    )
    device = torch.device(model_cfg.device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )
    sink = {}
    evaluate_accuracy(
        model,
        loader,
        device,
        vocab.token2id["ANS"],
        vocab.token2id["True"],
        vocab.token2id["False"],
        precision=train_cfg.precision,
        detail_sink=sink,
    )
    records = build_records(
        diagnose_packed(challenge.instances, mod=task_cfg.mod),
        _predicted_labels(sink["predicted_ids"], vocab),
        sink.get("true_logits"),
        sink.get("false_logits"),
    )
    report = challenge_set_report(challenge, summarize_strata(records))
    report["errors"] = error_records(records)
    return report


def main():
    args = get_parser().parse_args()
    task_cfg, model_cfg, train_cfg_base = build_configs(args)

    if args.val_samples <= 0:
        raise ValueError("val_samples must be greater than zero.")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be greater than zero.")
    if not args.seeds:
        raise ValueError("At least one training seed is required.")
    if args.checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be non-negative.")
    if args.resume_checkpoint is not None and len(args.seeds) != 1:
        raise ValueError("--resume_checkpoint requires exactly one training seed.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = get_report_path(args, task_cfg, model_cfg, train_cfg_base)
    run_id = compute_run_id(
        model_cfg,
        train_cfg_base,
        task_cfg,
        args.eval_seed,
        args.val_samples,
        args.seeds,
    )
    current_run_config = canonical_run_config(
        model_cfg,
        train_cfg_base,
        task_cfg,
        args.eval_seed,
        args.val_samples,
        args.seeds,
    )
    if _check_existing_report(report_path, current_run_config):
        return

    checkpointing_requested = (
        args.checkpoint_every_steps > 0
        or args.checkpoint_dir is not None
        or args.resume_checkpoint is not None
    )
    checkpoint_root = None
    if checkpointing_requested:
        checkpoint_root = (
            Path(args.checkpoint_dir).expanduser()
            if args.checkpoint_dir is not None
            else out_dir / "checkpoints" / run_id
        )

    vocab = build_default_vocab(
        length=args.length,
        dimension=args.dimension,
        mod=task_cfg.mod,
    )
    if model_cfg.output_vocab_size is not None and model_cfg.output_vocab_size < len(vocab):
        raise ValueError(
            "output_vocab_size must cover the resolved task vocabulary: "
            f"output={model_cfg.output_vocab_size}, task={len(vocab)}."
        )

    val_instances = generate_protocol_packed_instances(
        num_samples=args.val_samples,
        length=args.length,
        dimension=args.dimension,
        mod=task_cfg.mod,
        true_rate=task_cfg.true_rate,
        rng=random.Random(args.eval_seed),
        generator_mode=task_cfg.generator_mode,
        corruption_rate=task_cfg.corruption_rate,
        # Recording provenance does not touch RNG ordering or instance
        # contents, so the canonical validation set is bit-identical either way.
        collect_provenance=args.construction_diagnostics,
    )
    filler_val_ds = Task3SumDataset(
        val_instances,
        format_type="filler",
        num_filler=task_cfg.num_filler,
        vocab=vocab,
        seed=args.eval_seed,
        vocab_reduction=args.vocab_reduction,
    )
    cot_val_ds = Task3SumDataset(
        val_instances,
        format_type="parallel_cot",
        num_filler=task_cfg.num_filler,
        vocab=vocab,
        seed=args.eval_seed,
        vocab_reduction=args.vocab_reduction,
    )

    num_pos = int(val_instances.has_3sum.sum().item())
    majority_baseline = max(
        num_pos,
        len(val_instances) - num_pos,
    ) / len(val_instances)

    per_seed_results = []
    realized_counts_aggregate: dict[str, int] = {}

    for seed in args.seeds:
        train_instances = generate_protocol_packed_instances(
            num_samples=args.num_samples,
            length=args.length,
            dimension=args.dimension,
            mod=task_cfg.mod,
            true_rate=task_cfg.true_rate,
            rng=random.Random(seed),
            generator_mode=task_cfg.generator_mode,
            corruption_rate=task_cfg.corruption_rate,
        )
        train_ds = Task3SumDataset(
            train_instances,
            format_type=args.format_type,
            num_filler=task_cfg.num_filler,
            vocab=vocab,
            seed=seed,
            vocab_reduction=args.vocab_reduction,
            parallel_ratio=args.parallel_ratio,
            filler_ratio=args.filler_ratio,
            serial_ratio=args.serial_ratio,
            immediate_ratio=args.immediate_ratio,
            neutral_ratio=args.neutral_ratio,
        )

        for fmt, count in train_ds.realized_counts.items():
            realized_counts_aggregate[fmt] = realized_counts_aggregate.get(fmt, 0) + count

        train_cfg = TrainConfig(
            seed=seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            mixture=(args.format_type if args.format_type else "50_50_cot_filler"),
            parallel_ratio=args.parallel_ratio,
            filler_ratio=args.filler_ratio,
            serial_ratio=args.serial_ratio,
            immediate_ratio=args.immediate_ratio,
            neutral_ratio=args.neutral_ratio,
            early_stop_metric=args.early_stop_metric,
            early_stop_target=args.early_stop_target,
            early_stop_tolerance=args.early_stop_tolerance,
            early_stop_patience=args.early_stop_patience,
            immediate_protocol=args.immediate_protocol,
            tf32_matmul=args.tf32,
            torch_compile=args.torch_compile,
            grouped_execution=args.grouped_execution,
            num_workers=args.num_workers,
            val_num_workers=args.val_num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            precision=args.precision,
            fused_adamw=args.fused_adamw,
        )

        seed_checkpoint_dir = (
            checkpoint_root / f"seed_{seed}"
            if checkpoint_root is not None
            else None
        )
        resume_checkpoint = (
            args.resume_checkpoint if args.resume_checkpoint is not None else None
        )

        trained_model, history = train_model(
            model_cfg,
            train_cfg,
            task_cfg,
            train_ds,
            filler_val_dataset=filler_val_ds,
            cot_val_dataset=cot_val_ds,
            checkpoint_dir=seed_checkpoint_dir,
            checkpoint_every_steps=args.checkpoint_every_steps,
            resume_checkpoint=resume_checkpoint,
            checkpoint_run_id=run_id,
            collect_validation_details=args.construction_diagnostics,
        )
        history["seed"] = seed
        history["training_seed"] = seed
        history["task_seed"] = seed
        per_seed_results.append(history)

        if args.construction_diagnostics:
            history["construction_diagnostics"] = _canonical_diagnostics(
                history.pop("final_validation_details", None),
                val_instances,
                vocab,
                task_cfg,
            )
        if args.challenge_per_class > 0:
            history["challenge_diagnostics"] = _challenge_diagnostics(
                args, task_cfg, vocab, trained_model, train_cfg, model_cfg
            )

        del trained_model, train_ds, train_instances
        if model_cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = compile_experiment_report(
        model_cfg,
        train_cfg_base,
        task_cfg,
        per_seed_results,
        majority_class_baseline=majority_baseline,
        realized_mixture_counts=realized_counts_aggregate,
        eval_seed=args.eval_seed,
        val_samples=args.val_samples,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {report_path}")
    print(
        f"Mean accuracy: {report['metrics']['mean_accuracy']:.4f} "
        f"(baseline: {majority_baseline:.4f})"
    )


if __name__ == "__main__":
    main()
