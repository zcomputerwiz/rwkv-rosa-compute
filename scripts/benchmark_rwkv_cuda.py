#!/usr/bin/env python3
"""Reproducible RWKV-7 CUDA benchmark plan and execution harness.

All specification, schema, and reporting helpers in this module are CPU-only.
``CudaExecutor`` is the deliberately narrow hardware-dependent boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from exp0.config import ModelConfig  # noqa: E402
from exp0.models.rwkv import RWKV7_OP  # noqa: E402
from exp0.models.rwkv_cuda import (  # noqa: E402
    CHUNK_LEN,
    SUPPORTED_HEAD_DIM,
    rwkv7_cuda_recurrence,
)
from exp0.train import create_model  # noqa: E402

SCHEMA_VERSION = 1
MODES = (
    "fused_forward",
    "fused_forward_backward",
    "reference_forward",
    "reference_forward_backward",
    "full_rwkv_forward",
    "full_rwkv_forward_backward",
)
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32)
DEFAULT_TIMESTEPS = (1, 2, 4, 8, 15, 16, 17, 32, 64, 128)
SMOKE_BATCHES = (1,)
SMOKE_TIMESTEPS = (1, 16, 17)


@dataclass(frozen=True)
class Workload:
    mode: str
    batch: int
    timesteps: int
    hidden_size: int
    head_dim: int
    heads: int
    logical_timesteps: int
    padded_timesteps: int
    padding_timesteps: int
    padding_fraction: float
    logical_transitions: int
    physical_kernel_transitions: int


def workload_accounting(batch: int, timesteps: int, heads: int) -> dict[str, Any]:
    """Return sequence and recurrence work; transition counts include B and heads.

    Logical timesteps are requested tokens. Padded timesteps are the chunk-aligned
    tokens presented to the fused kernel. A transition is one head's recurrent
    state update, hence ``B*T*heads`` (and padded T for physical work).
    """
    padded = math.ceil(timesteps / CHUNK_LEN) * CHUNK_LEN
    padding = padded - timesteps
    return {
        "logical_timesteps": timesteps,
        "padded_timesteps": padded,
        "padding_timesteps": padding,
        "padding_fraction": padding / padded,
        "logical_transitions": batch * timesteps * heads,
        "physical_kernel_transitions": batch * padded * heads,
    }


def validate_dimensions(hidden_size: int, head_dim: int) -> int:
    if hidden_size <= 0 or head_dim <= 0:
        raise ValueError("hidden size and head dim must be positive")
    if hidden_size % head_dim:
        raise ValueError("hidden size must be divisible by head dim")
    return hidden_size // head_dim


def build_matrix(
    modes: Sequence[str], batches: Sequence[int], timesteps: Sequence[int],
    hidden_size: int, head_dim: int,
) -> list[Workload]:
    heads = validate_dimensions(hidden_size, head_dim)
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if not batches or not timesteps or any(x <= 0 for x in (*batches, *timesteps)):
        raise ValueError("batches and timesteps must contain positive integers")
    return [
        Workload(mode, batch, steps, hidden_size, head_dim, heads,
                 **workload_accounting(batch, steps, heads))
        for mode in modes for batch in batches for steps in timesteps
    ]


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(x) for x in values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def calculate_statistics(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("at least one timing sample is required")
    values = [float(x) for x in samples_ms]
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p10_ms": percentile(values, 0.1),
        "p90_ms": percentile(values, 0.9),
    }


def calculate_throughput(workload: Workload, median_ms: float) -> dict[str, float]:
    seconds = median_ms / 1000.0
    if seconds <= 0:
        raise ValueError("median timing must be positive")
    return {
        "samples_per_second": workload.batch / seconds,
        "logical_transitions_per_second": workload.logical_transitions / seconds,
        "physical_transitions_per_second": (
            workload.physical_kernel_transitions / seconds
        ),
    }


def make_result(
    workload: Workload, status: str, *, timings_ms: Sequence[float] | None = None,
    memory_allocated_bytes: int | None = None,
    memory_reserved_bytes: int | None = None, error: str | None = None,
) -> dict[str, Any]:
    if status not in {"success", "oom", "unsupported", "error", "planned"}:
        raise ValueError(f"invalid result status: {status}")
    record = asdict(workload)
    record.update({
        "status": status,
        "timing_samples_ms": list(timings_ms) if timings_ms is not None else None,
        "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None,
        "p10_ms": None, "p90_ms": None, "samples_per_second": None,
        "logical_transitions_per_second": None,
        "physical_transitions_per_second": None,
        "max_memory_allocated_bytes": memory_allocated_bytes,
        "max_memory_reserved_bytes": memory_reserved_bytes,
        "error": error,
    })
    if status == "success":
        stats = calculate_statistics(timings_ms or ())
        record.update(stats)
        record.update(calculate_throughput(workload, stats["median_ms"]))
    return record


def _command_output(command: Sequence[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5,
                                check=False, cwd=None if cwd is None else str(cwd))
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def collect_provenance(cuda_available: bool | None = None) -> dict[str, Any]:
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    gpu: dict[str, Any] = {
        "name": None, "compute_capability": None, "total_vram_bytes": None,
    }
    if available:
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name,
               "compute_capability": f"{props.major}.{props.minor}",
               "total_vram_bytes": props.total_memory}
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except (ImportError, ModuleNotFoundError):
        CUDA_HOME = None
    return {
        # Without cwd this records whatever repository the caller happened to
        # be standing in, which defeats the point of a provenance field.
        "git_commit": _command_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": available,
        "gpu": gpu,
        "cuda_toolkit_detected": CUDA_HOME is not None,
        # Kept for diagnostics, but consumers must not use paths as benchmark identity.
        "cuda_toolkit_path": str(CUDA_HOME) if CUDA_HOME else None,
        "nvcc_version": _command_output(["nvcc", "--version"]),
    }


def document(configuration: dict[str, Any], results: list[dict[str, Any]],
             *, cuda_available: bool | None = None) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION,
            "environment": collect_provenance(cuda_available),
            "benchmark_configuration": configuration, "results": results}


def write_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def execute_safely(workload: Workload, executor: Callable[[Workload], dict[str, Any]]) -> dict[str, Any]:
    try:
        return executor(workload)
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return make_result(workload, "oom", error=str(exc))
    except NotImplementedError as exc:
        return make_result(workload, "unsupported", error=str(exc))
    except Exception as exc:  # per-workload isolation is intentional
        return make_result(workload, "error", error=f"{type(exc).__name__}: {exc}")


def inference_context(backward: bool):
    """Grad-disabling context for forward-only workloads.

    Model parameters always require grad, so eval mode alone still builds an
    autograd graph and saves activations for a backward that never runs. That
    inflates forward-only peak memory above true inference memory and makes the
    forward/forward+backward delta understate backward's real cost.
    """
    return nullcontext() if backward else torch.no_grad()


def check_supported(workload: Workload) -> None:
    """Reject configurations the fused kernel cannot run.

    Raised as NotImplementedError so ``execute_safely`` records "unsupported"
    rather than "error"; an unsupported head dimension is a property of the
    kernel, not a failure of the run.
    """
    uses_fused_kernel = workload.mode.startswith(("fused_", "full_"))
    if uses_fused_kernel and workload.head_dim != SUPPORTED_HEAD_DIM:
        raise NotImplementedError(
            f"fused RWKV-7 kernel supports head_dim={SUPPORTED_HEAD_DIM}, "
            f"got {workload.head_dim}"
        )


class CudaExecutor:
    """Only component that allocates, launches, or synchronizes CUDA work."""

    def __init__(
        self,
        warmups: int,
        iterations: int,
        layers: int = 4,
        full_model_compile_backend: str | None = None,
    ):
        self.warmups, self.iterations, self.layers = warmups, iterations, layers
        self.full_model_compile_backend = full_model_compile_backend

    def _operation(self, spec: Workload) -> Callable[[], None]:
        shape = (spec.batch, spec.timesteps, spec.hidden_size)
        backward = spec.mode.endswith("_backward")
        if spec.mode.startswith(("fused_", "reference_")):
            tensors = [torch.randn(shape, device="cuda", dtype=torch.bfloat16,
                                   requires_grad=backward) for _ in range(6)]
            r, raw_w, k, v, a, b = tensors
            def operation() -> None:
                with inference_context(backward):
                    if spec.mode.startswith("fused_"):
                        out = rwkv7_cuda_recurrence(r, raw_w, k, v, a, b,
                                                    head_dim=spec.head_dim)
                    else:
                        w = -F.softplus(-raw_w.float()) - 0.5
                        out = RWKV7_OP(r, w, k, v, a, b, head_dim=spec.head_dim)
                    if backward:
                        # Training zeroes gradients every step. Accumulating
                        # them here would time a different operation.
                        for tensor in tensors:
                            tensor.grad = None
                        out.backward(torch.ones_like(out), retain_graph=False)
            return operation

        model = create_model(
            ModelConfig(
                architecture="rwkv", hidden_size=spec.hidden_size,
                num_hidden_layers=self.layers,
                intermediate_size=spec.hidden_size * 4,
                head_dim=spec.head_dim, rwkv_kernel="cuda", device="cuda",
            ),
            d_input=spec.hidden_size,
        ).cuda()
        model.train(backward)
        if self.full_model_compile_backend is not None:
            # Each matrix row owns a fresh model and fixed input shape. Clear
            # Dynamo's process-global cache so independent rows do not exhaust
            # its recompile limit and silently fall back to eager execution.
            torch.compiler.reset()
            model = torch.compile(
                model,
                backend=self.full_model_compile_backend,
                fullgraph=False,
                dynamic=False,
            )
        # A zero-length tuple prefix plus T synthetic continuation tokens drives
        # the same InputEmbedWrapper/create_model path used by Experiment 0.
        inputs = torch.empty((spec.batch, 0, spec.hidden_size), device="cuda",
                             dtype=torch.bfloat16)
        targets = torch.randint(0, 256, (spec.batch, spec.timesteps), device="cuda")
        def operation() -> None:
            with inference_context(backward):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(inputs, targets)
                if backward:
                    model.zero_grad(set_to_none=True)
                    out.backward(torch.ones_like(out), retain_graph=False)
        return operation

    def __call__(self, spec: Workload) -> dict[str, Any]:
        check_supported(spec)
        torch.cuda.reset_peak_memory_stats()
        operation = self._operation(spec)
        for _ in range(self.warmups):
            operation()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.iterations)]
        for start, end in zip(starts, ends):
            start.record()
            operation()
            end.record()
        torch.cuda.synchronize()
        samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        result = make_result(
            spec, "success", timings_ms=samples,
            memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        # Full-model modes build a fresh model per matrix cell. Dropping the
        # closure and releasing cached blocks keeps a long matrix from failing
        # its later, larger cells on fragmentation rather than real capacity.
        del operation
        torch.cuda.empty_cache()
        return result

    def profile(self, spec: Workload, iterations: int) -> None:
        check_supported(spec)
        operation = self._operation(spec)
        for _ in range(self.warmups):
            operation()
        torch.cuda.synchronize()
        labels = {
            "fused_forward": "rwkv_fused_forward",
            "fused_forward_backward": "rwkv_fused_forward_backward",
            "reference_forward": "rwkv_reference_forward",
            "reference_forward_backward": "rwkv_reference_forward_backward",
            "full_rwkv_forward": "full_model_forward",
            "full_rwkv_forward_backward": "full_model_forward_backward",
        }
        label = labels[spec.mode]
        if self.full_model_compile_backend is not None and spec.mode.startswith("full_"):
            label += f"_{self.full_model_compile_backend}"
        with torch.cuda.nvtx.range(label):
            for _ in range(iterations):
                operation()
        torch.cuda.synchronize()


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(x <= 0 for x in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--batches", type=parse_int_list)
    parser.add_argument("--batch", type=int, help="single-workload batch size")
    parser.add_argument("--timesteps-list", type=parse_int_list)
    parser.add_argument("--timesteps", type=int, help="single-workload timesteps")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=100)
    parser.add_argument(
        "--full-model-compile-backend",
        choices=("cudagraphs",),
        help="opt-in compiler backend applied only to full-model workloads",
    )
    parser.add_argument("--output", type=Path)
    return parser


def plan_from_args(args: argparse.Namespace) -> tuple[list[Workload], dict[str, Any]]:
    if min(args.warmups, args.iterations, args.layers, args.profile_iterations) <= 0:
        raise ValueError("warmups, iterations, layers, and profile iterations must be positive")
    if (args.batch is None) != (args.timesteps is None):
        raise ValueError("--batch and --timesteps must be supplied together")
    if args.batch is not None and (args.batches or args.timesteps_list):
        raise ValueError("single-workload and matrix dimension options cannot be mixed")
    modes = tuple(args.mode or ("fused_forward",))
    if args.full_model_compile_backend and not any(
        mode.startswith("full_") for mode in modes
    ):
        raise ValueError(
            "--full-model-compile-backend requires at least one full-model mode"
        )
    if args.batch is not None:
        batches, timesteps = (args.batch,), (args.timesteps,)
    elif args.smoke:
        batches, timesteps = SMOKE_BATCHES, SMOKE_TIMESTEPS
    else:
        batches = args.batches or DEFAULT_BATCHES
        timesteps = args.timesteps_list or DEFAULT_TIMESTEPS
    matrix = build_matrix(modes, batches, timesteps, args.hidden_size, args.head_dim)
    if args.profile and len(matrix) != 1:
        raise ValueError("--profile requires one --mode and explicit --batch/--timesteps")
    config = {"modes": list(modes), "batches": list(batches),
              "timesteps": list(timesteps), "hidden_size": args.hidden_size,
              "head_dim": args.head_dim, "heads": args.hidden_size // args.head_dim,
              "chunk_len": CHUNK_LEN, "precision": "bf16", "warmups": args.warmups,
              "iterations": args.iterations, "layers": args.layers,
              "recurrence_backend": "mode-dependent", "profile": args.profile,
              "profile_iterations": args.profile_iterations,
              "full_model_compile_backend": args.full_model_compile_backend,
              "dry_run": args.dry_run}
    return matrix, config


def render(results: Sequence[dict[str, Any]]) -> None:
    print(f"{'mode':29} {'B':>3} {'T':>4} {'pad_T':>6} {'median_ms':>10} "
          f"{'transitions/s':>15} {'VRAM MiB':>10} status")
    for row in results:
        median = "-" if row["median_ms"] is None else f"{row['median_ms']:.3f}"
        rate = row["logical_transitions_per_second"]
        rate_text = "-" if rate is None else f"{rate:.0f}"
        memory = row["max_memory_allocated_bytes"]
        memory_text = "-" if memory is None else f"{memory / 2**20:.1f}"
        print(f"{row['mode']:29} {row['batch']:3d} {row['timesteps']:4d} "
              f"{row['padded_timesteps']:6d} {median:>10} {rate_text:>15} "
              f"{memory_text:>10} {row['status']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        matrix, config = plan_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output
    if args.dry_run:
        results = [make_result(spec, "planned") for spec in matrix]
        payload = document(config, results)
        render(results)
        if output:
            write_json(payload, output)
        return 0
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --dry-run to validate the benchmark plan.")
    executor = CudaExecutor(
        args.warmups,
        args.iterations,
        args.layers,
        full_model_compile_backend=args.full_model_compile_backend,
    )
    if args.profile:
        executor.profile(matrix[0], args.profile_iterations)
        return 0
    results = [execute_safely(spec, executor) for spec in matrix]
    payload = document(config, results)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = REPO_ROOT / "results" / "cuda_benchmarks" / f"rwkv7_{stamp}.json"
    write_json(payload, output)
    render(results)
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
