#!/usr/bin/env python3
"""Compare the padded training recurrence with the persistent B=1/T=1 step."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import torch

from exp0.models.rwkv_cuda import rwkv7_cuda_recurrence, rwkv7_cuda_step

REPO_ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "old_padded_eager",
    "old_padded_cudagraph",
    "step_eager",
    "step_cudagraph",
)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def command_output(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def provenance() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return {
        "git_commit": command_output(("git", "rev-parse", "HEAD")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvcc_version": command_output(("nvcc", "--version")),
        "gpu": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_vram_bytes": properties.total_memory,
        },
    }


def make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7001)
    shape = (1, 1, 768)
    r = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    raw_w = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    return r, raw_w, k, v, a, b


def capture_graph(operation: Callable[[], None]) -> tuple[Callable[[], None], object]:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = operation()

    def replay() -> None:
        graph.replay()

    return replay, captured_output


def build_operation(mode: str) -> tuple[Callable[[], None], torch.Tensor | None]:
    inputs = make_inputs()
    state = None
    if mode.startswith("old_"):

        def base_operation() -> None:
            rwkv7_cuda_recurrence(*inputs, head_dim=64)

    else:
        step_inputs = tuple(tensor[:, 0, :].contiguous() for tensor in inputs)
        state = torch.zeros((1, 12, 64, 64), device="cuda")

        def base_operation() -> None:
            rwkv7_cuda_step(*step_inputs, state)

    base_operation()
    torch.cuda.synchronize()
    if mode.endswith("_cudagraph"):
        operation, captured_output = capture_graph(base_operation)
        # Keep graph-owned outputs alive for the lifetime of the replay closure.
        operation._captured_output = captured_output  # type: ignore[attr-defined]
        if state is not None:
            state.zero_()
            torch.cuda.synchronize()
        return operation, state
    return base_operation, state


def profile_launches(operation: Callable[[], None], iterations: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for _ in range(iterations):
            operation()
            # Kineto can otherwise miss the first very short step when all
            # launches are queued before its GPU activity buffer is active.
            torch.cuda.synchronize()

    cuda_events = [
        event
        for event in trace.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    memcpy_events = [
        event for event in cuda_events if "memcpy" in event.name.lower()
    ]
    memset_events = [
        event for event in cuda_events if "memset" in event.name.lower()
    ]
    kernel_events = [
        event
        for event in cuda_events
        if event not in memcpy_events and event not in memset_events
    ]
    return {
        "profile_iterations": iterations,
        "kernel_launches": len(kernel_events),
        "kernel_launches_per_step": len(kernel_events) / iterations,
        "memcpy_events": len(memcpy_events),
        "memcpy_events_per_step": len(memcpy_events) / iterations,
        "memset_events": len(memset_events),
        "memset_events_per_step": len(memset_events) / iterations,
        "kernel_names": sorted({event.name for event in kernel_events}),
    }


def benchmark_mode(
    mode: str,
    warmups: int,
    iterations: int,
    profile_iterations: int,
) -> dict:
    torch.cuda.empty_cache()
    operation, state = build_operation(mode)
    if state is not None:
        state.zero_()
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    median_ms = statistics.median(samples)
    launch_profile = profile_launches(operation, profile_iterations)
    return {
        "mode": mode,
        "status": "success",
        "batch": 1,
        "logical_timesteps": 1,
        "padded_timesteps": 16 if mode.startswith("old_") else 1,
        "hidden_size": 768,
        "heads": 12,
        "head_dim": 64,
        "state_dtype": "float32" if mode.startswith("step_") else "internal",
        "state_mutation": "in_place" if mode.startswith("step_") else "reset",
        "timing_samples_ms": samples,
        "median_ms": median_ms,
        "p10_ms": percentile(samples, 0.1),
        "p90_ms": percentile(samples, 0.9),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "token_steps_per_second": 1000.0 / median_ms,
        "logical_head_transitions_per_second": 12_000.0 / median_ms,
        "physical_head_transitions_per_second": (
            (192_000.0 if mode.startswith("old_") else 12_000.0) / median_ms
        ),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        **launch_profile,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.warmups, args.iterations, args.profile_iterations) <= 0:
        parser.error("warmups, iterations, and profile iterations must be positive")
    if args.profile and (args.mode is None or len(args.mode) != 1):
        parser.error("--profile requires exactly one --mode")
    if not args.profile and args.output is None:
        parser.error("--output is required unless --profile is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    modes = tuple(args.mode or MODES)
    if args.profile:
        operation, _ = build_operation(modes[0])
        for _ in range(args.warmups):
            operation()
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range(f"rwkv7_{modes[0]}"):
            for _ in range(args.profile_iterations):
                operation()
        torch.cuda.synchronize()
        return 0

    results = []
    for mode in modes:
        try:
            result = benchmark_mode(
                mode,
                args.warmups,
                args.iterations,
                args.profile_iterations,
            )
        except Exception as exc:
            result = {"mode": mode, "status": "error", "error": repr(exc)}
        results.append(result)
        median = result.get("median_ms")
        median_text = "-" if median is None else f"{median:.4f} ms"
        print(f"{mode:24} {median_text:>12} {result['status']}")

    payload = {
        "schema_version": 1,
        "environment": provenance(),
        "configuration": {
            "modes": list(modes),
            "warmups": args.warmups,
            "iterations": args.iterations,
            "profile_iterations": args.profile_iterations,
        },
        "results": results,
    }
    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"JSON: {args.output}")
    return 1 if any(result["status"] != "success" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
