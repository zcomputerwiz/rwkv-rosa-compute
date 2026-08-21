#!/usr/bin/env python3
"""Benchmark full-model RWKV-7 B=1/T=1 sequence and persistent-step paths."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from exp0.config import ModelConfig
from exp0.models.rwkv_cuda import rwkv7_cuda_recurrence
from exp0.train import create_model

try:
    from scripts.benchmark_rwkv7_step import percentile, profile_launches, provenance
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from benchmark_rwkv7_step import percentile, profile_launches, provenance

MODES = (
    "old_full_eager",
    "old_full_cudagraph",
    "step_full_eager",
    "step_full_cudagraph",
)


def make_model(layers: int):
    torch.manual_seed(7101)
    return (
        create_model(
            ModelConfig(
                architecture="rwkv",
                rwkv_kernel="cuda",
                hidden_size=768,
                num_hidden_layers=layers,
                intermediate_size=3072,
                head_dim=64,
                vocab_size=256,
                device="cuda",
            ),
            d_input=768,
        )
        .cuda()
        .bfloat16()
        .eval()
    )


def capture_graph(operation: Callable[[], object]) -> Callable[[], object]:
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

    def replay():
        graph.replay()
        return captured_output

    return replay


def zero_step_state(state) -> None:
    for layer_state in state:
        layer_state.time_mix_previous.zero_()
        layer_state.channel_mix_previous.zero_()
        layer_state.recurrence.zero_()


def build_full_operation(mode: str, layers: int):
    model = make_model(layers)
    target = torch.tensor([17], device="cuda")
    tuples = torch.empty((1, 0, 768), device="cuda", dtype=torch.bfloat16)
    state = None
    if mode.startswith("old_"):

        def base_operation():
            return model(tuples, target[:, None])

    else:
        state = model.init_rwkv_step_state(activation_dtype=torch.bfloat16)

        def base_operation():
            return model.rwkv_step(target, state)[0]

    base_operation()
    torch.cuda.synchronize()
    operation = base_operation
    if mode.endswith("_cudagraph"):
        operation = capture_graph(base_operation)
        if state is not None:
            zero_step_state(state)
            torch.cuda.synchronize()
    return operation, model, state, target, tuples


def statistics_record(samples: Sequence[float]) -> dict:
    median_ms = statistics.median(samples)
    return {
        "timing_samples_ms": list(samples),
        "median_ms": median_ms,
        "p10_ms": percentile(samples, 0.1),
        "p90_ms": percentile(samples, 0.9),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "token_steps_per_second": 1000.0 / median_ms,
    }


def time_operation(
    operation: Callable[[], object],
    warmups: int,
    iterations: int,
) -> dict:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    return statistics_record(
        [start.elapsed_time(end) for start, end in zip(starts, ends)]
    )


def benchmark_mode(
    mode: str,
    layers: int,
    warmups: int,
    iterations: int,
    profile_iterations: int,
) -> dict:
    torch.cuda.empty_cache()
    operation, model, state, target, tuples = build_full_operation(mode, layers)
    if state is not None:
        zero_step_state(state)
    torch.cuda.reset_peak_memory_stats()
    timing = time_operation(operation, warmups, iterations)
    launches = profile_launches(operation, profile_iterations)
    result = {
        "mode": mode,
        "status": "success",
        "batch": 1,
        "timesteps": 1,
        "hidden_size": 768,
        "heads": 12,
        "head_dim": 64,
        "layers": layers,
        "parameter_dtype": "bfloat16",
        "state": "persistent" if state is not None else "reset_per_sequence",
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        **timing,
        **launches,
    }
    # Keep graph inputs, model, and state alive through profiling.
    _ = model, target, tuples
    return result


def component_operations(model, target, tuples) -> dict[str, Callable[[], object]]:
    layer = model.backbone.layers[0]
    time_mix = layer.time_mix
    x = model._target_hidden(target).view(1, 1, 768)
    normalized = layer.ln1(layer.ln0(x))
    xx = time_mix.time_shift(normalized) - normalized
    xr = normalized + xx * time_mix.x_r
    xw = normalized + xx * time_mix.x_w
    xk = normalized + xx * time_mix.x_k
    xv = normalized + xx * time_mix.x_v
    xa = normalized + xx * time_mix.x_a
    xg = normalized + xx * time_mix.x_g
    r = time_mix.receptance(xr)
    raw_w = time_mix.w0 + torch.tanh(xw @ time_mix.w1) @ time_mix.w2
    k = time_mix.key(xk)
    v = time_mix.value(xv)
    a = torch.sigmoid(time_mix.a0 + (xa @ time_mix.a1) @ time_mix.a2)
    g = torch.sigmoid(xg @ time_mix.g1) @ time_mix.g2
    kk = k * time_mix.k_k
    kk = F.normalize(kk.view(1, 1, 12, 64), dim=-1, p=2.0).view(1, 1, 768)
    adjusted_k = k * (1 + (a - 1) * time_mix.k_a)
    recurrence_out = rwkv7_cuda_recurrence(
        r,
        raw_w,
        adjusted_k,
        v,
        -kk,
        kk * a,
        head_dim=64,
    )

    def input_projection():
        return model._target_hidden(target)

    def time_mix_projections():
        local_xx = time_mix.time_shift(normalized) - normalized
        local_xr = normalized + local_xx * time_mix.x_r
        local_xw = normalized + local_xx * time_mix.x_w
        local_xk = normalized + local_xx * time_mix.x_k
        local_xv = normalized + local_xx * time_mix.x_v
        local_xa = normalized + local_xx * time_mix.x_a
        local_xg = normalized + local_xx * time_mix.x_g
        local_r = time_mix.receptance(local_xr)
        local_raw_w = (
            time_mix.w0 + torch.tanh(local_xw @ time_mix.w1) @ time_mix.w2
        )
        local_k = time_mix.key(local_xk)
        local_v = time_mix.value(local_xv)
        local_a = torch.sigmoid(
            time_mix.a0 + (local_xa @ time_mix.a1) @ time_mix.a2
        )
        local_g = torch.sigmoid(local_xg @ time_mix.g1) @ time_mix.g2
        local_kk = F.normalize(
            (local_k * time_mix.k_k).view(1, 1, 12, 64),
            dim=-1,
            p=2.0,
        ).view(1, 1, 768)
        return local_r, local_raw_w, local_k, local_v, local_a, local_g, local_kk

    def recurrence():
        return rwkv7_cuda_recurrence(
            r,
            raw_w,
            adjusted_k,
            v,
            -kk,
            kk * a,
            head_dim=64,
        )

    def post_recurrence():
        local_out = time_mix.ln_x(recurrence_out.view(1, 768)).view(1, 1, 768)
        rkv = (
            r.view(1, 1, 12, 64)
            * adjusted_k.view(1, 1, 12, 64)
            * time_mix.r_k
        ).sum(dim=-1, keepdim=True)
        local_out = local_out + (rkv * v.view(1, 1, 12, 64)).view(1, 1, 768)
        return time_mix.output(local_out * g)

    return {
        "input_embedding_projection": input_projection,
        "layer_norm": lambda: layer.ln1(x),
        "time_mix_input_projections": time_mix_projections,
        "padded_fused_recurrence": recurrence,
        "time_mix_post_recurrence": post_recurrence,
        "channel_mix_ffn": lambda: layer.channel_mix(x),
        "one_full_layer": lambda: layer(x, None),
        "backbone": lambda: model.backbone(x),
        "output_head": lambda: model.head(x.view(1, 768)),
        "full_model": lambda: model(tuples, target[:, None]),
    }


def benchmark_components(model, target, tuples, warmups, iterations) -> list[dict]:
    results = []
    for name, base_operation in component_operations(model, target, tuples).items():
        base_operation()
        torch.cuda.synchronize()
        operation = capture_graph(base_operation)
        timing = time_operation(operation, warmups, iterations)
        results.append(
            {
                "component": name,
                "execution": "standalone_cudagraph",
                "status": "success",
                **timing,
            }
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--component-iterations", type=int, default=500)
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    values = (
        args.layers,
        args.warmups,
        args.iterations,
        args.profile_iterations,
        args.component_iterations,
    )
    if min(values) <= 0:
        parser.error("all iteration counts and layers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    modes = tuple(args.mode or MODES)
    results = []
    for mode in modes:
        try:
            result = benchmark_mode(
                mode,
                args.layers,
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

    component_results = []
    if not args.skip_components:
        model = make_model(args.layers)
        target = torch.tensor([17], device="cuda")
        tuples = torch.empty((1, 0, 768), device="cuda", dtype=torch.bfloat16)
        component_results = benchmark_components(
            model,
            target,
            tuples,
            min(args.warmups, 20),
            args.component_iterations,
        )

    payload = {
        "schema_version": 1,
        "environment": provenance(),
        "configuration": {
            "modes": list(modes),
            "layers": args.layers,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "profile_iterations": args.profile_iterations,
            "component_iterations": args.component_iterations,
            "component_note": (
                "Standalone CUDA Graph timings are non-additive because every "
                "component includes one graph replay launch."
            ),
        },
        "results": results,
        "component_results": component_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"JSON: {args.output}")
    return 1 if any(result["status"] != "success" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
