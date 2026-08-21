"""CPU tests for the standalone RWKV-7 CUDA benchmark apparatus."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.benchmark_rwkv_cuda import (
    DEFAULT_BATCHES,
    DEFAULT_TIMESTEPS,
    SCHEMA_VERSION,
    build_matrix,
    build_parser,
    calculate_statistics,
    calculate_throughput,
    collect_provenance,
    document,
    execute_safely,
    make_result,
    plan_from_args,
    workload_accounting,
)


@pytest.mark.parametrize(
    ("timesteps", "padded", "padding"),
    [(1, 16, 15), (15, 16, 1), (16, 16, 0), (17, 32, 15)],
)
def test_padding_boundaries(timesteps, padded, padding):
    work = workload_accounting(2, timesteps, 12)
    assert work["padded_timesteps"] == padded
    assert work["padding_timesteps"] == padding
    assert work["logical_transitions"] == 2 * timesteps * 12
    assert work["physical_kernel_transitions"] == 2 * padded * 12
    assert work["padding_fraction"] == padding / padded


def test_default_matrix_contains_all_combinations_and_boundaries():
    args = build_parser().parse_args([])
    matrix, _ = plan_from_args(args)
    assert len(matrix) == len(DEFAULT_BATCHES) * len(DEFAULT_TIMESTEPS)
    assert {(row.batch, row.timesteps) for row in matrix} == {
        (batch, timesteps)
        for batch in DEFAULT_BATCHES
        for timesteps in DEFAULT_TIMESTEPS
    }
    assert {1, 15, 16, 17}.issubset({row.timesteps for row in matrix})


def test_smoke_and_explicit_single_workload_cli():
    parser = build_parser()
    smoke, _ = plan_from_args(parser.parse_args(["--smoke"]))
    assert [(x.batch, x.timesteps) for x in smoke] == [(1, 1), (1, 16), (1, 17)]
    single, _ = plan_from_args(parser.parse_args([
        "--mode", "fused_forward_backward", "--batch", "16",
        "--timesteps", "64", "--profile",
    ]))
    assert len(single) == 1
    assert (single[0].batch, single[0].timesteps) == (16, 64)


def test_full_model_compile_backend_is_explicit_and_recorded():
    parser = build_parser()
    matrix, config = plan_from_args(
        parser.parse_args(
            [
                "--mode",
                "full_rwkv_forward",
                "--batch",
                "1",
                "--timesteps",
                "1",
                "--full-model-compile-backend",
                "cudagraphs",
            ]
        )
    )
    assert len(matrix) == 1
    assert config["full_model_compile_backend"] == "cudagraphs"


def test_full_model_compile_backend_rejects_a_silent_noop():
    parser = build_parser()
    args = parser.parse_args(
        ["--smoke", "--full-model-compile-backend", "cudagraphs"]
    )
    with pytest.raises(ValueError, match="requires at least one full-model mode"):
        plan_from_args(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["--hidden-size", "65"],
        ["--batch", "1"],
        ["--batch", "0", "--timesteps", "1"],
        ["--warmups", "0"],
        ["--profile", "--smoke"],
    ],
)
def test_invalid_cli_dimensions(argv):
    with pytest.raises(ValueError):
        plan_from_args(build_parser().parse_args(argv))


def test_statistics_use_linear_interpolated_percentiles():
    stats = calculate_statistics([1, 2, 3, 4, 5])
    assert stats == {
        "mean_ms": 3.0,
        "median_ms": 3.0,
        "min_ms": 1.0,
        "max_ms": 5.0,
        "p10_ms": 1.4,
        "p90_ms": 4.6,
    }


def test_throughput_calculation():
    workload = build_matrix(["fused_forward"], [2], [15], 768, 64)[0]
    rates = calculate_throughput(workload, 2.0)
    assert rates["samples_per_second"] == 1000
    assert rates["logical_transitions_per_second"] == 180_000
    assert rates["physical_transitions_per_second"] == 192_000


@pytest.mark.parametrize("status", ["success", "oom", "unsupported", "error"])
def test_result_status_schema_json_round_trip(status):
    workload = build_matrix(["fused_forward"], [1], [17], 64, 64)[0]
    kwargs = {"timings_ms": [1.0, 2.0]} if status == "success" else {"error": status}
    record = make_result(workload, status, **kwargs)
    decoded = json.loads(json.dumps(document({"test": True}, [record], cuda_available=False)))
    assert decoded["schema_version"] == SCHEMA_VERSION
    assert decoded["results"][0]["status"] == status
    assert "max_memory_allocated_bytes" in decoded["results"][0]
    assert decoded["environment"]["cuda_available"] is False


def test_cpu_only_provenance_has_null_gpu_measurements():
    provenance = collect_provenance(cuda_available=False)
    assert provenance["cuda_available"] is False
    assert provenance["gpu"]["name"] is None
    assert provenance["gpu"]["total_vram_bytes"] is None


def test_simulated_oom_and_ordinary_failure_are_distinct(monkeypatch):
    workload = build_matrix(["fused_forward"], [1], [1], 64, 64)[0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def oom(_):
        raise torch.cuda.OutOfMemoryError("simulated")

    def error(_):
        raise RuntimeError("not an oom")

    assert execute_safely(workload, oom)["status"] == "oom"
    result = execute_safely(workload, error)
    assert result["status"] == "error"
    assert "RuntimeError" in result["error"]


def test_dry_run_script_entry_point_writes_plan(tmp_path):
    output = tmp_path / "plan.json"
    script = Path(__file__).parents[1] / "scripts" / "benchmark_rwkv_cuda.py"
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--smoke", "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert len(payload["results"]) == 3
    assert all(row["status"] == "planned" for row in payload["results"])
    assert "fused_forward" in result.stdout


def test_cuda_unavailable_fails_clearly(monkeypatch, capsys):
    from scripts import benchmark_rwkv_cuda

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        benchmark_rwkv_cuda.main(["--smoke"])
    assert exc.value.code == 2
    assert "CUDA is unavailable" in capsys.readouterr().err


def test_forward_only_disables_autograd_but_backward_does_not():
    """Forward-only must not save activations for a backward that never runs.

    Model parameters always require grad, so eval mode alone leaves the graph
    intact and inflates forward-only peak memory.
    """
    from scripts.benchmark_rwkv_cuda import inference_context

    probe = torch.zeros(2, requires_grad=True)
    with inference_context(backward=False):
        assert not (probe * 2).requires_grad
    with inference_context(backward=True):
        assert (probe * 2).requires_grad


@pytest.mark.parametrize(
    "mode", ["fused_forward", "fused_forward_backward", "full_rwkv_forward"]
)
def test_unsupported_head_dim_is_reported_as_unsupported(mode):
    from scripts.benchmark_rwkv_cuda import check_supported

    workload = build_matrix([mode], [1], [16], 128, 32)[0]
    with pytest.raises(NotImplementedError):
        check_supported(workload)

    result = execute_safely(workload, check_supported)
    assert result["status"] == "unsupported"
    assert "head_dim" in result["error"]


def test_reference_modes_accept_any_head_dim():
    """Only the fused kernel is head-dim constrained; the reference is not."""
    from scripts.benchmark_rwkv_cuda import check_supported

    check_supported(build_matrix(["reference_forward"], [1], [16], 128, 32)[0])
    check_supported(build_matrix(["fused_forward"], [1], [16], 128, 64)[0])


def test_provenance_records_this_repository_not_the_caller_directory(
    monkeypatch, tmp_path
):
    import subprocess as sp

    expected = sp.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    monkeypatch.chdir(tmp_path)
    assert collect_provenance(cuda_available=False)["git_commit"] == expected
