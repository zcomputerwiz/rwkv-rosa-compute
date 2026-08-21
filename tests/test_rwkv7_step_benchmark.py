"""CPU-only contract tests for the RWKV-7 single-step benchmark."""

import pytest

from scripts.benchmark_rwkv7_step import MODES, parse_args, percentile


def test_step_benchmark_modes_cover_eager_and_cudagraph_paths():
    assert MODES == (
        "old_padded_eager",
        "old_padded_cudagraph",
        "step_eager",
        "step_cudagraph",
    )


def test_step_benchmark_percentile_interpolates():
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.1) == pytest.approx(1.3)
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.9) == pytest.approx(3.7)


def test_step_benchmark_requires_positive_iterations(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--iterations", "0", "--output", str(tmp_path / "out.json")])


def test_step_benchmark_profile_requires_one_mode():
    with pytest.raises(SystemExit):
        parse_args(["--profile"])
    args = parse_args(["--profile", "--mode", "step_eager"])
    assert args.output is None
