"""CPU-only contract tests for the RWKV-7 incremental benchmark."""

import pytest

from scripts.benchmark_rwkv7_incremental import MODES, parse_args


def test_incremental_benchmark_modes_cover_old_and_step_paths():
    assert MODES == (
        "old_full_eager",
        "old_full_cudagraph",
        "step_full_eager",
        "step_full_cudagraph",
    )


def test_incremental_benchmark_rejects_invalid_iterations(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--component-iterations",
                "0",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
