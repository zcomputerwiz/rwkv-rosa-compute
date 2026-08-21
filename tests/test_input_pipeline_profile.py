"""CPU tests for the input-pipeline profiler."""

import pytest

from scripts.profile_input_pipeline import (
    build_dataset,
    main,
    measure_capacity,
)

pytestmark = pytest.mark.exp0


def _dataset(samples=64, filler_only=False):
    return build_dataset(
        samples=samples, length=6, dimension=3, num_filler=4, seed=1,
        filler_only=filler_only,
    )


def test_capacity_is_derived_from_a_whole_epoch():
    """A median of inter-batch gaps overstates capacity; the epoch total does not."""
    dataset = _dataset()
    row = measure_capacity(dataset, batch_size=16, workers=0, prefetch=2,
                           pin_memory=False)
    assert row["epoch_seconds"] > 0
    expected = len(dataset) / row["epoch_seconds"]
    assert row["capacity_samples_per_second"] == pytest.approx(expected)
    assert row["workers"] == 0
    # Sanity bound: a real single-process loader cannot exceed this on any host
    # we run on, so an epoch-total regression to gap-medians would trip here.
    assert row["capacity_samples_per_second"] < 1_000_000


def test_filler_only_dataset_drops_the_cot_arm():
    mixed = _dataset()
    filler_only = _dataset(filler_only=True)
    assert filler_only.realized_counts["parallel_cot"] == 0
    assert filler_only.realized_counts["filler"] == len(filler_only)
    assert mixed.realized_counts["parallel_cot"] > 0


def test_profiler_runs_end_to_end_without_cuda():
    assert main([
        "--samples", "64", "--workers", "0", "--batch-size", "16",
        "--no-pin-memory", "--demand", "10",
    ]) == 0


def test_components_report_runs():
    assert main([
        "--samples", "64", "--workers", "0", "--batch-size", "16",
        "--no-pin-memory", "--components",
    ]) == 0
