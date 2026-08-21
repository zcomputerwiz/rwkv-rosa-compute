"""DataLoader settings must not change run identity, and must not block resume.

Worker count, prefetch depth, and pinning are execution plumbing: they change
how batches are produced, never what they contain. Treating them as protocol
meant a run could not be given more workers without becoming, by convention, a
different experiment.
"""

import warnings
from dataclasses import asdict

import pytest

from exp0.checkpointing import validate_checkpoint_signature
from exp0.config import (
    DATALOADER_NEUTRAL_FIELDS,
    ModelConfig,
    Task3SumConfig,
    TrainConfig,
    drop_identity_neutral_fields,
)
from exp0.evaluate import compute_run_id

pytestmark = pytest.mark.exp0


def _run_id(**train_overrides):
    train_kwargs = {"epochs": 1, "batch_size": 8, **train_overrides}
    return compute_run_id(
        ModelConfig(architecture="llama", hidden_size=64, num_hidden_layers=1,
                    num_attention_heads=1, intermediate_size=128),
        TrainConfig(**train_kwargs),
        Task3SumConfig(num_samples=32),
        eval_seed=9999,
        val_samples=100,
        seeds_run=[42],
    )


def test_worker_count_does_not_change_run_id():
    """The whole point: more workers is the same experiment."""
    assert _run_id(num_workers=0) == _run_id(num_workers=8)


def test_all_loader_fields_are_neutral():
    baseline = _run_id()
    tuned = _run_id(
        num_workers=6, val_num_workers=3, pin_memory=False, prefetch_factor=8
    )
    assert baseline == tuned


def test_real_protocol_fields_still_change_run_id():
    """The exemption must not leak into anything that alters the objective."""
    baseline = _run_id()
    assert _run_id(batch_size=16) != baseline
    assert _run_id(learning_rate=3e-4) != baseline
    assert _run_id(precision="bf16") != baseline
    assert _run_id(grouped_execution=True) != baseline


def test_normalization_targets_the_dataclass_defaults():
    """A canonical value that is not the default would shift every run_id."""
    defaults = asdict(TrainConfig())
    for key, canonical in DATALOADER_NEUTRAL_FIELDS.items():
        assert defaults[key] == canonical, key


def _signature(**train_overrides):
    train = drop_identity_neutral_fields(asdict(TrainConfig(**train_overrides)))
    return {
        "run_id": "abc123",
        "model": {"architecture": "llama"},
        "training": train,
        "task": {"num_filler": 0},
        "train_dataset_size": 1000,
        "epochs": 5,
    }


def test_resume_tolerates_loader_difference_with_a_warning():
    saved = _signature()
    saved["training"]["num_workers"] = 2      # written before normalization
    expected = _signature()

    with pytest.warns(RuntimeWarning, match="DataLoader settings"):
        validate_checkpoint_signature(saved, expected)


def test_resume_tolerates_run_id_shift_caused_by_that_difference():
    """A pre-normalization checkpoint carries the old hash; accept it."""
    saved = _signature()
    saved["training"]["num_workers"] = 2
    saved["run_id"] = "old_hash_from_before"
    expected = _signature()

    with pytest.warns(RuntimeWarning):
        validate_checkpoint_signature(saved, expected)


def test_run_id_alone_is_not_enough_to_excuse_a_real_change():
    """run_id is exempt only as a consequence of loader fields, never alone."""
    saved = _signature()
    saved["training"]["batch_size"] = 999
    expected = _signature()

    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_signature(saved, expected)


def test_scientific_sections_are_still_enforced():
    for section, value in (
        ("model", {"architecture": "rwkv"}),
        ("task", {"num_filler": 36}),
        ("train_dataset_size", 2000),
        ("epochs", 7),
    ):
        saved = _signature()
        expected = _signature()
        expected[section] = value
        with pytest.raises(ValueError, match="does not match"):
            validate_checkpoint_signature(saved, expected)


def test_mixed_difference_is_rejected_not_partially_excused():
    """Loader drift must not smuggle a protocol change past validation."""
    saved = _signature()
    saved["training"]["num_workers"] = 4
    saved["training"]["learning_rate"] = 0.5
    expected = _signature()

    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_signature(saved, expected)


def test_identical_signature_warns_about_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_checkpoint_signature(_signature(), _signature())
