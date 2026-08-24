"""Legacy checkpoints must still load after a config field is retired.

Removing Task3SumConfig.seed (b649f2c) made every previously written checkpoint
unloadable, because _config_from_mapping rejects any field absent from the
dataclass. Nothing caught it: run_id was verified unchanged, the full suite
passed, and the breakage only surfaced when a real checkpoint was evaluated
twenty minutes later. This is that missing check.
"""

import pytest

from exp0.checkpoint_analysis import RETIRED_CONFIG_FIELDS, _config_from_mapping
from exp0.config import Task3SumConfig


def test_retired_field_is_tolerated_and_dropped():
    legacy = {"length": 6, "dimension": 3, "seed": 42, "num_samples": 2_000_000}
    cfg = _config_from_mapping(Task3SumConfig, legacy, "task")
    assert cfg.length == 6 and cfg.num_samples == 2_000_000
    assert not hasattr(cfg, "seed")


def test_unknown_fields_still_raise():
    with pytest.raises(ValueError, match="unsupported fields"):
        _config_from_mapping(Task3SumConfig, {"length": 6, "typo": 1}, "task")


def test_retired_names_are_not_current_fields():
    """A retired name that came back would be silently dropped on load."""
    from dataclasses import fields
    for cls_name, retired in RETIRED_CONFIG_FIELDS.items():
        if cls_name == "Task3SumConfig":
            current = {f.name for f in fields(Task3SumConfig)}
            assert not (retired & current), (
                f"{retired & current} is both retired and current; loading would "
                "drop it instead of using it"
            )
