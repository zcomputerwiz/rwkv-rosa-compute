import json
import pytest
from pathlib import Path
import sys

# Add the scripts directory to the path so we can import the script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_eval_artifact import main

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def valid_payload():
    return {
        "run_id": "test_run_123",
        "seed": 42,
        "epochs": 5,
        "task_config": {"name": "test"},
        "input_sha256": "abcdef",
        "commit": "1234567",
        "checkpoint": "my_checkpoint.pt",
        "settings": {
            "batch_size": 32,
            "precision": "bf16"
        },
        "environment": {
            "gpu": "RTX 3090",
            "capability": [8, 6]
        }
    }

def test_complete_artifact_passes(tmp_path, capsys):
    fpath = tmp_path / "valid.json"
    write_json(fpath, valid_payload())

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

def test_missing_identity_group(tmp_path, capsys):
    fpath = tmp_path / "missing_epoch.json"
    payload = valid_payload()
    del payload["epochs"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing the evaluated epoch" in captured.out

def test_missing_inputs_group(tmp_path, capsys):
    fpath = tmp_path / "missing_hash.json"
    payload = valid_payload()
    del payload["input_sha256"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing challenge or dataset identifier AND its content hash" in captured.out

def test_missing_code_group(tmp_path, capsys):
    fpath = tmp_path / "missing_commit.json"
    payload = valid_payload()
    del payload["commit"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing commit and/or script hash" in captured.out

def test_missing_checkpoint_group(tmp_path, capsys):
    fpath = tmp_path / "missing_checkpoint.json"
    payload = valid_payload()
    del payload["checkpoint"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing checkpoint identifier or hash" in captured.out

def test_missing_settings_group(tmp_path, capsys):
    fpath = tmp_path / "missing_batch_size.json"
    payload = valid_payload()
    del payload["settings"]["batch_size"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing evaluation batch size and precision" in captured.out

def test_missing_device_group(tmp_path, capsys):
    fpath = tmp_path / "missing_capability.json"
    payload = valid_payload()
    del payload["environment"]["capability"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing device name and compute capability" in captured.out

def test_alias_spelling_accepted(tmp_path, capsys):
    fpath = tmp_path / "alias_payload.json"
    payload = valid_payload()
    # Change "epochs" to "eval_epoch" (which is in our ALIASES)
    del payload["epochs"]
    payload["eval_epoch"] = 5
    # Change "seed" to nested "evaluation.seeds_run"
    del payload["seed"]
    payload["evaluation"] = {"seeds_run": [42]}

    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""

def test_strict_mode_exit_code(tmp_path):
    fpath = tmp_path / "missing_epoch_strict.json"
    payload = valid_payload()
    del payload["epochs"]
    write_json(fpath, payload)

    assert main([str(fpath), "--strict"]) == 1

def test_strict_mode_success_exit_code(tmp_path):
    fpath = tmp_path / "valid_strict.json"
    write_json(fpath, valid_payload())

    assert main([str(fpath), "--strict"]) == 0

def test_non_eval_json(tmp_path, capsys):
    fpath = tmp_path / "non_eval.json"
    write_json(fpath, {"just": "some", "data": 123})

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "not an eval artifact" in captured.out
    assert "missing" not in captured.out

def test_malformed_json(tmp_path, capsys):
    fpath = tmp_path / "malformed.json"
    fpath.write_text("{ this is not valid json }")

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "Failed to parse JSON" in captured.out

def test_missing_run_id_group(tmp_path, capsys):
    fpath = tmp_path / "missing_run_id.json"
    payload = valid_payload()
    del payload["run_id"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing run id" in captured.out

def test_real_artifact_shape(tmp_path, capsys):
    fpath = tmp_path / "real_eval_artifact.json"

    # Realistic shape based on prompt feedback
    real_payload = {
        "run_id": "test_real_123",
        "seed": 43,
        "epochs": 5,
        "checkpoint": "n0_checkpoint.pt",
        "evaluation_settings": {
            "batch_size": 128,
            "precision": "bf16"
        },
        "structural_challenge": {
            "challenge_id": "challenge_test_0",
            "content_sha256": "abcd1234abcd"
        },
        "canonical_validation": {
            "eval_seed": 9999
        },
        "environment": {
            "gpu": "RTX 3070"
            # Note: compute_capability and commit are intentionally missing
            # to match real artifacts as requested
        }
    }

    write_json(fpath, real_payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()

    assert "missing commit and/or script hash" in captured.out
    assert "device name and compute capability" in captured.out

    # Make sure we don't accidentally report other fields missing
    assert "the evaluated epoch" not in captured.out
    assert "evaluation batch size and precision" not in captured.out
    assert "challenge or dataset identifier AND its content hash" not in captured.out
    assert "checkpoint identifier or hash" not in captured.out
