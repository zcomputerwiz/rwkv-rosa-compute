import json
import sys
from pathlib import Path

# Add the scripts directory to the path so we can import the script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_eval_artifact import main


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def valid_payload():
    return {
        "artifact_kind": "evaluation",
        "schema_version": "1.0",
        "run_id": "test_run_123",
        "seed": 42,
        "evaluated_epoch": 5,
        "task": "test",
        "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "checkpoint": "my_checkpoint.pt",
        "checkpoint_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "settings": {
            "batch_size": 32,
            "precision": "bf16"
        },
        "provenance": {
            "repository_commit": "cccccccccccccccccccccccccccccccccccccccc",
            "producer_script_path": "scripts/evaluate.py",
            "producer_script_git_blob_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "producer_script_hash_basis": "git_blob_at_repository_commit",
            "device": {
                "gpu_name": "RTX 3090",
                "gpu_compute_capability": [8, 6],
                "python_version": "3.12.0",
                "torch_version": "2.4.0",
                "cuda_version": "12.1"
            }
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
    del payload["evaluated_epoch"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "the evaluated epoch" in captured.out

def test_missing_inputs_group(tmp_path, capsys):
    fpath = tmp_path / "missing_hash.json"
    payload = valid_payload()
    del payload["input_sha256"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing challenge or dataset identifier AND its content hash" in captured.out

def test_missing_commit(tmp_path, capsys):
    fpath = tmp_path / "missing_commit.json"
    payload = valid_payload()
    del payload["provenance"]["repository_commit"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "commit" in captured.out
    assert "script hash" not in captured.out

def test_missing_script_hash(tmp_path, capsys):
    fpath = tmp_path / "missing_script_hash.json"
    payload = valid_payload()
    del payload["provenance"]["producer_script_git_blob_sha256"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing script hash" in captured.out
    assert "missing commit" not in captured.out

def test_short_commit_rejected(tmp_path, capsys):
    fpath = tmp_path / "short_commit.json"
    payload = valid_payload()
    payload["provenance"]["repository_commit"] = "12345678"
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "commit" in captured.out

def test_missing_checkpoint_group(tmp_path, capsys):
    fpath = tmp_path / "missing_checkpoint.json"
    payload = valid_payload()
    del payload["checkpoint"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing evaluated checkpoint identifier and 64-hex hash" in captured.out

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
    del payload["provenance"]["device"]["gpu_compute_capability"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing device name and compute capability" in captured.out

def test_evaluation_seed_list_does_not_replace_model_seed(tmp_path, capsys):
    fpath = tmp_path / "alias_payload.json"
    payload = valid_payload()
    # eval_epoch is a valid spelling; an evaluation seed list is not the
    # singular model/training seed this artifact must identify.
    del payload["evaluated_epoch"]
    payload["eval_epoch"] = 5
    del payload["seed"]
    payload["evaluation"] = {"seeds_run": [42]}

    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing seed" in captured.out
    assert "the evaluated epoch" not in captured.out

def test_strict_mode_exit_code(tmp_path):
    fpath = tmp_path / "missing_epoch_strict.json"
    payload = valid_payload()
    del payload["evaluated_epoch"]
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
            "content_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        },
        "canonical_validation": {
            "eval_seed": 9999
        },
        "environment": {
            "gpu": "RTX 3070"
            # Note: compute_capability, python/torch/cuda, and commit are intentionally missing
            # to match old real artifacts as requested
        }
    }

    write_json(fpath, real_payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()

    assert "commit" in captured.out
    assert "script hash" in captured.out
    assert "device name and compute capability" in captured.out
    assert "environment Python, Torch, and CUDA versions" in captured.out

    # Make sure we don't accidentally report other fields missing
    assert "the evaluated epoch" in captured.out
    assert "evaluation batch size and precision" not in captured.out
    assert "challenge or dataset identifier AND its content hash" not in captured.out
    assert "evaluated checkpoint identifier and 64-hex hash" in captured.out

def test_missing_strict_exit_on_malformed(tmp_path):
    fpath = tmp_path / "malformed2.json"
    fpath.write_text("{")
    assert main([str(fpath), "--strict"]) == 1

def test_missing_strict_exit_on_empty_dir(tmp_path):
    assert main([str(tmp_path), "--strict"]) == 1

def test_empty_value_is_missing(tmp_path, capsys):
    fpath = tmp_path / "empty_epoch.json"
    payload = valid_payload()
    payload["evaluated_epoch"] = ""
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "the evaluated epoch" in captured.out

def test_invalid_hash_is_missing(tmp_path, capsys):
    fpath = tmp_path / "invalid_hash.json"
    payload = valid_payload()
    payload["input_sha256"] = "abcdef" # only 6 chars, invalid hash length
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing challenge or dataset identifier AND its content hash" in captured.out

def test_valid_hash_passes(tmp_path, capsys):
    fpath = tmp_path / "valid_hash.json"
    payload = valid_payload()
    payload["input_sha256"] = "a" * 64
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing challenge or dataset identifier AND its content hash" not in captured.out


def test_short_script_hash_rejected(tmp_path, capsys):
    fpath = tmp_path / "short_script.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_git_blob_sha256"] = "12345678"
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing script hash" in captured.out

def test_missing_environment_versions(tmp_path, capsys):
    fpath = tmp_path / "missing_env_versions.json"
    payload = valid_payload()
    del payload["provenance"]["device"]["python_version"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing environment Python, Torch, and CUDA versions" in captured.out

def test_missing_checkpoint_hash(tmp_path, capsys):
    fpath = tmp_path / "missing_checkpoint_hash.json"
    payload = valid_payload()
    del payload["checkpoint_sha256"]
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing evaluated checkpoint identifier and 64-hex hash" in captured.out

def test_short_checkpoint_hash_rejected(tmp_path, capsys):
    fpath = tmp_path / "short_checkpoint_hash.json"
    payload = valid_payload()
    payload["checkpoint_sha256"] = "12345"
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "missing evaluated checkpoint identifier and 64-hex hash" in captured.out

def test_wrapper_produced_integration_fixture(tmp_path, capsys):
    fpath = tmp_path / "wrapper_artifact.json"
    # An artifact shaped like what PR 73 wrapper produces
    payload = {
        "artifact_kind": "evaluation",
        "schema_version": "1.0",
        "run_id": "test_wrapper_123",
        "seed": 42,
        "evaluated_epoch": 5,
        "task": "test",
        "input_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",

        "model": {
            "checkpoint": "my_checkpoint.pt",
            "checkpoint_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        },
        "settings": {
            "batch_size": 32,
            "precision": "bf16"
        },
        "provenance": {
            "repository_commit": "cccccccccccccccccccccccccccccccccccccccc",
            "producer_script_path": "scripts/evaluate.py",
            "producer_script_git_blob_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "producer_script_hash_basis": "git_blob_at_repository_commit",
            "device": {
            "gpu_name": "RTX 3090",
            "gpu_compute_capability": [8, 6],
            "python": "3.12.0",
            "torch": "2.4.0",
            "cuda_version": "12.1",
            "python_version": "3.12.0",
            "torch_version": "2.4.0"
        }
        }
    }
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""

def test_legacy_script_hash_warning(tmp_path, capsys):
    fpath = tmp_path / "legacy.json"
    payload = valid_payload()
    del payload["provenance"]
    payload["script_sha256"] = "a" * 64
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "legacy working-tree script hash (not platform-independent)" in captured.out

def test_script_hash_uppercase_rejected(tmp_path, capsys):
    fpath = tmp_path / "uppercase_hash.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_git_blob_sha256"] = "A" * 64
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "script hash (requires repository-relative path, lowercase 64-hex blob hash, and correct basis string)" in captured.out

def test_script_hash_wrong_basis_rejected(tmp_path, capsys):
    fpath = tmp_path / "wrong_basis.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_hash_basis"] = "something_else"
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "script hash (requires repository-relative path, lowercase 64-hex blob hash, and correct basis string)" in captured.out

def test_script_hash_absolute_path_rejected(tmp_path, capsys):
    fpath = tmp_path / "absolute_path.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_path"] = "/absolute/path/to/script.py"
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    captured = capsys.readouterr()
    assert "script hash (requires repository-relative path, lowercase 64-hex blob hash, and correct basis string)" in captured.out

def test_strict_mode_exits_nonzero_on_empty_artifact(tmp_path):
    fpath = tmp_path / "empty.json"
    write_json(fpath, {})
    assert main([str(fpath), "--strict"]) == 1

def test_script_hash_posix_absolute_rejected(tmp_path, capsys):
    fpath = tmp_path / "posix_abs.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_path"] = "/evaluate.py"
    write_json(fpath, payload)
    assert main([str(fpath)]) == 0
    assert "script hash (requires repository-relative path" in capsys.readouterr().out

def test_script_hash_windows_absolute_rejected(tmp_path, capsys):
    fpath = tmp_path / "win_abs.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_path"] = "C:\\evaluate.py"
    write_json(fpath, payload)
    assert main([str(fpath)]) == 0
    assert "script hash (requires repository-relative path" in capsys.readouterr().out

def test_script_hash_unc_rejected(tmp_path, capsys):
    fpath = tmp_path / "unc_abs.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_path"] = "\\\\server\\share\\evaluate.py"
    write_json(fpath, payload)
    assert main([str(fpath)]) == 0
    assert "script hash (requires repository-relative path" in capsys.readouterr().out

def test_script_hash_traversal_rejected(tmp_path, capsys):
    fpath = tmp_path / "traversal.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_path"] = "../evaluate.py"
    write_json(fpath, payload)
    assert main([str(fpath)]) == 0
    assert "script hash (requires repository-relative path" in capsys.readouterr().out


def test_initialization_checkpoint_does_not_replace_evaluated_checkpoint(
    tmp_path, capsys
):
    fpath = tmp_path / "initialization_only.json"
    payload = valid_payload()
    del payload["checkpoint"]
    del payload["checkpoint_sha256"]
    payload["initialization"] = {
        "checkpoint_path": "base_model.pt",
        "checkpoint_sha256": "e" * 64,
    }
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    assert "evaluated checkpoint identifier" in capsys.readouterr().out


def test_wrong_artifact_kind_is_rejected(tmp_path, capsys):
    fpath = tmp_path / "training.json"
    payload = valid_payload()
    payload["artifact_kind"] = "training"
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    assert "missing artifact kind" in capsys.readouterr().out


def test_invalid_schema_version_is_rejected(tmp_path, capsys):
    fpath = tmp_path / "bad_schema.json"
    payload = valid_payload()
    payload["schema_version"] = False
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    assert "missing schema version" in capsys.readouterr().out


def test_script_hash_drive_relative_path_rejected(tmp_path, capsys):
    fpath = tmp_path / "drive_relative.json"
    payload = valid_payload()
    payload["provenance"]["producer_script_path"] = "C:evaluate.py"
    write_json(fpath, payload)

    assert main([str(fpath)]) == 0
    assert "script hash (requires repository-relative path" in capsys.readouterr().out
