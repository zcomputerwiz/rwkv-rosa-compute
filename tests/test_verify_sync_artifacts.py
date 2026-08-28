import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

# Import the module under test
import importlib.util
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_sync_artifacts.py"
spec = importlib.util.spec_from_file_location("verify_sync_artifacts", SCRIPT_PATH)
verify_module = importlib.util.module_from_spec(spec)
sys.modules["verify_sync_artifacts"] = verify_module
spec.loader.exec_module(verify_module)
from verify_sync_artifacts import classify

def run_script(args, cwd=None):
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result

def create_artifact_and_sidecar(tmp_path, filename, content=b"hello world", sidecar_fmt="{}  {}", newline="\n", custom_digest=None):
    artifact = tmp_path / filename
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(content)
    digest = custom_digest if custom_digest is not None else hashlib.sha256(content).hexdigest()
    sidecar = tmp_path / f"{filename}.sha256"
    sidecar_content = sidecar_fmt.format(digest, filename) + newline
    # Note: writing as binary to strictly control line endings
    sidecar.write_bytes(sidecar_content.encode("utf-8"))
    return str(artifact), str(sidecar)

# Direct classify() tests

def test_matching_lf_sidecar_verifies(tmp_path):
    artifact, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt", newline="\n")
    assert classify(artifact, sidecar, has_artifact=True, has_sidecar=True) == "verified"

def test_matching_crlf_sidecar_verifies(tmp_path):
    artifact, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt", newline="\r\n")
    assert classify(artifact, sidecar, has_artifact=True, has_sidecar=True) == "verified"

def test_sidecar_with_asterisk_marker(tmp_path):
    artifact, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt", sidecar_fmt="{} *{}")
    assert classify(artifact, sidecar, has_artifact=True, has_sidecar=True) == "verified"

def test_tampered_file_reports_mismatch(tmp_path):
    artifact_path, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt")
    Path(artifact_path).write_bytes(b"tampered")
    assert classify(artifact_path, sidecar, has_artifact=True, has_sidecar=True) == "MISMATCH"

def test_file_no_sidecar_does_not_fail(tmp_path):
    artifact = tmp_path / "test.txt"
    artifact.write_bytes(b"hello world")
    sidecar = str(tmp_path / "test.txt.sha256")
    assert classify(str(artifact), sidecar, has_artifact=True, has_sidecar=False) == "no sidecar"

def test_orphaned_sidecar_fails(tmp_path):
    artifact_path, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt")
    assert classify(artifact_path, sidecar, has_artifact=False, has_sidecar=True) == "orphaned sidecar"

def test_empty_sidecar_reports_malformed(tmp_path):
    artifact = tmp_path / "test.txt"
    artifact.write_bytes(b"payload")
    sidecar = tmp_path / "test.txt.sha256"
    sidecar.write_bytes(b"")
    assert classify(str(artifact), str(sidecar), has_artifact=True, has_sidecar=True) == "malformed sidecar"

def test_truncated_digest_reports_malformed(tmp_path):
    artifact = tmp_path / "test.txt"
    artifact.write_bytes(b"payload")
    sidecar = tmp_path / "test.txt.sha256"
    sidecar.write_bytes(b"9f86d081")
    assert classify(str(artifact), str(sidecar), has_artifact=True, has_sidecar=True) == "malformed sidecar"

def test_non_hex_digest_reports_malformed(tmp_path):
    artifact, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt", custom_digest="z" * 64)
    assert classify(artifact, sidecar, has_artifact=True, has_sidecar=True) == "malformed sidecar"

def test_unreadable_reports_unreadable(tmp_path):
    artifact, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt")

    # Mock builtins.open to raise OSError to simulate an unreadable file,
    # ensuring this test is OS-independent (avoids chmod 000 limitations on Windows).
    original_open = open
    def mock_open(*args, **kwargs):
        raise OSError("Mock unreadable error")

    with patch("builtins.open", mock_open):
        assert classify(artifact, sidecar, has_artifact=True, has_sidecar=True) == "unreadable"

# Subprocess tests for CLI contract & directory crawling

def test_different_working_directory(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    create_artifact_and_sidecar(artifact_dir, "test.txt")

    work_dir = tmp_path / "workdir"
    work_dir.mkdir()

    result = run_script([str(artifact_dir)], cwd=work_dir)
    assert result.returncode == 0
    assert "verified" in result.stdout
    assert "test.txt" in result.stdout

def test_exclude_stversions_default_and_no_default_excludes(tmp_path):
    stversions_dir = tmp_path / ".stversions"
    create_artifact_and_sidecar(stversions_dir, "test.txt")

    result = run_script([str(tmp_path)])
    assert result.returncode == 0
    assert "test.txt" not in result.stdout

    result_no_default = run_script(["--no-default-excludes", str(tmp_path)])
    assert result_no_default.returncode == 0
    assert "verified" in result_no_default.stdout
    assert "test.txt" in result_no_default.stdout

def test_cli_json_output(tmp_path):
    create_artifact_and_sidecar(tmp_path, "test.txt")
    result = run_script(["--json", str(tmp_path)])
    assert result.returncode == 0

    data = json.loads(result.stdout)
    assert len(data) == 1
    assert data[0]["status"] == "verified"
    assert "test.txt" in data[0]["file"]
