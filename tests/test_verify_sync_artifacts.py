import hashlib
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_sync_artifacts.py"

def run_script(args, cwd=None):
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result

def create_artifact_and_sidecar(tmp_path, filename, content=b"hello world", sidecar_fmt="{}  {}", newline="\n"):
    artifact = tmp_path / filename
    artifact.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    sidecar = tmp_path / f"{filename}.sha256"
    sidecar_content = sidecar_fmt.format(digest, filename) + newline
    # Note: writing as binary to strictly control line endings
    sidecar.write_bytes(sidecar_content.encode("utf-8"))
    return artifact, sidecar

def test_matching_lf_sidecar_verifies(tmp_path):
    create_artifact_and_sidecar(tmp_path, "test.txt", newline="\n")
    result = run_script([str(tmp_path)])
    assert result.returncode == 0
    assert "verified" in result.stdout
    assert "test.txt" in result.stdout

def test_matching_crlf_sidecar_verifies(tmp_path):
    # a matching CRLF sidecar verifies (this is the bug that motivated the script; assert it explicitly)
    create_artifact_and_sidecar(tmp_path, "test.txt", newline="\r\n")
    result = run_script([str(tmp_path)])
    assert result.returncode == 0
    assert "verified" in result.stdout
    assert "test.txt" in result.stdout

def test_sidecar_with_asterisk_marker(tmp_path):
    # a sidecar using the `*` binary marker verifies
    create_artifact_and_sidecar(tmp_path, "test.txt", sidecar_fmt="{} *{}")
    result = run_script([str(tmp_path)])
    assert result.returncode == 0
    assert "verified" in result.stdout
    assert "test.txt" in result.stdout

def test_tampered_file_reports_mismatch(tmp_path):
    artifact, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt")
    # tamper the artifact
    artifact.write_bytes(b"tampered")
    result = run_script([str(tmp_path)])
    assert result.returncode != 0
    assert "MISMATCH" in result.stdout
    assert "test.txt" in result.stdout

def test_file_no_sidecar_does_not_fail(tmp_path):
    artifact = tmp_path / "test.txt"
    artifact.write_bytes(b"hello world")
    result = run_script([str(tmp_path)])
    assert result.returncode == 0
    assert "no sidecar" in result.stdout
    assert "test.txt" in result.stdout

def test_orphaned_sidecar_fails(tmp_path):
    artifact, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt")
    artifact.unlink() # remove artifact to orphan the sidecar
    result = run_script([str(tmp_path)])
    assert result.returncode != 0
    assert "orphaned sidecar" in result.stdout
    assert "test.txt.sha256" in result.stdout

def test_different_working_directory(tmp_path):
    # verification succeeds when the script is invoked from a different working
    # directory than the artifacts (this is the other bug; assert it explicitly)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    create_artifact_and_sidecar(artifact_dir, "test.txt")

    work_dir = tmp_path / "workdir"
    work_dir.mkdir()

    # We pass the relative or absolute path of artifact_dir while running from work_dir
    result = run_script([str(artifact_dir)], cwd=work_dir)
    assert result.returncode == 0
    assert "verified" in result.stdout
    assert "test.txt" in result.stdout
