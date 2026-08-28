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

def create_artifact_and_sidecar(tmp_path, filename, content=b"hello world", sidecar_fmt="{}  {}", newline="\n", custom_digest=None):
    artifact = tmp_path / filename
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(content)
    digest = custom_digest if custom_digest is not None else hashlib.sha256(content).hexdigest()
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

def test_empty_sidecar_reports_malformed(tmp_path):
    artifact = tmp_path / "test.txt"
    artifact.write_bytes(b"payload")
    sidecar = tmp_path / "test.txt.sha256"
    sidecar.write_bytes(b"")
    result = run_script([str(tmp_path)])
    assert result.returncode != 0
    assert "malformed sidecar" in result.stdout

def test_truncated_digest_reports_malformed(tmp_path):
    artifact = tmp_path / "test.txt"
    artifact.write_bytes(b"payload")
    sidecar = tmp_path / "test.txt.sha256"
    sidecar.write_bytes(b"9f86d081")
    result = run_script([str(tmp_path)])
    assert result.returncode != 0
    assert "malformed sidecar" in result.stdout

def test_non_hex_digest_reports_malformed(tmp_path):
    create_artifact_and_sidecar(tmp_path, "test.txt", custom_digest="z" * 64)
    result = run_script([str(tmp_path)])
    assert result.returncode != 0
    assert "malformed sidecar" in result.stdout

def test_unreadable_reports_unreadable(tmp_path):
    artifact, sidecar = create_artifact_and_sidecar(tmp_path, "test.txt")

    # Restrict read permissions to force OSError (unreadable)
    os.chmod(artifact, 0o000)
    try:
        result = run_script([str(tmp_path)])
        assert result.returncode != 0
        assert "unreadable" in result.stdout
    finally:
        os.chmod(artifact, 0o644)

def test_exclude_stversions_default_and_no_default_excludes(tmp_path):
    # Test exclusion logic
    stversions_dir = tmp_path / ".stversions"
    create_artifact_and_sidecar(stversions_dir, "test.txt")

    result = run_script([str(tmp_path)])
    assert result.returncode == 0
    assert "test.txt" not in result.stdout

    # Using no-default-excludes should include it
    result_no_default = run_script(["--no-default-excludes", str(tmp_path)])
    assert result_no_default.returncode == 0
    assert "verified" in result_no_default.stdout
    assert "test.txt" in result_no_default.stdout
