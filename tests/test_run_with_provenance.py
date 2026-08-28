import json
import subprocess
import sys
from unittest.mock import patch

import pytest

# We need to run the script via subprocess to fully test it,
# but we can't easily inject a mock via subprocess.
# Wait, we can patch it by modifying the environment or by mocking the get_environment_info function if we run main() directly,
# but main() relies on sys.argv and exit().
# Let's import main from scripts.run_with_provenance and run it within the test process,
# patching sys.argv.
# However, if we do that, we need to change CWD to the temporary git repo.
import scripts.run_with_provenance as run_with_provenance


class NonSerializable:
    pass

class BuildCapabilities:
    def __init__(self):
        self.variant = "test"

@pytest.fixture
def repo_dir(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    return tmp_path

def test_forced_serialization_failure_leaves_artifact_untouched(repo_dir, monkeypatch):
    artifact = repo_dir / "out.json"
    producer = repo_dir / "producer.py"
    producer.write_text("import json, sys\nwith open(sys.argv[1], 'w') as f:\n  json.dump({'original': 'content'}, f)", encoding="utf-8")

    subprocess.run(["git", "add", "producer.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    monkeypatch.chdir(repo_dir)
    monkeypatch.setattr(sys, "argv", [
        "run_with_provenance.py",
        "--artifact", "out.json",
        "--producer", "producer.py",
        "--",
        sys.executable, "producer.py", "out.json"
    ])

    # Force json.dumps to fail by mocking get_environment_info
    with patch("scripts.run_with_provenance.get_environment_info") as mock_env:
        mock_env.return_value = {
            "python_version": NonSerializable(),
            "torch_version": "1.0",
            "cuda_available": False,
        }

        with pytest.raises(SystemExit) as exc:
            run_with_provenance.main()

        assert exc.value.code != 0

    # verify artifact is intact
    assert artifact.exists()
    assert json.loads(artifact.read_text()) == {"original": "content"}

    # verify no sidecar
    assert not (repo_dir / "out.json.sha256").exists()

def test_git_blob_digest_stable_across_line_endings(repo_dir, monkeypatch):
    producer = repo_dir / "producer.py"

    # Commit with LF
    content_lf = b"print('hello')\nprint('world')\n"
    producer.write_bytes(content_lf)

    subprocess.run(["git", "add", "producer.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    monkeypatch.chdir(repo_dir)
    commit = run_with_provenance.get_git_commit()
    blob_sha256_lf = run_with_provenance.get_git_blob_sha256(commit, "producer.py")

    # Rewrite working tree with CRLF
    content_crlf = b"print('hello')\r\nprint('world')\r\n"
    producer.write_bytes(content_crlf)

    # Blob should still be LF because we ask git for the blob from the commit
    blob_sha256_crlf = run_with_provenance.get_git_blob_sha256(commit, "producer.py")

    assert blob_sha256_lf == blob_sha256_crlf

def test_build_capabilities_survives_dumps(repo_dir, monkeypatch):
    artifact = repo_dir / "out.json"
    producer = repo_dir / "producer.py"
    producer.write_text("import json, sys\nwith open(sys.argv[1], 'w') as f:\n  json.dump({'original': 'content'}, f)", encoding="utf-8")

    subprocess.run(["git", "add", "producer.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    monkeypatch.chdir(repo_dir)
    monkeypatch.setattr(sys, "argv", [
        "run_with_provenance.py",
        "--artifact", "out.json",
        "--producer", "producer.py",
        "--",
        sys.executable, "producer.py", "out.json"
    ])

    with patch("scripts.run_with_provenance.get_environment_info") as mock_env:
        mock_env.return_value = {
            "python_version": "3.10",
            "torch_version": "2.0",
            "cuda_available": False,
            "gpu_compute_capability": None,
            "rosa_soft_build_capabilities": BuildCapabilities() # Non-serializable, but stripped!
        }

        with pytest.raises(SystemExit) as exc:
            run_with_provenance.main()

        assert exc.value.code == 0

    assert artifact.exists()
    data = json.loads(artifact.read_text())
    assert "provenance" in data
    assert "rosa_soft_build_capabilities" not in data["provenance"]["device"]

def test_all_keys_preserved_and_sidecar_matches(repo_dir, monkeypatch):
    artifact = repo_dir / "out.json"
    producer = repo_dir / "producer.py"
    producer.write_text("import json, sys\nwith open(sys.argv[1], 'w') as f:\n  json.dump({'original_key': 'value'}, f)", encoding="utf-8")

    subprocess.run(["git", "add", "producer.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    monkeypatch.chdir(repo_dir)
    monkeypatch.setattr(sys, "argv", [
        "run_with_provenance.py",
        "--artifact", "out.json",
        "--producer", "producer.py",
        "--",
        sys.executable, "producer.py", "out.json"
    ])

    with pytest.raises(SystemExit) as exc:
        run_with_provenance.main()

    assert exc.value.code == 0

    data = json.loads(artifact.read_text())
    assert data["original_key"] == "value"
    assert "provenance" in data
    assert len(data["provenance"]["repository_commit"]) == 40

    sidecar = repo_dir / "out.json.sha256"
    assert sidecar.exists()
    sidecar_content = sidecar.read_text().strip()
    expected_digest = run_with_provenance.hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert sidecar_content == f"{expected_digest}  out.json"

def test_non_zero_exit_stamps_and_propagates(repo_dir, monkeypatch):
    artifact = repo_dir / "out.json"
    producer = repo_dir / "producer.py"
    producer.write_text("import json, sys\nwith open(sys.argv[1], 'w') as f:\n  json.dump({'failed': True}, f)\nsys.exit(42)", encoding="utf-8")

    subprocess.run(["git", "add", "producer.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    monkeypatch.chdir(repo_dir)
    monkeypatch.setattr(sys, "argv", [
        "run_with_provenance.py",
        "--artifact", "out.json",
        "--producer", "producer.py",
        "--",
        sys.executable, "producer.py", "out.json"
    ])

    with pytest.raises(SystemExit) as exc:
        run_with_provenance.main()

    assert exc.value.code == 42

    data = json.loads(artifact.read_text())
    assert data["failed"] is True
    assert data["provenance"]["exit_code"] == 42

def test_pre_existing_artifact_rejected(repo_dir, monkeypatch):
    artifact = repo_dir / "out.json"
    artifact.write_text("{}")

    producer = repo_dir / "producer.py"
    producer.write_text("pass", encoding="utf-8")
    subprocess.run(["git", "add", "producer.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    monkeypatch.chdir(repo_dir)
    monkeypatch.setattr(sys, "argv", [
        "run_with_provenance.py",
        "--artifact", "out.json",
        "--producer", "producer.py",
        "--",
        sys.executable, "producer.py", "out.json"
    ])

    with pytest.raises(SystemExit) as exc:
        run_with_provenance.main()

    assert exc.value.code != 0

def test_dirty_checkout_rejected(repo_dir, monkeypatch):
    producer = repo_dir / "producer.py"
    producer.write_text("pass", encoding="utf-8")
    subprocess.run(["git", "add", "producer.py"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    # dirty the repo
    (repo_dir / "dirty.txt").write_text("dirty")
    subprocess.run(["git", "add", "dirty.txt"], cwd=repo_dir, check=True)

    monkeypatch.chdir(repo_dir)
    monkeypatch.setattr(sys, "argv", [
        "run_with_provenance.py",
        "--artifact", "out.json",
        "--producer", "producer.py",
        "--",
        sys.executable, "producer.py", "out.json"
    ])

    with pytest.raises(SystemExit) as exc:
        run_with_provenance.main()

    assert exc.value.code != 0
