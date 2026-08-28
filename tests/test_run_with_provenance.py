import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def test_provenance_block_added(tmp_path):
    artifact = tmp_path / "out.json"
    producer = tmp_path / "producer.py"
    producer.write_text(f"""
import json
with open('{artifact.as_posix()}', 'w', encoding='utf-8') as f:
    json.dump({{"some": "data"}}, f)
""")

    wrapper = Path("scripts/run_with_provenance.py").absolute()
    res = subprocess.run(
        [sys.executable, str(wrapper), "--artifact", str(artifact), "--", sys.executable, str(producer)]
    )
    assert res.returncode == 0

    with open(artifact, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "some" in data
    assert data["some"] == "data"
    assert "provenance" in data

    prov = data["provenance"]
    assert "repository_commit" in prov
    assert "dirty" in prov
    assert "command" in prov
    assert prov["command"] == [sys.executable, str(producer)]
    assert "producer_script_sha256" in prov
    assert prov["producer_script_sha256"] is not None
    assert "device" in prov
    assert "python_version" in prov["device"]
    assert "torch_version" in prov["device"]
    assert "cuda_version" in prov["device"]
    assert "platform" in prov
    assert "started" in prov
    assert "finished" in prov
    assert "exit_code" in prov
    assert prov["exit_code"] == 0


def test_sidecar_digest_matches_modified(tmp_path):
    artifact = tmp_path / "out.json"
    producer = tmp_path / "producer.py"
    producer.write_text(f"""
import json
with open('{artifact.as_posix()}', 'w', encoding='utf-8') as f:
    json.dump({{"some": "data"}}, f)
""")

    wrapper = Path("scripts/run_with_provenance.py").absolute()
    res = subprocess.run(
        [sys.executable, str(wrapper), "--artifact", str(artifact), "--", sys.executable, str(producer)]
    )
    assert res.returncode == 0

    sidecar = tmp_path / "out.json.sha256"
    assert sidecar.exists()

    import hashlib

    h = hashlib.sha256()
    with open(artifact, "rb") as f:
        h.update(f.read())
    expected_digest = h.hexdigest()

    with open(sidecar, "r", encoding="utf-8") as f:
        content = f.read()

    assert content.strip() == f"{expected_digest}  out.json"


def test_nonzero_exit_produces_provenance(tmp_path):
    artifact = tmp_path / "out.json"
    producer = tmp_path / "producer.py"
    producer.write_text(f"""
import json
import sys
with open('{artifact.as_posix()}', 'w', encoding='utf-8') as f:
    json.dump({{"some": "data"}}, f)
sys.exit(42)
""")

    wrapper = Path("scripts/run_with_provenance.py").absolute()
    res = subprocess.run(
        [sys.executable, str(wrapper), "--artifact", str(artifact), "--", sys.executable, str(producer)]
    )
    assert res.returncode == 42

    with open(artifact, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "provenance" in data
    assert data["provenance"]["exit_code"] == 42


def test_missing_artifact(tmp_path):
    artifact = tmp_path / "out.json"
    producer = tmp_path / "producer.py"
    producer.write_text("""
import sys
sys.exit(0)
""")

    wrapper = Path("scripts/run_with_provenance.py").absolute()
    res = subprocess.run(
        [sys.executable, str(wrapper), "--artifact", str(artifact), "--", sys.executable, str(producer)],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "does not exist" in res.stderr


def test_artifact_with_existing_provenance(tmp_path):
    artifact = tmp_path / "out.json"
    producer = tmp_path / "producer.py"
    producer.write_text(f"""
import json
with open('{artifact.as_posix()}', 'w', encoding='utf-8') as f:
    json.dump({{"provenance": "exists"}}, f)
""")

    wrapper = Path("scripts/run_with_provenance.py").absolute()
    res = subprocess.run(
        [sys.executable, str(wrapper), "--artifact", str(artifact), "--", sys.executable, str(producer)],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "already contains a 'provenance' key" in res.stderr


def test_no_cuda_available(tmp_path):
    # Instead of monkeypatching in a subprocess, let's inject a mock script that mocks torch
    artifact = tmp_path / "out.json"
    producer = tmp_path / "producer.py"
    producer.write_text(f"""
import json
with open('{artifact.as_posix()}', 'w', encoding='utf-8') as f:
    json.dump({{"some": "data"}}, f)
""")

    # We need to test the wrapper's behavior, so we mock get_environment_info
    sys.path.insert(0, str(Path("scripts").absolute()))
    import run_with_provenance

    # Create an artificial arg list
    args = ["--artifact", str(artifact), "--", sys.executable, str(producer)]

    def mock_get_environment_info():
        return {
            "python_version": "3.10.0",
            "torch_version": "2.0.0",
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "gpu_compute_capability": None,
        }

    with patch("run_with_provenance.get_environment_info", side_effect=mock_get_environment_info):
        try:
            import sys as _sys

            with patch.object(_sys, "argv", ["run_with_provenance.py"] + args):
                exit_code = run_with_provenance.main()
                assert exit_code == 0

                with open(artifact, "r", encoding="utf-8") as f:
                    data = json.load(f)

                prov = data["provenance"]
                assert prov["device"]["gpu_name"] is None
        finally:
            _sys.path.pop(0)


def test_full_commit_sha_format(tmp_path):
    artifact = tmp_path / "out.json"
    producer = tmp_path / "producer.py"
    producer.write_text(f"""
import json
with open('{artifact.as_posix()}', 'w', encoding='utf-8') as f:
    json.dump({{"some": "data"}}, f)
""")

    wrapper = Path("scripts/run_with_provenance.py").absolute()
    res = subprocess.run(
        [sys.executable, str(wrapper), "--artifact", str(artifact), "--", sys.executable, str(producer)]
    )
    assert res.returncode == 0

    with open(artifact, "r", encoding="utf-8") as f:
        data = json.load(f)

    prov = data["provenance"]
    commits = prov["repository_commit"]
    if isinstance(commits, list):
        for c in commits:
            assert len(c) == 40
            assert all(char in "0123456789abcdef" for char in c)
    else:
        assert len(commits) == 40
        assert all(char in "0123456789abcdef" for char in commits)
