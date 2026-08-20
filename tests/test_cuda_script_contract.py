from pathlib import Path


def test_windows_cuda_runner_warns_on_unvalidated_visual_studio_family():
    script = Path("scripts/run_cuda_tests.ps1").read_text(encoding="utf-8")

    assert "$validatedVsMajor = 17" in script
    assert "$selectedVsMajor -ne $validatedVsMajor" in script
    assert "Write-Warning" in script
    assert "the test run will continue" in script
    assert "throw" not in script.split("$selectedVsMajor -ne $validatedVsMajor", 1)[1].split(
        "$vcvars", 1
    )[0]
