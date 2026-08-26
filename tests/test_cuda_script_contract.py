from pathlib import Path


def test_windows_cuda_runner_warns_on_unvalidated_visual_studio_family():
    # The VS contract lives in the shared bootstrap (init_cuda_env.ps1) since
    # the Pueue launcher refactor; run_cuda_tests.ps1 consumes it.
    bootstrap = Path("scripts/init_cuda_env.ps1").read_text(encoding="utf-8")

    assert "$validatedVsMajor = 17" in bootstrap
    assert "$selectedVsMajor -ne $validatedVsMajor" in bootstrap
    assert "Write-Warning" in bootstrap
    assert "execution continues" in bootstrap
    after = bootstrap.split("$selectedVsMajor -ne $validatedVsMajor", 1)[1]
    assert "throw" not in after.split("$vcvars", 1)[0]


def test_runner_uses_shared_bootstrap():
    script = Path("scripts/run_cuda_tests.ps1").read_text(encoding="utf-8")
    assert "init_cuda_env.ps1" in script
    assert "-RequireCuda" in script
