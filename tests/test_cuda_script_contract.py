import re
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


def test_bootstrap_requires_callable_python():
    bootstrap = Path("scripts/init_cuda_env.ps1").read_text(encoding="utf-8")
    assert "& $PythonExe --version" in bootstrap
    assert "$pythonExitCode -ne 0" in bootstrap


def test_wrapper_has_checked_in_execution_regression():
    regression = Path("tests/test_pueue_wrap.ps1").read_text(encoding="utf-8")
    for contract in ("exit 7", "one-element", "scalar JSON",
                     "mismatched repository", "uncallable venv Python"):
        assert contract in regression


def _launcher_source() -> str:
    return Path("scripts/start_pueued.ps1").read_text(encoding="utf-8")


def _launcher_code() -> str:
    """The script with its comment block and comment lines removed.

    Absence contracts have to run against code. This script explains why it
    avoids `[Math]::Pow` and `-Wait`, so asserting on the raw text asserts the
    opposite of what is meant.
    """
    source = _launcher_source()
    source = re.sub(r"<#.*?#>", "", source, flags=re.DOTALL)
    return "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))


def _flat(text: str) -> str:
    """Collapse whitespace so a line-wrapped sentence still matches."""
    return " ".join(text.split())


def test_daemon_launcher_fails_closed_on_a_wrong_mask():
    """A starter that reports success on an unverified mask is the whole bug.

    The earlier version printed the current mask and exited zero whenever a
    daemon was already running, so a daemon started by hand at 0xFFFF passed
    silently and every task it spawned inherited core 0.
    """
    code = _launcher_code()
    # The target is derived before any daemon is considered, so the
    # already-running branch has something to compare against.
    assert code.index("$target = $sys") < code.index("Get-Process -Name pueued")
    # Success on an existing daemon is conditional on the mask matching.
    assert "if ($existing -eq $target) {" in code
    assert "exit 0" in code.split("if ($existing -eq $target) {", 1)[1]
    assert "will not kill a daemon" in _flat(code)


def test_daemon_launcher_has_no_force_kill_path():
    """A starter must not be able to terminate active queued jobs."""
    code = _launcher_code()
    assert "Stop-Process" not in code
    assert "-Force" not in code


def test_daemon_launcher_derives_the_mask_from_the_system_not_a_count():
    """[Math]::Pow(2,n)-1 is silently wrong at n=54..62 and throws at n>=63."""
    code = _launcher_code()
    assert "GetProcessAffinityMask" in code
    assert "[Math]::Pow" not in code
    assert "NumberOfLogicalProcessors" not in code
    # One affinity mask cannot span processor groups.
    assert "GetActiveProcessorGroupCount" in code
    assert "$groups -ne 1" in code


def test_daemon_launcher_sets_affinity_at_creation_without_pipes():
    """The mask must precede the image, and the caller must not block on pipes."""
    code = _launcher_code()
    assert "/affinity $hex" in code
    assert 'Start-Process -FilePath "cmd.exe"' in code
    assert "-Wait" not in code


def test_daemon_launcher_cites_the_controlled_rate_not_the_withdrawn_one():
    """The approximate 9% campaign figure was withdrawn as confounded."""
    flat = _flat(_launcher_source())
    assert "14 failures in 200 launches" in flat
    assert "INTERMITTENT" in flat
    # The withdrawn rate may appear, but only inside its own retraction --
    # never standing alone as the fault's current rate.
    start = 0
    while (found := flat.find("9%", start)) != -1:
        sentence = flat[max(0, found - 200):found + 200]
        assert "withdrawn" in sentence, f"unretracted 9% near: {sentence}"
        start = found + 2
