<#
.SYNOPSIS
    Run the CUDA-marked test suite the way the GitHub Actions GPU job runs it.

.DESCRIPTION
    `pytest -m cuda` does not work from a plain shell on Windows. The fused
    RWKV-7 recurrence kernel is compiled on first use by
    torch.utils.cpp_extension, which needs two things a normal prompt lacks:

      * ninja, which lives in .venv\Scripts and is therefore invisible unless
        the venv is activated or that directory is on PATH;
      * a host compiler, which means importing the MSVC environment.

    Without them every rwkv7_fused_cuda test fails with "Ninja is required to
    load C++ extensions", which looks like a code failure and is not. This
    script assembles that environment and runs the same pytest selection as
    .github/workflows/cuda-tests.yml, with the same EXP0_REQUIRE_RWKV_CUDA=1
    that turns a missing CUDA toolkit into a failure instead of a skip.

    Visual Studio 2022 (17.x) is the validated Windows host-toolchain family for
    this repository. vswhere still selects the latest installed C++ toolchain;
    if that is a newer/older Visual Studio family, the script warns but proceeds
    so compatible future toolchains are not artificially blocked.

    It does not install anything. Use scripts/bootstrap_env.py for that.

.PARAMETER Cold
    Delete cached kernel builds first, so the run compiles from scratch the way
    a fresh CI runner does. A cold run takes roughly 20s longer; if a run
    finishes suspiciously fast it used the cache and proved nothing about the
    build.

.PARAMETER PytestArgs
    Remaining arguments are passed through to pytest, e.g. -k rwkv7 or -x.

.EXAMPLE
    .\scripts\run_cuda_tests.ps1

.EXAMPLE
    .\scripts\run_cuda_tests.ps1 -Cold -- -k rwkv7_fused
#>
[CmdletBinding()]
param(
    [switch]$Cold,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"

# Environment assembly lives in init_cuda_env.ps1 so other entry points
# (e.g. the Pueue job launcher) reuse the same validated bootstrap instead of
# maintaining a second, divergent copy. Dot-sourcing shares its scope: after
# this call $RepoRoot, $PythonExe, $VsVersion and $Toolset are set, the venv
# Scripts directory is on PATH, the MSVC toolset is activated, and
# EXP0_REQUIRE_RWKV_CUDA=1 is exported.
$initEnv = Join-Path $PSScriptRoot "init_cuda_env.ps1"
if (-not (Test-Path -LiteralPath $initEnv)) {
    throw "Missing environment bootstrap: $initEnv"
}
. $initEnv -RequireCuda

if ($Cold) {
    Write-Host "`nClearing cached kernel builds (cold run)..."
    & (Join-Path $PSScriptRoot "clean_rwkv_cuda_cache.ps1")
}

# The workflow asserts rosa_soft was built with CUDA. That build does not work
# on Windows -- MSVC links the extension without exporting PyInit__C, and the
# artifact then breaks every rosa_compute import -- so report the capabilities
# rather than asserting them, and expect 'reference' here but CUDA in CI.
# rosa_soft is not pip-installed here: rosa_compute.rosa_compat is what puts
# external/rosa_soft on sys.path, so it has to be imported first.
$caps = & $PythonExe -c "import rosa_compute, rosa_soft; print(rosa_soft.BUILD_CAPABILITIES)"
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nrosa_soft : $caps"
} else {
    Write-Host "`nrosa_soft : could not be imported" -ForegroundColor Yellow
}

Write-Host "`nRunning: pytest -m cuda $($PytestArgs -join ' ')`n"
& $PythonExe -m pytest -m "cuda" -v @PytestArgs
$pytestExit = $LASTEXITCODE

if ($pytestExit -eq 0) {
    Write-Host "`nCUDA suite passed." -ForegroundColor Green
} else {
    Write-Host "`nCUDA suite failed (exit $pytestExit)." -ForegroundColor Red
    Write-Host "If the failures mention a build, try: .\scripts\clean_rwkv_cuda_cache.ps1"
}
exit $pytestExit
