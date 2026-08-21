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

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# --- Python interpreter -----------------------------------------------------
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
    $env:PATH = (Join-Path $repoRoot ".venv\Scripts") + ";$env:PATH"
} else {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if (-not $found) {
        throw "No .venv and no python on PATH. Run: python scripts/bootstrap_env.py"
    }
    $python = $found.Source
    Write-Host "No .venv found; using $python" -ForegroundColor Yellow
}

# --- MSVC environment -------------------------------------------------------
# vswhere ships with every VS 2017+ installer and is the supported way to find
# an installation without hardcoding a path or edition.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw @"
vswhere.exe not found, so Visual Studio is probably not installed.
Compiling the RWKV-7 kernel needs the MSVC toolchain. Install "Visual Studio
Build Tools" with the "Desktop development with C++" workload, then rerun.
"@
}

$vswhereArgs = @(
    "-latest",
    "-products", "*",
    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
)
$vsPath = & $vswhere @vswhereArgs -property installationPath
if (-not $vsPath) {
    throw @"
Visual Studio is installed but without the C++ toolset. Add the
"Desktop development with C++" workload in the Visual Studio Installer.
"@
}

$vsVersion = & $vswhere @vswhereArgs -property installationVersion
$validatedVsMajor = 17  # Visual Studio 2022; locally validated with CUDA 12.9.
if ($vsVersion) {
    try {
        $selectedVsMajor = ([version]$vsVersion).Major
        if ($selectedVsMajor -ne $validatedVsMajor) {
            Write-Warning @"
vswhere selected Visual Studio $vsVersion at:
  $vsPath
This repository has validated the Windows CUDA extension build with Visual
Studio 2022 (17.x). This toolchain may still work, so the test run will continue;
interpret compiler/build failures as a possible host-toolchain compatibility issue.
"@
        }
    } catch {
        Write-Warning "Could not parse Visual Studio installationVersion '$vsVersion'; continuing with the selected toolchain."
    }
} else {
    Write-Warning "vswhere did not report a Visual Studio version; continuing with $vsPath."
}

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvars)) {
    throw "Expected vcvars64.bat at $vcvars but it is missing."
}

# vcvars64.bat activates the installation's *default* toolset, which is not
# necessarily the newest installed one. That matters beyond style: MSVC 14.38
# cannot compile the C that Triton generates ("error C2059: syntax error: '}'"
# in cuda_utils.c), so torch.compile fails on a box where 14.44 is present but
# is not the default. Select the newest toolset unless EXP0_MSVC_TOOLSET
# overrides it.
$toolsetRoot = Join-Path $vsPath "VC\Tools\MSVC"
$toolset = $env:EXP0_MSVC_TOOLSET
if (-not $toolset -and (Test-Path -LiteralPath $toolsetRoot)) {
    $newest = Get-ChildItem -Path $toolsetRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            $parsed = $null
            if ([version]::TryParse($_.Name, [ref]$parsed)) { $parsed }
        } | Sort-Object -Descending | Select-Object -First 1
    if ($newest) { $toolset = "$($newest.Major).$($newest.Minor)" }
}
$vcvarsArgs = if ($toolset) { "-vcvars_ver=$toolset" } else { "" }

# Run vcvars in cmd and copy the resulting environment into this session; a
# child process cannot modify its parent, so this is the only way to get the
# compiler onto PATH for the pytest process below.
cmd /c "`"$vcvars`" $vcvarsArgs >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] -ErrorAction SilentlyContinue
    }
}

# Matches the workflow: a missing CUDA toolkit is a failure, not a silent skip.
$env:EXP0_REQUIRE_RWKV_CUDA = "1"

# --- Report what we assembled ----------------------------------------------
function Show-Tool {
    param([string]$Label, [string]$Name)
    $path = (Get-Command $Name -ErrorAction SilentlyContinue).Source
    if ($path) {
        Write-Host ("  {0,-9}: {1}" -f $Label, $path)
    } else {
        Write-Host ("  {0,-9}: NOT FOUND" -f $Label) -ForegroundColor Red
    }
}

Write-Host "Toolchain:"
Write-Host ("  {0,-9}: {1}" -f "VS", "$vsVersion  $vsPath")
Write-Host ("  {0,-9}: {1}" -f "toolset", $(if ($toolset) { $toolset } else { "default" }))
Show-Tool "cl.exe" "cl"
Show-Tool "nvcc" "nvcc"
Show-Tool "ninja" "ninja"
Write-Host ("  {0,-9}: {1}" -f "python", $python)

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
$caps = & $python -c "import rosa_compute, rosa_soft; print(rosa_soft.BUILD_CAPABILITIES)"
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nrosa_soft : $caps"
} else {
    Write-Host "`nrosa_soft : could not be imported" -ForegroundColor Yellow
}

Write-Host "`nRunning: pytest -m cuda $($PytestArgs -join ' ')`n"
& $python -m pytest -m "cuda" -v @PytestArgs
$pytestExit = $LASTEXITCODE

if ($pytestExit -eq 0) {
    Write-Host "`nCUDA suite passed." -ForegroundColor Green
} else {
    Write-Host "`nCUDA suite failed (exit $pytestExit)." -ForegroundColor Red
    Write-Host "If the failures mention a build, try: .\scripts\clean_rwkv_cuda_cache.ps1"
}
exit $pytestExit
