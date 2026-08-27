<#
.SYNOPSIS
    Shared repository environment bootstrap for scripts that need the venv,
    Ninja, and/or the MSVC toolchain on Windows.

.DESCRIPTION
    Dot-source this script; do not invoke it directly. It centralizes the
    environment discovery that used to be inline in run_cuda_tests.ps1:

      1. Resolves the repository root by absolute path and sets the working
         directory to it.
      2. Prefers .venv\Scripts\python.exe; falls back to PATH python, and
         verifies that the selected interpreter can start.
      3. Locates Visual Studio via vswhere and activates the newest installed
         64-bit MSVC toolset (EXP0_MSVC_TOOLSET overrides).
      4. Optionally requires the CUDA toolkit (EXP0_REQUIRE_RWKV_CUDA=1).

    After dot-sourcing, callers get:

        $RepoRoot    absolute repository root
        $PythonExe   resolved python interpreter (absolute path)
        $VsVersion   selected Visual Studio installation version (or "")
        $Toolset     selected MSVC toolset (or "default")
        $CudaRequired  mirrors the -RequireCuda switch

.PARAMETER RequireCuda
    Sets EXP0_REQUIRE_RWKV_CUDA=1 so a missing CUDA toolkit is a failure
    instead of a silent skip. Use for any job that will compile or run the
    RWKV-7 CUDA kernel.

.EXAMPLE
    . .\scripts\init_cuda_env.ps1 -RequireCuda
    & $PythonExe -m pytest -m cuda
#>
param(
    [switch]$RequireCuda,
    # Optional explicit checkout. Callers that select a repository (e.g.
    # pueue_wrap.ps1) pass it in; when omitted, the checkout containing this
    # script is used.
    [string]$RepoRoot
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
if (-not (Test-Path -LiteralPath $RepoRoot)) {
    throw "Repository root not found: $RepoRoot"
}
Set-Location $RepoRoot

# --- Python interpreter -----------------------------------------------------
$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $PythonExe = $venvPython
    $env:PATH = (Join-Path $RepoRoot ".venv\Scripts") + ";$env:PATH"
} else {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if (-not $found) {
        throw "No .venv and no python on PATH. Run: python scripts/bootstrap_env.py"
    }
    $PythonExe = $found.Source
    Write-Host "No .venv found; using $PythonExe" -ForegroundColor Yellow
}

try {
    $pythonVersion = (& $PythonExe --version 2>&1 | Out-String).Trim()
    $pythonExitCode = $LASTEXITCODE
} catch {
    throw "Python interpreter cannot start: $PythonExe`n$($_.Exception.Message)"
}
if ($pythonExitCode -ne 0) {
    throw ("Python interpreter failed its --version check (exit " +
           "$pythonExitCode): $PythonExe`n$pythonVersion")
}

# --- MSVC environment -------------------------------------------------------
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

$VsVersion = & $vswhere @vswhereArgs -property installationVersion
$validatedVsMajor = 17  # Visual Studio 2022; locally validated with CUDA 12.9.
if ($VsVersion) {
    try {
        $selectedVsMajor = ([version]$VsVersion).Major
        if ($selectedVsMajor -ne $validatedVsMajor) {
            Write-Warning @"
vswhere selected Visual Studio $VsVersion at:
  $vsPath
This repository has validated the Windows CUDA extension build with Visual
Studio 2022 (17.x). This toolchain may still work, so execution continues;
interpret compiler/build failures as a possible host-toolchain compatibility issue.
"@
        }
    } catch {
        Write-Warning "Could not parse Visual Studio installationVersion '$VsVersion'; continuing with the selected toolchain."
    }
} else {
    Write-Warning "vswhere did not report a Visual Studio version; continuing with $vsPath."
}

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvars)) {
    throw "Expected vcvars64.bat at $vcvars but it is missing."
}

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
# compiler onto PATH for the pytest process below. A failing vcvars must fail
# the bootstrap: silently continuing leaves cl.exe missing and any -SelfCheck
# or kernel build would report a confusing error much later.
$vcvarsOutput = cmd /c "`"$vcvars`" $vcvarsArgs >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) {
    throw "vcvars64.bat failed with exit code $LASTEXITCODE (toolset: $toolset)."
}
$vcvarsOutput | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] -ErrorAction SilentlyContinue
    }
}

$CudaRequired = [bool]$RequireCuda
if ($CudaRequired) {
    $env:EXP0_REQUIRE_RWKV_CUDA = "1"
}

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
Write-Host ("  {0,-9}: {1}" -f "VS", "$VsVersion  $vsPath")
Write-Host ("  {0,-9}: {1}" -f "toolset", $(if ($toolset) { $toolset } else { "default" }))
Show-Tool "cl.exe" "cl"
Show-Tool "nvcc" "nvcc"
Show-Tool "ninja" "ninja"
Write-Host ("  {0,-9}: {1}  ({2})" -f "python", $PythonExe, $pythonVersion)
