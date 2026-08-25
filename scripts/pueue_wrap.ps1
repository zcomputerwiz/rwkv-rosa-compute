<#
.SYNOPSIS
    Repository-owned launcher for jobs submitted to the machine-local Pueue
    queue (see agent-sync-protocol docs/DISPATCHER_PUEUE_WINDOWS.md).

.DESCRIPTION
    Pueue owns process lifetime; this repository owns its own environment.
    When pueued starts a task whose command is this wrapper, the wrapper:

      1. Resolves the repository root (absolute) and selects it as the
         working directory, so relative paths in queued arguments behave
         identically no matter which directory the daemon used to spawn us.
      2. Records provenance: repository commit + dirty state, resolved
         working directory, final argument array, and the environment
         discovered by scripts/init_cuda_env.ps1 (venv python, VS/toolset).
      3. -RequireCuda additionally exports EXP0_REQUIRE_RWKV_CUDA=1 for
         jobs that will compile or run the RWKV-7 CUDA kernel.
      4. Executes the target as an ARGUMENT ARRAY: the first remaining
         argument is the executable/script, the rest are its arguments.
         Nothing is re-parsed from a string; nothing is read from ProjectSync.

    -SelfCheck runs steps 1-3 and prints a PROVENANCE JSON object without
    executing anything - no Torch import, no compilation, no CUDA access.
    This is the CPU-only validation required before queueing real jobs.

.PARAMETER RepoRoot
    Absolute path to the rwkv-rosa-compute checkout running the job.
    Defaults to the parent of this script's directory.

.PARAMETER RequireCuda
    Export EXP0_REQUIRE_RWKV_CUDA=1 (kernel-compiling jobs).

.PARAMETER SelfCheck
    Validate and print provenance only. The argument array is echoed but not
    executed.

.PARAMETER TaskArgs
    Argument array: <executable-or-script> [args...]. Preserved verbatim;
    PowerShell array splatting means embedded spaces and semicolons survive
    without an intermediate shell.

.EXAMPLE (submission happens on the operator/agent side)
    pueue add --group gpu0 -- powershell -NoProfile -File ^
      D:\GitHub\rwkv-rosa-compute\scripts\pueue_wrap.ps1 ^
      -RepoRoot D:\GitHub\rwkv-rosa-compute -RequireCuda ^
      -- .\scripts\run_experiment.py --architecture rwkv ...

.EXAMPLE (CPU-only self check)
    powershell -NoProfile -File .\scripts\pueue_wrap.ps1 -SelfCheck -- echo hi
#>
[CmdletBinding()]
param(
    [string]$RepoRoot,
    [switch]$RequireCuda,
    [switch]$SelfCheck,
    # Canonical machine interface: a JSON array string, so the argument
    # ARRAY survives the text-only boundary between agent, pueue add, and
    # daemon. Example:  -ArgsJson '["python","-u","scripts/x.py","--k","v"]'
    [string]$ArgsJson,
    # Human/interactive form: everything after named parameters.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TaskArgs
)

$ErrorActionPreference = "Stop"

# --- Effective task array ----------------------------------------------------
$Task = @()
if (-not $ArgsJson -and $env:PUEUE_TASK_JSON) {
    # Machine channel: an environment variable survives any transport
    # byte-exactly, unlike command-line quoting.
    $ArgsJson = $env:PUEUE_TASK_JSON
}
if ($ArgsJson) {
    $parsed = ConvertFrom-Json $ArgsJson
    foreach ($item in @($parsed)) {
        if ($null -eq $item -or $item.GetType().Name -ne "String") {
            throw "ArgsJson must be a JSON array of strings."
        }
        $Task += [string]$item
    }
} elseif ($TaskArgs) {
    $Task = @($TaskArgs)
}

# --- 1. Repository selection ------------------------------------------------
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
Set-Location -LiteralPath $RepoRoot

# --- 2. Provenance: commit + dirty state ------------------------------------
$commit = ""
$dirty = $true
try {
    $commit = git -C $RepoRoot rev-parse HEAD
    $dirty = (git -C $RepoRoot status --porcelain).Count -gt 0
} catch {
    $commit = "unavailable"
}

# --- 3. Environment discovery (shared bootstrap) ----------------------------
$initEnv = Join-Path $PSScriptRoot "init_cuda_env.ps1"
if (-not (Test-Path -LiteralPath $initEnv)) {
    throw "Missing environment bootstrap: $initEnv"
}
. $initEnv -RequireCuda:$RequireCuda

# --- Provenance object ------------------------------------------------------
$provenance = [ordered]@{
    mode          = if ($SelfCheck) { "self-check" } else { "execute" }
    repo_root     = $RepoRoot
    commit        = $commit
    dirty         = $dirty
    working_dir   = (Get-Location).Path
    python        = $PythonExe
    vs_version    = $VsVersion
    msvc_toolset  = $(if ($toolset) { $toolset } else { "default" })
    cuda_required = [bool]$RequireCuda
    arg_count     = @($Task).Count
    args          = @($Task)
}
$json = $provenance | ConvertTo-Json -Depth 3
Write-Host "PROVENANCE $json"

if ($SelfCheck) {
    # Assertions that must hold for any queued job, CPU-only:
    if (@($TaskArgs).Count -lt 1) {
        throw "SelfCheck: expected at least one argument in the task array."
    }
    if ((Get-Location).Path -ne $RepoRoot) {
        throw "SelfCheck: working directory does not match repository root."
    }
    Write-Host "SELFCHECK OK - nothing was executed." -ForegroundColor Green
    exit 0
}

if (@($TaskArgs).Count -lt 1) {
    throw "No task arguments supplied after parameter block."
}

# --- 4. Execute the checked-in command as an array --------------------------
& $TaskArgs[0] @($TaskArgs[1..($TaskArgs.Count - 1)])
exit $LASTEXITCODE
