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
    $command = 'powershell.exe -NoProfile -NonInteractive ' +
      '-ExecutionPolicy Bypass -File "D:\repo\scripts\pueue_wrap.ps1" ' +
      '-RepoRoot "D:\repo" -RequireCuda'
    $id = pueue add --group gpu0 --stashed --print-task-id $command
    pueue env set $id PUEUE_TASK_JSON '["python","-u","scripts/x.py"]'
    pueue enqueue $id

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

# --- Effective task array (single canonical form) ----------------------------
# Machine channel (-ArgsJson / PUEUE_TASK_JSON env) or human positional args
# both normalize into $Task. A JSON scalar is rejected: tasks are arrays.
$Task = @()
if ($ArgsJson -or $env:PUEUE_TASK_JSON) {
    $raw = if ($ArgsJson) { $ArgsJson } else { $env:PUEUE_TASK_JSON }
    $parsed = ConvertFrom-Json $raw
    if ($parsed -isnot [System.Array]) {
        throw "ArgsJson must be a JSON ARRAY of strings; a scalar was given."
    }
    foreach ($item in @($parsed)) {
        if ($null -eq $item -or $item.GetType().Name -ne "String") {
            throw "ArgsJson must be a JSON array of strings."
        }
        $Task += [string]$item
    }
} elseif ($TaskArgs) {
    $Task = @($TaskArgs)
}

# --- 1. Repository selection (checkout-bound, fails closed on mismatch) -----
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$wrapperCheckout = Split-Path -Parent $PSScriptRoot
if ((Resolve-Path -LiteralPath $wrapperCheckout).Path -ne $RepoRoot) {
    throw ("-RepoRoot '$RepoRoot' does not match this launcher's checkout " +
           "'$wrapperCheckout'. Clone-bound launchers do not run against " +
           "foreign trees.")
}
Set-Location -LiteralPath $RepoRoot

# --- 2. Provenance: commit + dirty state (explicit failure states) ----------
$commit = "UNAVAILABLE (git rev-parse failed)"
$dirty = "UNKNOWN"
if (Get-Command git -ErrorAction SilentlyContinue) {
    $previousErrorAction = $ErrorActionPreference
    try {
        # Windows PowerShell promotes native stderr to ErrorRecord objects.
        # Keep an expected git failure in provenance instead of aborting the job.
        $ErrorActionPreference = "Continue"
        $commitOutput = @(git -C $RepoRoot rev-parse HEAD 2>$null)
        $commitExitCode = $LASTEXITCODE
        if ($commitExitCode -eq 0 -and $commitOutput.Count -eq 1) {
            $commit = [string]$commitOutput[0]
            $statusOutput = @(git -C $RepoRoot status --porcelain 2>$null)
            $statusExitCode = $LASTEXITCODE
            if ($statusExitCode -eq 0) {
                $dirty = [bool]$statusOutput.Count
            } else {
                $dirty = "UNKNOWN (git status failed)"
            }
        }
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
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
    if (@($Task).Count -lt 1) {
        throw "SelfCheck: expected at least one argument in the task array."
    }
    if ($RequireCuda) {
        # Discovery IS validation for kernel-compiling jobs: fail closed,
        # naming every missing tool.
        $missing = @()
        foreach ($t in "cl", "nvcc", "ninja") {
            if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { $missing += $t }
        }
        if ($missing) {
            Write-Host ("SELFCHECK FAILED - missing required tools: " +
                        ($missing -join ", ")) -ForegroundColor Red
            exit 1
        }
    }
    if ((Get-Location).Path -ne $RepoRoot) {
        throw "SelfCheck: working directory does not match repository root."
    }
    Write-Host "SELFCHECK OK - nothing was executed." -ForegroundColor Green
    exit 0
}

if (@($Task).Count -lt 1) {
    throw "No task arguments supplied (-ArgsJson/PUEUE_TASK_JSON or positional)."
}

# --- 4. Execute the checked-in command as an array --------------------------
if ($Task.Count -eq 1) {
    & $Task[0]
} else {
    & $Task[0] @($Task[1..($Task.Count - 1)])
}
exit $LASTEXITCODE
