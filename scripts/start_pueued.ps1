<#
.SYNOPSIS
    Start the Pueue daemon with core 0 excluded from its affinity mask.

.DESCRIPTION
    Start pueued through this script rather than by running pueued.exe directly.

    Why this exists
    ---------------
    A Process Lasso rule on this host excludes core 0 from Python process
    affinity (prolasso.ini: DefaultAffinitiesEx=python.exe,0,1-15), to keep
    interrupt handling off the cores doing dispatch-bound work. Lasso applies
    that rule a short time AFTER a process starts -- measured at 0.385 s and
    0.608 s in two observations.

    Importing PyTorch takes about 1.75 s. The rewrite therefore lands mid-import,
    while OpenMP is initializing. OpenMP reads the process mask (core 0 still
    allowed), computes thread affinities from it, then calls
    SetThreadAffinityMask after Lasso has removed core 0. Windows rejects a
    thread mask that is not a subset of the process mask and returns
    ERROR_INVALID_PARAMETER (87):

        OMP: Error #135: Cannot set thread affinity mask.
        OMP: System error #87: The parameter is incorrect.

    The process dies during import, before any work is done. The fault is
    INTERMITTENT: a controlled interleaved experiment measured 14 failures in
    200 launches with the mask deliberately widened to include core 0, against
    0 in 200 with it excluded (Fisher exact, two-sided, p = 0.000096). An
    earlier "roughly 9%" figure taken from campaign observations was withdrawn:
    those arms ran as contiguous blocks, so the arm was collinear with
    wall-clock position and the rate was not attributable.

    The fix is not to stop excluding core 0 -- that exclusion is deliberate. It
    is to exclude it EARLIER. A child inherits its parent's affinity mask at
    creation, so if the daemon already excludes core 0, every task it spawns
    starts excluded, OpenMP never sees core 0 as legal, and Lasso's later
    rewrite is a no-op. There is nothing left to race.

    Starting pueued.exe by hand reintroduces the fault silently. Use this script.

.PARAMETER PueueHome
    Directory holding pueued.exe. Defaults to this script's directory, then to
    pueued on PATH. This script lives in the repository while the binaries do
    not, so the usual invocation passes -PueueHome explicitly.

.NOTES
    This script starts a daemon. It deliberately cannot stop one: killing a
    daemon can terminate active queued jobs, and a starter must not be able to
    do that as a side effect. If a daemon is already running with the wrong
    mask, this script fails and tells you to perform a separately authorized
    clean shutdown.
#>
[CmdletBinding()]
param([string]$PueueHome)

$ErrorActionPreference = "Stop"

# --- 0. Locate the daemon ----------------------------------------------------
$daemon = $null
foreach ($candidate in @(
    $(if ($PueueHome) { Join-Path $PueueHome "pueued.exe" }),
    (Join-Path $PSScriptRoot "pueued.exe")
)) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) { $daemon = $candidate; break }
}
if (-not $daemon) {
    $onPath = Get-Command pueued -ErrorAction SilentlyContinue
    if ($onPath) { $daemon = $onPath.Source }
}
if (-not $daemon) {
    Write-Error ("pueued.exe not found. Pass -PueueHome <directory>, put it " +
                 "beside this script, or place it on PATH.")
    exit 2
}

# --- 1. Derive the target mask FIRST, before considering any daemon ----------
#
# Derive from the SYSTEM affinity mask rather than from a processor count.
# Counting is unsafe: [Math]::Pow(2,n)-1 goes through Double and is silently
# off by one at n=54..62 (2^54-1 returns ...984 instead of ...983, with no
# error) and throws when cast to Int64 at n>=63. The system mask is what the
# OS actually permits, needs no arithmetic, and is correct even when a machine
# does not expose every processor.
Add-Type -Namespace Win32 -Name Aff -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool GetProcessAffinityMask(IntPtr h, out UIntPtr proc, out UIntPtr sys);
[DllImport("kernel32.dll")]
public static extern IntPtr GetCurrentProcess();
[DllImport("kernel32.dll")]
public static extern ushort GetActiveProcessorGroupCount();
'@ -ErrorAction Stop

[UIntPtr]$procMask = 0
[UIntPtr]$sysMask = 0
if (-not [Win32.Aff]::GetProcessAffinityMask([Win32.Aff]::GetCurrentProcess(),
                                             [ref]$procMask, [ref]$sysMask)) {
    Write-Error "GetProcessAffinityMask failed: $([ComponentModel.Win32Exception]::new().Message)"
    exit 2
}

# A single affinity mask addresses one processor group only. Above 64 logical
# processors Windows uses multiple groups and no single mask can span them, so
# refuse rather than silently affinitize the daemon to group 0.
$groups = [Win32.Aff]::GetActiveProcessorGroupCount()
if ($groups -ne 1) {
    Write-Error ("This host reports $groups processor groups. A single affinity " +
                 "mask cannot span groups, so this script cannot set the intended " +
                 "mask safely. Configure the exclusion through Process Lasso or a " +
                 "group-aware launcher instead.")
    exit 2
}

$sys = [uint64]$sysMask
$target = $sys -band (-bnot [uint64]1)
if ($target -eq 0) {
    Write-Error "System affinity mask 0x$('{0:X}' -f $sys) leaves nothing after excluding core 0."
    exit 2
}
Write-Host ("System mask 0x{0:X}  ->  target mask 0x{1:X} (core 0 excluded)" -f $sys, $target)

# --- 2. An existing daemon is verified, never killed -------------------------
$running = @(Get-Process -Name pueued -ErrorAction SilentlyContinue)
if ($running.Count -gt 1) {
    Write-Error ("$($running.Count) pueued processes are running (PIDs " +
                 "$($running.Id -join ', ')). Resolve that before starting one.")
    exit 1
}
if ($running.Count -eq 1) {
    $existing = [uint64][int64]$running[0].ProcessorAffinity
    if ($existing -eq $target) {
        Write-Host ("pueued already running (PID $($running[0].Id)) with the correct " +
                    "mask 0x{0:X}. Nothing to do." -f $existing) -ForegroundColor Green
        exit 0
    }
    Write-Error ("pueued is running (PID $($running[0].Id)) with mask " +
                 "0x$('{0:X}' -f $existing), but the target is 0x$('{0:X}' -f $target). " +
                 "Tasks it spawns will inherit the wrong mask and can hit the OpenMP " +
                 "race. This script will not kill a daemon, because that can terminate " +
                 "active queued jobs. Drain the queue and perform a separately " +
                 "authorized clean shutdown, then run this script again.")
    exit 1
}

# --- 3. Start with the mask applied at creation ------------------------------
#
# `start /affinity` is the only built-in way to set affinity at creation, so cmd
# is the vehicle. It must be launched via Start-Process, not the call operator:
# the call operator makes PowerShell create stdout/stderr pipes for cmd, the
# daemon inherits those handles, and PowerShell then blocks reading them until
# every holder exits -- which never happens for a daemon. Both `start /b` and a
# detached window hang for that reason; the console is not the problem, the
# inherited pipe is. Start-Process without -Wait creates no pipes.
$hex = "{0:X}" -f $target
Start-Process -FilePath "cmd.exe" `
              -ArgumentList "/c", "start `"pueued`" /affinity $hex `"$daemon`"" `
              -WindowStyle Hidden
Start-Sleep -Seconds 3

# --- 4. Verify, and fail closed ---------------------------------------------
$now = @(Get-Process -Name pueued -ErrorAction SilentlyContinue)
if ($now.Count -ne 1) {
    Write-Error "Expected exactly one pueued after start; found $($now.Count)."
    exit 1
}
$actual = [uint64][int64]$now[0].ProcessorAffinity
Write-Host ("pueued PID {0} affinity 0x{1:X}" -f $now[0].Id, $actual)
if ($actual -ne $target) {
    Write-Error ("Affinity is 0x{0:X}, expected 0x{1:X}. Tasks would inherit core 0 " +
                 "and can hit the OpenMP race. Investigate before queueing work." -f $actual, $target)
    exit 1
}
Write-Host "Daemon started with core 0 excluded. Queued tasks inherit this mask." -ForegroundColor Green
exit 0
