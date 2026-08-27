$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wrapper = Join-Path $repoRoot "scripts\pueue_wrap.ps1"

function Invoke-Wrapper {
    param(
        [string]$Json,
        [string]$Root = $repoRoot,
        [switch]$SelfCheck,
        [string]$Wrapper = $wrapper
    )

    $previous = $env:PUEUE_TASK_JSON
    $previousErrorAction = $ErrorActionPreference
    try {
        $env:PUEUE_TASK_JSON = $Json
        $ErrorActionPreference = "Continue"
        $arguments = @("-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                       "-File", $Wrapper, "-RepoRoot", $Root)
        if ($SelfCheck) { $arguments += "-SelfCheck" }
        $output = (& powershell.exe @arguments 2>&1 | Out-String)
        [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = $output }
    } finally {
        $ErrorActionPreference = $previousErrorAction
        $env:PUEUE_TASK_JSON = $previous
    }
}

function Assert-Result {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

$result = Invoke-Wrapper -Json '["cmd.exe","/d","/c","exit 7"]'
Assert-Result ($result.ExitCode -eq 7) "environment JSON lost child exit 7"

$result = Invoke-Wrapper -Json '["whoami.exe"]'
Assert-Result ($result.ExitCode -eq 0) "one-element task failed"

$result = Invoke-Wrapper -Json '"whoami.exe"' -SelfCheck
Assert-Result ($result.ExitCode -ne 0) "scalar JSON was accepted"
Assert-Result ($result.Output -match "JSON ARRAY") "scalar rejection was unclear"

$result = Invoke-Wrapper -Json '["whoami.exe"]' -SelfCheck `
    -Root (Split-Path -Parent $repoRoot)
Assert-Result ($result.ExitCode -ne 0) "mismatched repository root was accepted"
Assert-Result ($result.Output -match "does not match") "root rejection was unclear"

$tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$tempRoot = Join-Path $tempBase ("rwkv-pueue-test-" + [guid]::NewGuid())
try {
    New-Item -ItemType Directory -Path (Join-Path $tempRoot "scripts") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $tempRoot ".venv\Scripts") -Force | Out-Null
    Copy-Item $wrapper (Join-Path $tempRoot "scripts\pueue_wrap.ps1")
    Copy-Item (Join-Path $repoRoot "scripts\init_cuda_env.ps1") `
        (Join-Path $tempRoot "scripts\init_cuda_env.ps1")
    Copy-Item (Join-Path $env:WINDIR "System32\where.exe") `
        (Join-Path $tempRoot ".venv\Scripts\python.exe")

    $result = Invoke-Wrapper -Json '["whoami.exe"]' -SelfCheck -Root $tempRoot `
        -Wrapper (Join-Path $tempRoot "scripts\pueue_wrap.ps1")
    Assert-Result ($result.ExitCode -ne 0) "uncallable venv Python was accepted"
    Assert-Result ($result.Output -match "Python interpreter") `
        ("uncallable Python failure did not name the interpreter:`n" +
         $result.Output)
} finally {
    $resolvedTemp = [IO.Path]::GetFullPath($tempRoot)
    Assert-Result ($resolvedTemp.StartsWith($tempBase)) "unsafe temporary path"
    Remove-Item -LiteralPath $resolvedTemp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Pueue wrapper regression passed." -ForegroundColor Green
