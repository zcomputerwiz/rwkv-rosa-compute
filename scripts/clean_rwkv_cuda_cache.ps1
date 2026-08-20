$ErrorActionPreference = "Stop"

$roots = @(
    "$env:LOCALAPPDATA\torch_extensions",
    "$env:TEMP\torch_extensions"
)

$matches = @()

foreach ($root in $roots) {
    if (Test-Path $root) {
        $matches += Get-ChildItem `
            -Path $root `
            -Directory `
            -Recurse `
            -Filter "rwkv7_clampw_exp0" `
            -ErrorAction SilentlyContinue
    }
}

if (-not $matches) {
    Write-Host "No cached rwkv7_clampw_exp0 builds found."
    exit 0
}

foreach ($dir in $matches) {
    Write-Host "Removing $($dir.FullName)"
    Remove-Item -Recurse -Force $dir.FullName
}

Write-Host "RWKV CUDA extension cache cleaned."