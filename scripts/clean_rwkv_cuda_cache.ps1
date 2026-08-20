# Remove compiled-extension artifacts left behind by a broken or stale build.
#
# Two kinds of artifact can wedge the repository, and both survive a normal
# `git clean` of the working tree because they live outside it or inside a
# submodule:
#
#   1. The JIT build cache for the fused RWKV-7 kernel, under torch_extensions.
#      A stale or half-written cache entry makes load_rwkv7_cuda_kernel() fail
#      for the rest of the process.
#   2. A compiled rosa_soft `_C` extension inside external/rosa_soft. This one
#      is worse than useless on Windows: rosa_compute.rosa_compat puts that
#      directory on sys.path, and MSVC links the extension without exporting
#      PyInit__C, so every `import rosa_compute` then dies with
#      "dynamic module does not define module export function (PyInit__C)"
#      and the whole test suite fails to collect. rosa_soft runs fine in
#      reference mode with no compiled extension present.
#
# Safe to run at any time: it only deletes build output, never source.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$removed = 0

function Remove-Artifact {
    param([string]$Path, [string]$Why)

    if (Test-Path -LiteralPath $Path) {
        Write-Host "Removing $Path  ($Why)"
        Remove-Item -LiteralPath $Path -Recurse -Force
        $script:removed++
    }
}

# 1. Cached JIT builds of the pinned RWKV-7 kernel.
$cacheRoots = @(
    "$env:LOCALAPPDATA\torch_extensions",
    "$env:TEMP\torch_extensions"
)

foreach ($root in $cacheRoots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    # NOTE: not $matches -- that is an automatic variable clobbered by -match.
    $cached = Get-ChildItem -Path $root -Directory -Recurse `
        -Filter "rwkv7_clampw_exp0" -ErrorAction SilentlyContinue
    foreach ($dir in $cached) {
        Remove-Artifact -Path $dir.FullName -Why "cached RWKV-7 kernel build"
    }
}

# 2. Compiled rosa_soft extensions and their build trees. The submodule is
#    imported from source, so a compiled _C here is always a leftover.
$rosaSoft = Join-Path $repoRoot "external\rosa_soft"
if (Test-Path -LiteralPath $rosaSoft) {
    $compiled = Get-ChildItem -Path (Join-Path $rosaSoft "rosa_soft") `
        -Include "_C*.pyd", "_C*.so" -File -Recurse -ErrorAction SilentlyContinue
    foreach ($file in $compiled) {
        Remove-Artifact -Path $file.FullName -Why "compiled rosa_soft extension"
    }

    Remove-Artifact -Path (Join-Path $rosaSoft "build") -Why "rosa_soft build tree"
    $eggInfo = Get-ChildItem -Path $rosaSoft -Directory -Filter "*.egg-info" `
        -ErrorAction SilentlyContinue
    foreach ($dir in $eggInfo) {
        Remove-Artifact -Path $dir.FullName -Why "rosa_soft egg-info"
    }

    # A __pycache__ holding the deleted extension's import record makes the
    # next import fail in a way that looks unrelated to the build.
    $caches = Get-ChildItem -Path $rosaSoft -Directory -Recurse `
        -Filter "__pycache__" -ErrorAction SilentlyContinue
    foreach ($dir in $caches) {
        Remove-Artifact -Path $dir.FullName -Why "stale rosa_soft __pycache__"
    }
}

if ($removed -eq 0) {
    Write-Host "Nothing to clean; no stale build artifacts found."
} else {
    Write-Host "Cleaned $removed artifact(s)."
    Write-Host "The RWKV-7 kernel rebuilds on next use (~20s)."
}
