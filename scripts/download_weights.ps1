param(
    [string]$RepoId = "Ruihang/Wan-Move-14B-480P",
    [string]$LocalDir = ".\Wan-Move-14B-480P",
    [switch]$ForceInstallCli
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

if ($ForceInstallCli) {
    python -m pip install --upgrade "huggingface_hub[cli]"
}

$hasCli = Get-Command huggingface-cli -ErrorAction SilentlyContinue
if (-not $hasCli) {
    python -m pip install --upgrade "huggingface_hub[cli]"
}

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

huggingface-cli download $RepoId --local-dir $LocalDir
