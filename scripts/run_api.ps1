param(
    [string]$WeightsDir = ".\Wan-Move-14B-480P",
    [string]$OutputsDir = ".\outputs",
    [int]$Port = 8000,
    [string]$ImageName = "wan-move-api:latest",
    [switch]$Build
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

$weightsPath = Resolve-Path -LiteralPath $WeightsDir -ErrorAction SilentlyContinue
if (-not $weightsPath) {
    throw "Weights directory not found: $WeightsDir. Run scripts\download_weights.ps1 first."
}

New-Item -ItemType Directory -Force -Path $OutputsDir | Out-Null

$env:WAN_MOVE_IMAGE = $ImageName
$env:WAN_MOVE_WEIGHTS_DIR = (Resolve-Path -LiteralPath $WeightsDir).Path
$env:WAN_MOVE_OUTPUTS_DIR = (Resolve-Path -LiteralPath $OutputsDir).Path
$env:WAN_MOVE_PORT = [string]$Port

if ($Build) {
    docker compose up --build wan-move-api
} else {
    docker compose up wan-move-api
}
