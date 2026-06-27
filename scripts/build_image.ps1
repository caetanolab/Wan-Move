param(
    [string]$ImageName = "wan-move-api:latest",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

$env:WAN_MOVE_IMAGE = $ImageName

$args = @("compose", "build")
if ($NoCache) {
    $args += "--no-cache"
}
$args += "wan-move-api"

docker @args
