# Installs Metamod:Source and CounterStrikeSharp into an existing CS2 dedicated server and
# deploys the telemetry plugin. Safe to re-run. Run after SteamCMD finishes downloading.
#   powershell -File setup_server.ps1 [-Server D:\cs2server] [-Downloads D:\downloads]
param(
    [string]$Server = 'D:\cs2server',
    [string]$Downloads = 'D:\downloads',
    [string]$PluginBuild = (Join-Path $PSScriptRoot '..\plugin\bin\Release\net10.0')
)
$ErrorActionPreference = 'Stop'
$csgo = Join-Path $Server 'game\csgo'
if (-not (Test-Path (Join-Path $csgo 'gameinfo.gi'))) { throw "gameinfo.gi not found in $csgo; is the server download complete?" }

# 1. Copy both addon trees (archives contain a top-level addons\ folder).
foreach ($src in "$Downloads\mms\x\addons", "$Downloads\css\x\addons") {
    if (-not (Test-Path $src)) { throw "missing extracted archive: $src" }
    Copy-Item $src -Destination $csgo -Recurse -Force
}

# 2. Metamod loads only if gameinfo.gi lists it first under SearchPaths. A CS2 update rewrites
#    gameinfo.gi, so re-run this script after updates.
$gi = Join-Path $csgo 'gameinfo.gi'
$text = Get-Content $gi -Raw
if ($text -notmatch 'csgo/addons/metamod') {
    Copy-Item $gi "$gi.bak" -Force
    $patched = $text -replace '(?m)^(\s*)(Game_LowViolence\s+csgo_lv.*)$', "`$1`$2`r`n`$1Game`tcsgo/addons/metamod"
    if ($patched -eq $text) { throw 'could not find the Game_LowViolence line to patch; edit gameinfo.gi by hand' }
    Set-Content $gi $patched -NoNewline
    Write-Host 'patched gameinfo.gi (backup: gameinfo.gi.bak)'
}

# 3. Deploy the telemetry plugin and an empty consent file (recording stays off until filled).
$plugin = Join-Path $csgo 'addons\counterstrikesharp\plugins\AnticheatTelemetry'
New-Item -ItemType Directory -Force $plugin | Out-Null
Copy-Item (Join-Path $PluginBuild 'AnticheatTelemetry.dll') $plugin -Force
$consent = Join-Path $plugin 'consent.txt'
if (-not (Test-Path $consent)) { New-Item -ItemType File $consent | Out-Null }
Write-Host "plugin deployed to $plugin"
Write-Host 'consent.txt is empty: add one SteamID64 per consenting player (bots are always recorded for tests).'
