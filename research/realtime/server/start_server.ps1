# Starts the local CS2 test server with bots that fight without a human, GOTV auto-recording,
# and the telemetry plugin. Redeploys the latest plugin build first (the server must be stopped).
#   powershell -File start_server.ps1 -LanIp <this PC's LAN IP> -RconPassword <password> [-Map de_dust2]
# Both -LanIp and -RconPassword are required: no default is stored in the repository.
# Stop it cleanly over RCON with `tv_stoprecord` then `quit`, so the GOTV demo is finalized.
param(
    [string]$Server = 'D:\cs2server',
    [string]$Map = 'de_dust2',
    [Parameter(Mandatory)][string]$RconPassword,
    [Parameter(Mandatory)][string]$LanIp,
    [ValidateRange(0, 3)][int]$BotDifficulty = 2,   # 0 easy, 1 normal, 2 hard, 3 expert
    [ValidateRange(0, 20)][int]$BotQuota = 8,       # 0 for humans-only sessions (FRIEND_SESSIONS.md)
    [switch]$Competitive,                           # standard round timings (no bot-test shortcuts)
    [string]$PluginBuild = (Join-Path $PSScriptRoot '..\plugin\bin\Release\net10.0\AnticheatTelemetry.dll')
)
$ErrorActionPreference = 'Stop'
if (Get-Process cs2 -ErrorAction SilentlyContinue) { throw 'a cs2 process is already running; stop it first' }
$plugin = Join-Path $Server 'game\csgo\addons\counterstrikesharp\plugins\AnticheatTelemetry\AnticheatTelemetry.dll'
Copy-Item $PluginBuild $plugin -Force
$launch = @(
    '-dedicated', '-console', '-insecure', '-usercon',   # -insecure is required for server plugins
    "-ip $LanIp",                                         # bind to the home-network address so other PCs can join
    '+game_type 0', '+game_mode 1', "+map $Map",
    '+sv_hibernate_when_empty 0', '+mp_do_warmup_period 0',
    '+bot_join_after_player 0', '+bot_quota_mode normal', "+bot_quota $BotQuota", "+bot_difficulty $BotDifficulty",
    '+tv_enable 1', '+tv_autorecord 1',
    "+rcon_password $RconPassword", '+sv_lan 1')
if (-not $Competitive) { $launch += @('+mp_freezetime 2', '+mp_roundtime 3') }   # bot-test shortcuts
$launch = $launch -join ' '
$exe = Join-Path $Server 'game\bin\win64\cs2.exe'
$p = Start-Process -FilePath $exe -ArgumentList $launch -WorkingDirectory (Split-Path $exe) `
    -RedirectStandardOutput (Join-Path $Server 'server_stdout.log') `
    -RedirectStandardError (Join-Path $Server 'server_stderr.log') -PassThru
Write-Host "server started (pid $($p.Id)); log: $Server\server_stdout.log"
