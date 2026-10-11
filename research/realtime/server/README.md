# Running the local CS2 test server

Local test server for the real-time telemetry work (Phase 1). Records raw player state and
events through the CounterStrikeSharp plugin, plus a GOTV demo of the same session.
Records only consenting players; bots are recorded for testing only.

## Before you start

- The CS2 dedicated server is installed at `D:\cs2server`.
- Steam is installed and signed in on the laptop (`D:\Steam`).
- The plugin is built (`research\realtime\plugin\bin\Release\net10.0\AnticheatTelemetry.dll`).
- Your PC and the laptop are on the same Wi-Fi network.
- Each human player's SteamID64 is in `consent.txt` (see "Consent" below).

## 1. Start the server

From the project root, in PowerShell:

```powershell
research\realtime\server\start_server.ps1 -LanIp 192.168.1.132 -RconPassword "your-password"
```

- `-LanIp` is the laptop's address on the home network (check with `ipconfig`, adapter *Wi-Fi*).
- `-RconPassword` is a password you choose. It is required and is not stored in the repo.
- The script copies the latest plugin build to the server, then starts it with bots, one
  auto-recorded GOTV demo, and the plugin. It does not show a window.

Wait about 30 seconds. The server log should show `logged on to Steam` and
`Finished loading plugin AnticheatTelemetry`.

## 2. Join from your PC

1. Start Steam and sign in.
2. Start CS2 and open the console (`~` key). If it's disabled, enable it in
   Settings → Game → Enable Developer Console.
3. Type:

   ```
   connect 192.168.1.132:27015
   ```

If the join fails, check that both PCs are on the same network and allow the Windows
Firewall prompt on the laptop.

## 3. Write commands to the server

The server runs without a visible window, so you can't type into it directly. Use the RCON
password from step 1 and the in-game console on your PC:

```
rcon_address 192.168.1.132:27015
rcon_password your-password
rcon status
```

After that, `rcon <command>` runs a command on the server. Useful commands:

| Command | Effect |
|---|---|
| `rcon status` | Players, bots, map |
| `rcon bot_add_t` | Add a T-side bot |
| `rcon bot_add_ct` | Add a CT-side bot |
| `rcon mp_autoteambalance 0` and `rcon mp_limitteams 0` | Allow uneven teams (needed before adding bots that get "Team is stacked") |
| `rcon mp_restartgame 1` | Restart the round |
| `rcon tv_stoprecord` | Finish the GOTV demo |
| `rcon quit` | Shut the server down |

Stop the server with `tv_stoprecord` first, then `quit`. Don't close the window or end the
process: that can lose the GOTV demo.

`mp_warmup_end` and `mp_restartgame` appear to cut the automatic GOTV recording short (a
session that used them kept only 12 s of demo). If you need the demo for a parity check,
avoid those two, or start a new recording afterwards with `rcon tv_record <name>`.

## Live scoring (Phase 3)

With the server running, from the `research` folder:

```powershell
..\venv\Scripts\python.exe -I -W ignore -u realtime\live_scoring.py "D:\cs2server\game\csgo\addons\counterstrikesharp\plugins\AnticheatTelemetry" --include-bot-victims
```

- It follows the newest capture and prints one line per encounter: attacker, victim, weapon,
  encounter score, player-match score, review flags, processing delay.
- `--include-bot-victims` is needed when the opponents are bots (pipeline testing only).
- Flags are marked *provisional* below 10 encounters.
- The log is written to `local_runs\live_scores.jsonl` (change with `--log`).
- Stop it with Ctrl+C.
- To re-run a finished capture: add `--replay` and give the capture file.

Scores are review signals calibrated on CS2CD, not cheating verdicts.

If the in-game `rcon` commands are rejected, tell the project maintainer. The RCON path
through the in-game console is not yet tested.

## 4. Where to find the output

| What | Location |
|---|---|
| Server log (startup, errors, plugin messages) | `D:\cs2server\server_stdout.log` |
| Server errors | `D:\cs2server\server_stderr.log` |
| Telemetry capture (plugin output) | `D:\cs2server\game\csgo\addons\counterstrikesharp\plugins\AnticheatTelemetry\capture_*.ndjson` |
| Consent list | `...\AnticheatTelemetry\consent.txt` (same folder) |
| GOTV demos (written when recording stops) | `D:\cs2server\game\csgo\replays\auto-*.dem` |

**Check that the plugin is recording:** search the server log for `AnticheatTelemetry`.
`consent.txt missing or empty` means only bots are being recorded.

**Check that data is coming in:** the newest capture file grows while the server runs.

**Check a capture:** from the `research` folder:

```powershell
..\venv\Scripts\python.exe -I realtime\check_capture.py <path-to-capture.ndjson>
```

## Consent

- Add one SteamID64 per line to `consent.txt`. Only listed humans are recorded.
- Bots are always recorded (test only). Bots are never scored.
- Tell every player that their telemetry is recorded before they join.
- Don't publish captures or demos: they contain SteamIDs and per-tick movement.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Could not determine Steam client install directory` | Steam client not installed or signed in on the laptop. |
| `Team is stacked` when adding bots | Run `mp_autoteambalance 0` and `mp_limitteams 0` first. |
| `ConnectionRefusedError` on RCON | Wrong address. The server listens on the `-LanIp` address, not the virtual adapter. |
| Plugin loads but records only bots | `consent.txt` is empty, or the SteamID is wrong. |
| Server exits right after starting | Check `server_stdout.log` and `server_stderr.log` for the last error. |
| Capture ends in a broken line | Stopped without `quit` or the window was closed. The reader drops the last partial line automatically. |

## After a CS2 update

The server's `gameinfo.gi` can be overwritten by an update, which removes the Metamod entry.
Re-run `research\realtime\server\setup_server.ps1` to restore it.
