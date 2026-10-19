# How many players the live scorer can handle

Planning guide for sessions with friends. Measured on the author's laptop on 2026-10-19
with the code as of Phase 3 (`research/realtime/live_scoring.py`). Numbers marked
*estimate* are derived, not measured.

## Short answer

| Human players on the server | Status with the current code |
|---|---|
| up to 10 (a full 5v5) | **Supported.** Uses a small fraction of the listener's capacity. |
| 11–20 | Probably fine in normal matches; not tested. Watch the delay column. |
| more than 20 | **Not supported yet.** Could fall behind and lose hits (see "Known limits"). |

A competitive CS2 match has at most 10 players, so a normal 5v5 with friends is within
the supported range.

## What gets scored

- **Every hit between two consenting human players**, for every player at once. Nobody
  needs to spectate: the plugin records all consenting players from the server itself.
- **Hits where the victim has not consented are dropped.** Both the shooter and the
  victim must be in `consent.txt`. If one friend is missing, hits on that friend are lost.
- **Bots are never scored.** Hits on bots count only with `--include-bot-victims`, and
  those windows are for testing the pipeline: training never had bot victims.
- **Spectators** produce no encounters.

## Measured cost per hit

| Step | 10 players | 20 players | 40 players |
|---|---|---|---|
| Build the encounter window (rebuild over the 32 s buffer) | 206 ms | 277 ms | 559 ms |
| Score with the model (player-match + encounter score) | 96–173 ms | 96–173 ms | 96–173 ms |
| **Total per hit** | **≈ 0.3–0.4 s** | **≈ 0.4–0.45 s** | **≈ 0.65–0.75 s** |

- The first hit of a session takes 4–7 s extra while the model loads.
- Scoring time barely depends on how many encounters a player already has (1–51 tested).
- The listener handles hits one at a time.
- End-to-end delay also includes up to ~1 s before the plugin saves its data.

## How busy the listener gets (*estimate*)

Two hit rates were used:

- **Real matches:** CS2CD has about 19 eligible encounters per player per full match
  (141,633 encounters / 7,508 player-matches). Over a ~40 minute match that is
  about 0.5 encounters per player per minute.
- **Bot sessions (upper bound):** 3.75–5 encounters per human per minute were measured
  against bots, which are easy to hit.

| Players | Real match: share of time busy | Against bots (upper bound): share of time busy |
|---|---|---|
| 10 | ≈ 3% | ≈ 23–31% |
| 20 | ≈ 8% | ≈ 56–75% |
| 40 | ≈ 24% | ≈ 180–240% (**falls behind**) |

Above 100% the listener falls behind. Once it is more than about 64 s behind, it can
lose hits without warning (see "Known limits").

## Checklist for a session with friends

1. **Consent first.** Tell every player what is recorded (position, aim, shots, hits,
   SteamID) and why. Get their agreement before they join.
2. **Add every player's SteamID64** to
   `D:\cs2server\game\csgo\addons\counterstrikesharp\plugins\AnticheatTelemetry\consent.txt`,
   one per line. Include yourself if you play.
3. **Start (or restart) the server after editing `consent.txt`.** The plugin reads it only
   at startup.
4. **Start the listener right after the server**, not later in the session (see "Known
   limits"):

   ```powershell
   ..\venv\Scripts\python.exe -I -W ignore -u realtime\live_scoring.py "D:\cs2server\game\csgo\addons\counterstrikesharp\plugins\AnticheatTelemetry"
   ```

   Leave out `--include-bot-victims` when the session is humans only.
5. **One listener run per map.** Stop it (Ctrl+C) and start it again when the map changes.
6. **Network:** friends on the same home network join with `connect <laptop LAN IP>:27015`.
   Friends over the internet need port forwarding and `sv_lan 0`; this is **not tested**.
7. **Avoid `mp_warmup_end` and `mp_restartgame`** if you want the GOTV demo of the session;
   they appear to cut the recording short.
8. **After the session**, keep the capture, the GOTV demo and the listener log together,
   and note the date, map, players (by alias, not name) and plugin version.

## Data you can expect (*estimate*)

| Session | Eligible encounters |
|---|---|
| One full 5v5 match (~40 min), all 10 consenting | ≈ 190 |
| Two-hour evening, 10 players | ≈ 500–600 |

Each player in each match is one *player-match*: the unit the model scores and the unit
for any false-positive rate. Reporting a false-positive rate needs many player-matches
from many different people (see the project journal for the sample sizes).

## Known limits (from the Phase 3 code review)

| Limit | Effect | Workaround now | Fix in code |
|---|---|---|---|
| If the listener falls more than ~64 s behind, older hits are dropped before scoring | Hits lost without warning | Start the listener with the server; keep player count ≤ 20 | Never trim the buffer before scoring |
| Rejected windows (warmup, round boundary, missing data) are not logged | You can't see why a hit was not scored | Compare hit counts with `check_capture.py` afterwards | Log rejection counts on every update |
| The player score never resets | Several maps get mixed into one score | One listener run per map | Reset per map or round |
| A slot reused by another player can be credited to the previous owner | Rare mislabelled hit | Don't swap players mid-map | Key players by SteamID and session |
| The model is reloaded for every hit | Higher delay per hit | None needed at ≤ 10 players | Load the model once |

None of these fixes needs retraining. Fixing the first one would also remove the upper
limit on how far behind the listener can safely fall.

Scores are review signals calibrated on CS2CD, not cheating verdicts.
