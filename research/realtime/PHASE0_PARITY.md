# Phase 0 — Live-capture parity gate

**Date:** 2026-10-11 · **Code revision:** `dabbd65` (working tree: journal edits only) · **Status:** GO, with three conditions to settle empirically in Phase 1.

## 1. Reference match

| Field | Value |
|---|---|
| Match | `no_cheater_present/1` (de_inferno, 360 hurt events) |
| Split | `validation` (`experiments/cs2cd_v1/match_splits.csv`) |
| Windows | `examples/example_validation_player.npz` — 16 encounters, `[16, 256, 39]`, raw (pre-transform) |
| Training reference stats | `experiments/cs2cd_v1/normalization.json` (mean/std of transformed channels, training ticks only) |
| Source extractor | `anticheat/cs2_data.py::extract_match` (CS2CD Parquet + event JSON) |

Raw CS2CD Parquet is **not** stored locally (only `data/cs2cd/metadata/`). A CS2CD match also cannot be replayed through a live server plugin. Phase 1 parity therefore needs two references:

- **Tick-level:** a GOTV demo recorded with `tv_record` during the same captured session and run through `dem_extractor.py`. That extractor already has synthetic parity with `cs2_data.py` (Entry 13).
- **Distribution-level:** the reference windows and statistics above. These also cover channels GOTV does not record: velocity, flash and footsteps (Entry 14).

## 2. Key design point

Only **~22 raw per-tick fields + 4 events** come from the server. All 39 channels are derived from them by the existing math in `cs2_data.py` (`wrap`, `difference`, `time_since`, eye-height approximation, closing/lateral projection, weapon sets). The plugin must log **raw state only** and must not compute features in C#. Parity then reduces to raw-field parity plus tick alignment.

## 3. Raw-field mapping

CSS names are planned sources. **Each must be confirmed against the installed CounterStrikeSharp version in Phase 1.** Generic schema access (`Schema.GetSchemaValue`) is the fallback for any field without a typed wrapper.

| Training column (Parquet) | Planned CSS source | Risk | Note |
|---|---|---|---|
| `tick` | `Server.TickCount` | **High** | Alignment with state and events; see C1 |
| `steamid` | `controller.SteamID` | Medium | Bots have SteamID 0 and are **dropped** by training semantics (`if not player`) |
| `pitch`, `yaw` | `pawn.EyeAngles.X/.Y` (`m_angEyeAngles`) | Low | Yaw range may be [0,360) server-side; `wrap()` makes deltas and target error range-invariant. Pitch: positive = down in both |
| `X`,`Y`,`Z` | `pawn.AbsOrigin` | Low | Same origin as demo `m_vecOrigin` |
| `velocity_X/Y/Z` | `pawn.AbsVelocity` **and** derived `(pos[t]-pos[t-1])*64` | **High** | Training source unknown; see C2 |
| `is_alive` | `pawn.LifeState == ALIVE` | Low | |
| `is_warmup_period` | `GameRules.WarmupPeriod` | Low | |
| `team_num` | `controller.TeamNum` | Low | |
| `total_rounds_played` | `GameRules.TotalRoundsPlayed` | Low | |
| `is_scoped` | `pawn.IsScoped` | Low | |
| `duck_amount` | `pawn.MovementServices.DuckAmount` | Low | Also drives eye height in `dz` |
| `is_airborne` | `!(pawn.Flags & FL_ONGROUND)` | Medium | demoparser2 definition may differ on ladders and water; see C3 |
| `health` | `pawn.Health` | Low | |
| `armor_value` | `pawn.ArmorValue` | Low | |
| `fl_recoil_idx` | active weapon `m_flRecoilIndex` | Low | Real demos expose `m_flRecoilIndex` (Entry 14) |
| `aim_punch_angle[0/1]` | `pawn.AimPunchAngle.X/.Y` | Medium | CS2 also networks punch tick base and fraction; confirm raw field equality against demo |
| `shots_fired` | `pawn.ShotsFired` (`m_iShotsFired`) | Low | **Correction:** the research doc says "counter from weapon_fire events". Training uses the entity property |
| `active_weapon_ammo` | active weapon `Clip1` | Low | |
| `active_weapon_name` | active weapon `DesignerName`, minus `weapon_` | Low | Weapon sets contain both internal and display names |
| event `weapon_fire` | `EventWeaponFire` (userid, weapon) | Low | Filtered by `GUNS` after stripping `weapon_` |
| event `player_footstep` | `EventPlayerFootstep` | Medium | Missing in GOTV; must be logged by the plugin |
| event `player_blind` | `EventPlayerBlind` (`blind_duration`) | Medium | Same; feeds `flash_remaining` decay |
| event `player_hurt` | `EventPlayerHurt` (attacker, userid, weapon, `dmg_health`, `health`) | **High** | Anchor of every window; see C1 |

## 4. Channel checklist (39)

Derived from raw fields with existing code; no new math.

| # | Channel | Derived from | Parity depends on |
|---|---|---|---|
| 0–1 | delta_yaw, delta_pitch | yaw, pitch | angles, C1 |
| 2–3 | target_error_yaw/pitch | angles, positions, duck_amount | angles, origin |
| 4 | shot | weapon_fire ticks | C1 (event/state offset) |
| 5–8 | yaw/pitch acceleration, target yaw/pitch rate | ch 0–3 | as above |
| 9–10 | distance, relative_height | positions, duck_amount | origin |
| 11–16 | attacker/victim speed, closing, lateral, vertical speeds | velocity, positions | **C2** |
| 17–19 | scoped, crouch, airborne | entity props | C3 (airborne) |
| 20 | flash_remaining | player_blind | event logging |
| 21–23 | health, armor, recoil_index | entity props | — |
| 24–25 | punch_pitch, punch_yaw | aim punch | punch field |
| 26–27 | shots_in_burst, ammo | `m_iShotsFired`, `Clip1` | — |
| 28 | time_since_shot | weapon_fire | C1 |
| 29–31 | victim_shot, victim_footstep, time_since_victim_noise | weapon_fire, footstep | event logging, C1 |
| 32–33 | victim_crouch, victim_health | entity props | **C1 leakage check** |
| 34–38 | weapon one-hot | weapon name | — |

Reference ranges from the validation example (raw): speeds capped near 250 u/s; vertical speeds up to 865 u/s; flash up to 0.77 s (non-zero, so CS2CD did record `player_blind`); punch_pitch down to −12; shots_in_burst and recoil_index up to 7; time_since features capped at 5 s.

## 5. Conditions (none are blocking, because each has a code-level fix)

**C1 — Tick alignment and hit-tick leakage (highest priority).** The tick label CSS's `OnTick` attaches to state may be off by one from the demo label, and from the tick of events fired during simulation. If state is labeled one tick early, the window `[T-256, T)` would contain the post-damage state: `victim_health` would already show the hit. *Fix:* log state at a fixed point in the frame, apply one constant offset, and assert in a test that `victim_health` at the last window tick equals `player_hurt.health + dmg_health` (pre-hit health). Measure the offset by matching per-tick yaw against the same session's GOTV demo.

**C2 — Velocity source.** Training `velocity_X/Y/Z` may be networked velocity or a demoparser2 position-difference value. GOTV produced no velocity at all (Entry 14), which suggests the CS2CD values came from a different recording path. *Fix:* log both `AbsVelocity` and positions, then compare each candidate's speed and vertical-speed distributions with the reference (cap near 250 u/s, vertical tails). Pick the closer one and record the choice.

**C3 — Airborne definition.** Compare `FL_ONGROUND` against the demo's `is_airborne` per tick in the GOTV comparison; adopt whichever definition matches.

## 6. Non-parity risks (do not block capture, but limit interpretation)

- **Bots:** SteamID 0, excluded by training semantics, and out-of-distribution behaviour. Use bots only for capture testing (keyed by slot); never score them.
- **Domain shift:** friends' skill, server config and maps differ from CS2CD. Entry 14 showed AWP flicks inflate scores.
- **Thresholds:** calibrated on CS2CD prevalence. Live outputs are review signals, not validated rates. No retuning without labelled data.
- **Consent:** every participant whose telemetry is logged (victims included) must agree first.

## 7. Decision

**GO to Phase 1**, scoped to: plugin logs raw fields + 4 events to versioned NDJSON. Phase 1 must deliver the C1 offset measurement and leakage test, the C2 velocity comparison, and the C3 airborne comparison before any scoring. Phase 1 needs approval for the dedicated-server download (~60–65 GB) and Metamod/CSS install.
