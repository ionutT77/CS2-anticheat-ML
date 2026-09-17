# Private-Server Testing Pipeline for CS2CD Anti-Cheat Model

## Delivery Status — 2026-09-17

**Current deliverable: implementation with synthetic validation only. Real-demo acceptance is blocked pending a suitable authorized GOTV recording.** No real `.dem` was available or parsed during this work. The original capability and full-parity claims below are design expectations, not verified observations from a real match.

- Implemented strict telemetry extraction, NPZ export, player inference, neural encounter evidence, and UTF-8 review reports.
- Final research suite: **218 passed, 5 warnings**. Extraction-to-model tests use synthetic/mocked telemetry with the real saved model. Parser boundary mocks and synthetic Source 2 timing records do not establish real-demo compatibility.
- Public API is `extract_dem(...)`, not `extract_demo(...)`.
- Installed `demoparser2` 0.42.0 provides aim punch as `aim_punch_angle`; its header lacks playback timing. Added a standard-library `CDemoFileInfo` reader. Unknown/non-64-Hz timing and compressed file-info records are rejected; no timing is assumed.
- Missing required properties abort extraction; incomplete/nonfinite windows are rejected, never default-filled. Additional strict checks cover duplicate ticks and both players' round boundaries.
- Per-encounter sigmoid neural logits are **uncalibrated component evidence**, not player-ensemble probabilities. Existing `cs2_data.py`, `cs2_inference.py`, and trained artifacts remain unchanged.
- `--consent-ids` is an attacker scoring filter, not a telemetry collection boundary. All participants must consent to victim telemetry use.
- Remaining limitations: aggregate rather than individual rejection logs; no automated comparison to transformed training normalization; saved scikit-learn 1.6.1 artifact warning under installed 1.9.0; no measured private-server false-positive rate.

**Next acceptance gate:** provide a consenting private-server GOTV demo containing at least 256 contiguous pre-impact ticks and known encounter anchors. Run the actual parser and scoring CLI, verify timing/properties/events against the recording, compare encounter alignment and feature values against independently checked ground truth, and inspect distributions before interpreting scores. Until that gate passes, do not describe this pipeline as real-demo validated or deployment-ready. Further synthetic-test expansion is out of scope for this delivery.

See `PROJECT_JOURNAL.md`, Entry 13, for implementation details, test evidence, and limitations. All changes remain local; nothing was committed or pushed.

## Goal

Implement the smallest viable pipeline to extract 39-feature encounter windows from a private CS2 server match demo, produce NPZ files compatible with the existing `predict_cs2cd.py` inference, and generate human-readable review reports — all in shadow/review mode with no automated punishments.

---

## Phase 1: Capability Audit

### Telemetry collection method: **GOTV demo parsing via `demoparser2`**

#### Why demo parsing over a CounterStrikeSharp plugin

| Criterion | CounterStrikeSharp plugin | GOTV demo parsing (`demoparser2`) |
| :--- | :--- | :--- |
| **Setup complexity** | Must install Metamod:Source + CSS + compile C# plugin + deploy | `pip install demoparser2`, point at `.dem` file |
| **Tick rate** | Server tickrate (64 Hz) | Demo tickrate (typically 64 Hz for GOTV) |
| **Property access** | Full engine state via `CCSPlayerPawn` | Full entity snapshot via `parse_ticks()` |
| **Offline analysis** | Requires running server | Analyze after match, anywhere |
| **Reproducibility** | Must capture during live play | Demo file is a permanent record |
| **Risk of disruption** | Plugin bugs could crash server | Zero — post-hoc analysis only |
| **Training pipeline match** | CS2CD was built from Parquet files derived from demo data | Direct match — same data source type |

> [!IMPORTANT]
> **Decision: Use `demoparser2` to parse GOTV demo files.** This is the safest, most reproducible approach. The CS2CD training pipeline itself was built from demo-derived Parquet+JSON files. Parsing demos post-hoc ensures zero server disruption and perfect reproducibility. The private server records GOTV demos automatically when configured with `tv_enable 1`.

### Feature availability matrix — all 39 features

Every feature is assessed against what `demoparser2.parse_ticks()` and `demoparser2.parse_events()` can provide from a CS2 `.dem` file.

| # | Feature | Source in `cs2_data.py` | `demoparser2` availability | Status |
| :---: | :--- | :--- | :--- | :---: |
| 0 | `delta_yaw` | `difference(yaw, circular=True)` | `yaw` via `parse_ticks(["yaw"])` | ✅ Direct |
| 1 | `delta_pitch` | `difference(pitch)` | `pitch` via `parse_ticks(["pitch"])` | ✅ Direct |
| 2 | `target_error_yaw` | `wrap(bearing - attacker_yaw)` | Requires X,Y,Z of both players + yaw | ✅ Derivable |
| 3 | `target_error_pitch` | `elevation - attacker_pitch` | Requires X,Y,Z, duck_amount, pitch | ✅ Derivable |
| 4 | `shot` | `weapon_fire` events | `parse_events("weapon_fire")` | ✅ Direct |
| 5 | `yaw_acceleration` | `difference(delta_yaw)` | Derived from yaw | ✅ Derivable |
| 6 | `pitch_acceleration` | `difference(delta_pitch)` | Derived from pitch | ✅ Derivable |
| 7 | `target_yaw_rate` | `difference(bearing, circular=True)` | Derived from positions | ✅ Derivable |
| 8 | `target_pitch_rate` | `difference(elevation)` | Derived from positions + duck | ✅ Derivable |
| 9 | `distance` | `sqrt(horizontal² + dz²)` | Derived from X,Y,Z + duck_amount | ✅ Derivable |
| 10 | `relative_height` | `dz` (eye height difference) | Derived from Z + duck_amount | ✅ Derivable |
| 11 | `attacker_speed` | `hypot(velocity_X, velocity_Y)` | `parse_ticks(["velocity_X","velocity_Y"])` | ✅ Direct |
| 12 | `victim_speed` | `hypot(velocity_X, velocity_Y)` | Same, for victim | ✅ Direct |
| 13 | `closing_speed` | `-(rvx·dx + rvy·dy) / horizontal` | Derived from velocities + positions | ✅ Derivable |
| 14 | `lateral_speed` | `(rvx·dy - rvy·dx) / horizontal` | Derived from velocities + positions | ✅ Derivable |
| 15 | `attacker_vertical_speed` | `velocity_Z` | `parse_ticks(["velocity_Z"])` | ✅ Direct |
| 16 | `victim_vertical_speed` | `velocity_Z` | Same, for victim | ✅ Direct |
| 17 | `scoped` | `is_scoped` | `parse_ticks(["is_scoped"])` | ✅ Direct |
| 18 | `crouch` | `duck_amount` | `parse_ticks(["duck_amount"])` | ✅ Direct |
| 19 | `airborne` | `is_airborne` | Derivable from `flags` or `in_air` property | ✅ Direct |
| 20 | `flash_remaining` | `player_blind` events + decay | `parse_events("player_blind")` | ✅ Derivable |
| 21 | `health` | `health` | `parse_ticks(["health"])` | ✅ Direct |
| 22 | `armor` | `armor_value` | `parse_ticks(["armor_value"])` | ✅ Direct |
| 23 | `recoil_index` | `fl_recoil_idx` | `m_flRecoilIndex` via parse_ticks | ✅ Direct |
| 24 | `punch_pitch` | `aim_punch_angle[0]` | `m_aimPunchAngle` (X component) | ✅ Direct |
| 25 | `punch_yaw` | `aim_punch_angle[1]` | `m_aimPunchAngle` (Y component) | ✅ Direct |
| 26 | `shots_in_burst` | `shots_fired` | `parse_ticks(["shots_fired"])` or prop | ✅ Direct |
| 27 | `ammo` | `active_weapon_ammo` | `parse_ticks(["active_weapon_ammo"])` | ✅ Direct |
| 28 | `time_since_shot` | `time_since(ticks, shots)` | Derived from `weapon_fire` events | ✅ Derivable |
| 29 | `victim_shot` | Victim `weapon_fire` events | Same events, victim's fires | ✅ Derivable |
| 30 | `victim_footstep` | `player_footstep` events | `parse_events("player_footstep")` | ✅ Direct |
| 31 | `time_since_victim_noise` | `time_since(ticks, victim shots + steps)` | Derived from events | ✅ Derivable |
| 32 | `victim_crouch` | Victim `duck_amount` | Same as attacker | ✅ Direct |
| 33 | `victim_health` | Victim `health` | Same as attacker | ✅ Direct |
| 34 | `weapon_auto` | `isin(weapon, AUTO)` | `active_weapon_name` via parse_ticks | ✅ Direct |
| 35 | `weapon_sniper` | `isin(weapon, SNIPER)` | Same | ✅ Direct |
| 36 | `weapon_pistol` | `isin(weapon, PISTOL)` | Same | ✅ Direct |
| 37 | `weapon_smg` | `isin(weapon, SMG)` | Same | ✅ Direct |
| 38 | `weapon_shotgun` | `isin(weapon, SHOTGUN)` | Same | ✅ Direct |

### Summary

| Status | Count | Features |
| :--- | :---: | :--- |
| ✅ Directly available | 22 | pitch, yaw, velocities, health, armor, scoped, crouch, airborne, recoil, punch angles, shots_fired, ammo, weapon name, events |
| ✅ Derivable (exact same math as training) | 17 | All delta/rate/error/distance/speed features, flash_remaining, time_since_* |
| ⚠️ Approximate | 0 | — |
| ❌ Unavailable | 0 | — |

> [!TIP]
> **All 39 features are available with full parity.** The CS2CD dataset was itself built from parsed demo files (Parquet + JSON events). `demoparser2` exposes the same Source 2 entity properties. The mathematical derivations (angle wrapping, eye-height approximation, closing/lateral speed) are identical since we reuse the exact formulas from `cs2_data.py`.

### Known approximations (same as training pipeline)

These approximations exist in the **original training pipeline** and are faithfully reproduced, not introduced by this implementation:

1. **Eye height**: `Z + 64 - 18 * duck_amount` — approximation, not true hitbox position
2. **Flash remaining**: Computed from `player_blind` event duration decay — not true visual occlusion
3. **Victim footstep**: From `player_footstep` event — does not prove audibility through walls/smokes
4. **Line of sight / visibility**: Not available in either pipeline — not a feature

### Feature parity conclusion

> [!IMPORTANT]
> **Feature parity is sufficient for a meaningful experiment.** The demo-parsed telemetry provides the identical raw data that the CS2CD training pipeline used. The 39-feature computation uses the same math. The only domain gap is the *population* (your friends vs. CS2CD's player pool) and *recording conditions* (your private server vs. CS2CD's source matches), which is exactly what this experiment is designed to measure.

---

## Phase 2: Implementation Plan

### Architecture

```
Private CS2 Server (tv_enable 1)
    → GOTV .dem file
    → demoparser2: parse_ticks() + parse_events()
    → research/anticheat/dem_extractor.py  [NEW]
        → Tick DataFrame → per-player state arrays
        → player_hurt events → encounter anchors
        → 256-tick pre-impact windows
        → 39-feature vectors in documented order
        → NPZ export (x, player, tick)
    → research/scripts/score_private_match.py  [NEW]
        → Calls existing cs2_inference.predict()
        → Readable per-player + per-encounter report
```

### New files

---

#### [NEW] [dem_extractor.py](file:///c:/Users/Ionut/Desktop/desktop%20shit/Licenta/CS2-anticheat-ML/research/anticheat/dem_extractor.py)

**Purpose**: Extract 39-feature encounter windows from a CS2 `.dem` file, producing an NPZ compatible with `predict_cs2cd.py`.

**Key design decisions**:
- Reuse `FEATURES`, `wrap()`, `difference()`, `time_since()`, weapon sets from `cs2_data.py` — no code duplication
- Use `demoparser2.DemoParser.parse_ticks()` for tick data and `parse_events()` for game events
- Map `demoparser2` property names to the `BASE_COLUMNS` naming used by `cs2_data.py`
- Apply identical encounter detection logic: first damage in attacker-victim burst with 128-tick gap
- Apply identical rejection criteria: warmup, team damage, self damage, missing ticks, round boundary, not alive
- Log every approximation, missing tick, rejected encounter, and player with no encounters
- Return structured audit dict for the report

**Functions**:
- `extract_dem(dem_path, output_path=None, consent_players=None)` → dict with NPZ path, audit info
  - `consent_players`: optional set of Steam64 IDs — only process these players
  - Maps Steam64 IDs to match-local `Player_1`, `Player_2`, etc.
  - Returns full audit: encounter counts, rejections, feature stats, warnings

---

#### [NEW] [score_private_match.py](file:///c:/Users/Ionut/Desktop/desktop%20shit/Licenta/CS2-anticheat-ML/research/scripts/score_private_match.py)

**Purpose**: End-to-end CLI: `.dem` file → extraction → inference → readable report.

**Interface**:
```
python scripts/score_private_match.py match.dem \
    --model-dir experiments/cs2cd_v1 \
    --output reports/match_001/ \
    --player Player_3    # optional: score one player only
    --consent-ids 765...  # optional: only include these Steam IDs
```

**Outputs** (all in `--output` directory):
- `encounters.npz` — the extracted NPZ (compatible with `predict_cs2cd.py`)
- `extraction_audit.json` — full extraction log
- `scores.csv` — per-player scores (same format as existing `predict_cs2cd.py`)
- `report.txt` — human-readable report with:
  - Match metadata (map, duration, player count)
  - Per-player: encounters, raw score, calibrated score, all flags
  - Per-encounter scores (from event-level logits)
  - Extraction warnings (missing ticks, approximations, discarded encounters)
  - Safety disclaimer about benchmark vs. live performance

**Safety features**:
- Prints consent acknowledgment banner
- Prints disclaimer: "Frozen benchmark AUC ≈ 0.973 does not imply live-server accuracy"
- Never calls any kick/ban/punish API
- Steam IDs appear only in report metadata, never in model features

---

#### [NEW] [test_dem_extractor.py](file:///c:/Users/Ionut/Desktop/desktop%20shit/Licenta/CS2-anticheat-ML/research/tests/test_dem_extractor.py)

**Purpose**: Focused unit tests for the extraction pipeline.

**Test cases**:
1. **Feature ordering** — verify output column order matches `FEATURES` and `input_schema.json`
2. **Feature shapes and dtypes** — `[N, 256, 39]`, `float32`, finite
3. **Angle wrapping** — `wrap()` correctness on boundary values
4. **Chronological ordering** — tick array is sorted per player
5. **Damage event alignment** — window ends strictly before the first damage tick
6. **No future information** — modifying ticks ≥ anchor does not change extracted window
7. **Player filtering** — `consent_players` parameter correctly filters
8. **Missing telemetry** — gaps in tick data produce rejection, not silent zeros
9. **NPZ compatibility** — output loads correctly in existing `predict()` function
10. **Reproducibility** — deterministic output on fixture data

Since real `.dem` files are large and not committed, tests use **synthetic fixture data**: construct mock DataFrames matching `demoparser2` output format, and test the extraction logic without requiring a real parser.

The existing `example_validation_player.npz` is used as a regression test for the inference path — it must produce the same expected scores.

---

### Modified files

#### [MODIFY] [requirements.txt](file:///c:/Users/Ionut/Desktop/desktop%20shit/Licenta/CS2-anticheat-ML/research/requirements.txt)

Add `demoparser2` dependency.

#### [MODIFY] [PROJECT_JOURNAL.md](file:///c:/Users/Ionut/Desktop/desktop%20shit/Licenta/CS2-anticheat-ML/PROJECT_JOURNAL.md)

Add Entry 13 documenting the private-server pipeline implementation, capability audit results, and feature parity assessment.

---

## Phase 3: Testing

### Test matrix

| Test | What it verifies |
| :--- | :--- |
| `test_feature_order_matches_schema` | Column indices match `input_schema.json` exactly |
| `test_output_shape_and_dtype` | `[N, 256, 39]`, `float32` |
| `test_all_finite` | No NaN/Inf in output |
| `test_wrap_boundary_values` | `wrap(359)=-1`, `wrap(-359)=1`, `wrap(181)=-179` |
| `test_chronological_ticks` | Ticks are strictly increasing per encounter |
| `test_window_ends_before_damage` | Last tick in window < damage tick |
| `test_no_future_information` | Changing post-anchor data doesn't affect window |
| `test_player_filter` | `consent_players` filters correctly |
| `test_missing_ticks_rejected` | Gaps produce rejection log entry, not silent fill |
| `test_warmup_excluded` | Warmup encounters are rejected |
| `test_team_damage_excluded` | Same-team encounters rejected |
| `test_round_boundary_excluded` | Round crossings rejected |
| `test_npz_loads_in_predictor` | Output NPZ accepted by `cs2_inference.predict()` |
| `test_example_npz_regression` | Existing example produces unchanged expected scores |
| `test_deterministic_output` | Same input → same output |

### Commands
```bash
cd research
python -m pytest tests/test_dem_extractor.py -v
python -m pytest tests/ -v   # full suite including existing tests
```

---

## Phase 4: Safety and Research Methodology

### Built-in safeguards

| Requirement | Implementation |
| :--- | :--- |
| Authorized private server only | Documentation requires `tv_enable 1` on private server; no public matchmaking integration |
| Explicit consent | `--consent-ids` parameter; consent banner printed at start |
| Shadow/review mode only | Script produces reports; no server communication or player actions |
| No automatic ban/kick/punish | No game API calls; output is CSV + text report |
| No unnecessary personal data | Steam IDs are match-local aliases (`Player_1`); original IDs only in audit metadata |
| Benchmark disclaimer | Every report includes: "Frozen CS2CD benchmark AUC ≈ 0.973 does not predict live-server performance" |
| False positive tracking | Per-encounter scores + calibrated probabilities enable post-hoc FP/FN analysis |
| Independent legitimate testing | Run known-clean friends first; compare score distributions before testing suspicious play |

### Research methodology notes

The report will document:
- This is the first out-of-distribution test of the CS2CD model
- Training data came from a different player population
- Private-server conditions (map, skill level, player count) differ from training
- Results should be interpreted as exploratory, not as validated deployment metrics
- The system should be tested first on known-legitimate gameplay to establish a baseline

---

## Open Questions

> [!IMPORTANT]
> **Q1: `demoparser2` property name mapping.** The exact property names for `aim_punch_angle` components, `is_airborne`, and `shots_fired` in `demoparser2` may differ from the CS2CD Parquet column names. The implementation will probe available properties and log any that require alternative names (e.g., `m_aimPunchAngle` vs `aim_punch_angle`). This is a runtime discovery, not a design blocker.

> [!IMPORTANT]
> **Q2: GOTV demo tick rate.** CS2 GOTV demos from private servers typically record at 64 ticks/second (matching the server tick rate). If the demo uses a different rate, the 256-tick = 4-second assumption breaks. The extractor will verify the tick rate from the demo header and abort with a clear message if it doesn't match. If your server runs at 128 tick, we can add a tick-decimation step later.

> [!IMPORTANT]
> **Q3: `demoparser2` installation.** Requires `pip install demoparser2`. This is a Rust-based package with pre-built wheels for common platforms. If it fails on your system, we fall back to `awpy` which wraps it.

---

## Verification Plan

### Automated Tests
```bash
cd research
python -m pytest tests/test_dem_extractor.py -v
python -m pytest tests/ -v
python scripts/predict_cs2cd.py examples/example_validation_player.npz --output outputs/regression_check.csv
```

### Manual Verification
1. Record a short private-server match (1-2 rounds) with `tv_enable 1`
2. Run extraction: `python scripts/score_private_match.py path/to/match.dem --output reports/test_match/`
3. Verify extraction audit: check rejection reasons, encounter counts
4. Verify feature distributions against training data statistics in `normalization.json`
5. Compare scores against expectation (known-clean players should score low)
6. Run on the example NPZ to confirm regression: scores must match `example_expected_scores.csv`
