# OpenCS2 AWP addition: implementation plan (v1)

**Status:** step 1–3 implemented by `research/scripts/add_opencs2_awp.py`; steps 4–7 need
separate approval (they change the trained model and its evaluation).

## Goal

Reduce false positives on legitimate sniper-heavy players by adding **only as many
legitimate sniper-heavy player-maps as the training split is missing**, taken from
professional CS2 matches.

## Why (evidence, Entry 24)

| Training split (`cs2cd_v1`) | Legitimate | Cheater-labelled |
|---|---|---|
| Sniper encounters | 11.2% | 56.5% |
| Sniper-heavy player-matches (>50% of encounters with a sniper) | 207 | 418 |

On the (already used) test split, all 4 false positives at the strict `fpr_1pct`
threshold were legitimate sniper-heavy players (4/42 vs 0/874 for everyone else).

## Target

**+211 legitimate sniper-heavy player-maps** (418 − 207), so that legitimate and
cheater-labelled sniper-heavy player-matches are equal in training. Not more: adding all of
OpenCS2 (~7,900 player-maps) would cut the positive share from 16.7% to about 6% and turn the
new source into a shortcut.

Expected size (from one probe map: 11 encounters, 9 with a sniper): about 2,300 encounters,
about 1,900 of them sniper shots (*estimate*; the script reports the real numbers).

## Source

- OpenCS2 HLTV demos, `blanchon/opencs2_dataset_demo` on Hugging Face, revision
  `912a363098feec3f220e41d68634d2f71676c28b`, licence **CC BY 4.0** (attribution required).
- 1,988 `.dem` files (one per map), 400–700 MB each, with metadata tables listing every kill
  and its weapon.
- The metadata's `tick_rate` says 128, but the demo file itself is **64.00 Hz**
  (148,093 ticks / 2,313.95 s on the probe), so the demos fit the 256-tick windows. Every demo
  is still checked by `demo_timing.py`, which rejects anything that is not 64 Hz.

## Labels

All selected players are **legitimate (label 0)**: professional LAN matches. This is a strong
but not certain negative label; it is recorded as `source = opencs2_pro`.

## Selection rule (fixed before any download; never uses model scores)

1. From the metadata kills, a *candidate* is a player with ≥ 8 kills on that map and ≥ 50%
   of them with a sniper weapon (`cs2_data.SNIPER`).
2. Demos with candidates are ordered deterministically: seeded shuffle
   (`seed = 20261019`), then demos with two candidates first.
3. At most **3 maps per player**, so a few famous AWPers do not dominate.
4. Each demo is extracted with the project's demo extractor (`dem_extractor.extract_dem`),
   scoring only the candidates as attackers. A player-map is **accepted** if it has ≥ 5
   eligible encounters and more than 50% of them are with a sniper weapon.
5. Stop at 211 accepted player-maps.

## Split and grouping

- All accepted player-maps go to a new **train** addition. The frozen `cs2cd_v1` splits,
  thresholds and results are not touched.
- `group = opencs2/<hltv_match_id>`: all maps of one match stay together.
- The selected matches and players are recorded so that any future OpenCS2 *evaluation*
  cohort can exclude them.

## Output (`research/data/opencs2/awp_v1/`, not in git)

- `matches/<match>_m<map>.npz`: windows in the project format (`x`, `player`, `tick`).
- `events.parquet`: one row per accepted encounter, **same columns as**
  `experiments/cs2cd_v1/events.parquet` (`label = 0`, `split = train`,
  `source = opencs2_pro`).
- `manifest.json`: revision, licence, selection parameters, per-demo size, SHA-256, timing,
  approximations, accepted players, counts, and duplicate check against CS2CD windows.
- `reserve_events.parquet`: every **other** player on the downloaded maps (mostly rifles),
  `split = reserve`. Extracting them costs almost nothing, because parsing the demo is most of
  the work. **Not independent** of the training addition (same matches and teams): usable
  later as extra training data, but never as an independent evaluation cohort for a model
  trained on `events.parquet`.

Raw demos are kept in `D:\opencs2\demos\` (pinned revision; can be re-downloaded).

## Checks performed by the script

- 64 Hz timing per demo (extractor).
- Approximated channels per demo (velocity is always derived from position in demos).
- Duplicate windows against CS2CD (SHA-256 of each window).
- Final counts: accepted player-maps, distinct players, encounters, sniper encounters.

## Not in this step (need approval)

4. **New dataset version** (`cs2cd_v2`): CS2CD + this addition, a new protocol file, and an
   update to `DATA_SOURCES.md` with the attribution.
5. **Channel handling:** compare feature distributions per source; decide whether flash,
   footstep and velocity are treated identically across sources (masking or derived velocity
   everywhere) so the new source cannot become a shortcut.
6. **Retrain and recalibrate** on the new version (calibration thresholds change; old
   thresholds no longer apply).
7. **Evaluate on new data only:** CS2CD cheater recall before vs after; false-positive rate
   by weapon and source; the source-only diagnostic; a separate OpenCS2 evaluation cohort
   (different matches and players); and the friends' AWP sessions as a pre-registered
   evaluation group.

## Risks

- **Source shortcut:** a third source that is entirely legitimate. Checked in step 5 and 7.
- **Skill shift:** pro aim may lower recall on subtle cheaters. Checked in step 7.
- **Velocity is derived** from position in demos, while the CS2CD velocity source is
  unknown (Phase 0, C2).
- **Few distinct people:** the per-player cap limits this, but pro AWPers are a small group.
