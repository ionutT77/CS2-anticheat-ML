# Project state: CS2 anti-cheat ML thesis (as of 2026-11-16)

Read-only survey of the repository at commit `fb5f39b` (2026-10-28, branch `main`, clean working
tree). Nothing was trained, tested, downloaded or modified to produce it.

**How to read the evidence tags**

- **[read]**: taken directly from the cited file, JSON key or commit.
- **[checked]**: I recomputed or re-read it from a saved table or artifact (a count, a hash, a file listing).
- **[inferred]**: my conclusion from the evidence; not stated in the repo.
- **[unverified]**: stated in the repo, but I could not confirm it (section 8 lists these).

Paths are relative to the repository root. `J:E27 (L1260–1333)` means `PROJECT_JOURNAL.md`,
Entry 27, lines 1260–1333.

---

## 1. What the thesis is and claims

| Item | Statement | Source |
|---|---|---|
| Title | "Machine Learning-Based Server-Side Anti-Cheat for Counter-Strike 2" | [read] J: overview table (L9–15) |
| Task | Rank a player in one match by how likely their aim and shooting behaviour is cheating. Output is a calibrated score plus review flags, never an automatic action | [read] `research/ARCHITECTURE_CHOICES.md` §1 |
| Decision unit | One player in one match ("player-match") with at least one eligible encounter | [read] `research/experiments/cs2cd_v1/protocol.json` → `decision_unit` |
| Input | Encounter windows: 256 consecutive ticks at 64 Hz (4 s) ending strictly before the first bullet damage of an attacker-victim burst; a gap over 128 ticks starts a new burst; 39 channels per tick | [read] `research/METHODS.md` L47–61; channel list in `protocol.json` → `features` |
| Causality rule | The victim is identified retrospectively from the hit; the damage tick and everything after it are excluded | [read] `METHODS.md` L51; `.claude/CLAUDE.md` invariants |
| Labels | Publisher player-match labels. No cheat-activation intervals, no cheat-type labels | [read] `protocol.json` → `labels`; `METHODS.md` L79 |
| Model | Blend: 0.75 temporal CNN (`tcn39_seed123`, 160,641 parameters) + 0.25 LightGBM (`lgbm39_leaves15`), Platt-calibrated | [read] `cs2cd_v1/selection.json` → `components`; `tcn39_seed123_done.json` → `parameters` |
| Real-time part | CounterStrikeSharp plugin on a self-hosted server logs per-tick state to NDJSON; a Python listener builds windows and scores each hit with the frozen model. Shadow/review mode only | [read] J:E15–E23 (L816–1130); `research/realtime/` |
| Explicit non-claims | No validated live anti-cheat performance; no claim for new players, future game versions, unseen cheat families; AUC 0.973 is not 97.3% accuracy | [read] `protocol.json` → `claims`; `METHODS.md` L81–83; `README.md` L9 |

**Scope drift to know about.** The journal's stated goal is "detect aimbot and bhop cheats using
LSTM/Transformer models" (J L14). What is implemented is an aim-behaviour detector whose selected
model is a CNN + tree blend; bhop detection was never started (J L1405–1406 lists it as an open
challenge). [read + inferred]

**Authorship, from git history.** The original prototype (`src/`, `notebooks/`, the Kaggle LSTM) is
the thesis author's. The whole `research/` benchmark (CS2CD extraction, ten candidates, selection,
calibration, test, the Kaggle audit, `docs/` report) arrived in one commit, `b62c571` (2026-09-07,
author AndreiBulzan, merged as PR #2 "research/validated-baselines"). `METHODS.md` L5 speaks of
"Ionuț's journal, our Kaggle experiments". Everything after `f2de563` (demo extractor, real-time
pipeline, OpenCS2 data work, weapon analyses) is committed under the thesis author's account.
[checked: `git log`] Several journal entries are written in the voice of an AI coding assistant
working under the author's approval (J:E15 L862 "the implementation prompt for Opus 5.5"; J:E21
L1055 "I deleted the older capture files … without asking"; reviews by an `anticheat-reviewer`
agent, J:E17 L899). Who did what matters for a thesis; the repo does not discuss it. [inferred]

---

## 2. Datasets

### 2.1 Kaggle CS:GO cheating dataset (prototype and LSTM repair only)

- 12,000 records (10,000 negative, 2,000 positive), each `[30, 192, 5]`: 30 engagements, 192 ticks at 32 Hz (5 s before a kill, 1 s after), 5 channels. Licence listed as Unknown. No stable player or match IDs. [read] `research/DATA_SOURCES.md` L15–23; `METHODS.md` L29
- Split after the overlap audit: 7,616 train / 1,189 validation / 1,127 calibration / 2,038 test records; 30 records with contradictory labels excluded; split seed 20260906, ten stratified group folds. [read] `METHODS.md` L31; `research/reports/protocol.md` L9
- Known defect of the original split: 699 of 1,800 test records are connected to training through overlapping trajectories, 25 exact duplicates. [read] `METHODS.md` L25
- Local: `data/raw/cheaters.npy` and `legit.npy` are on disk, git-ignored. [checked]

### 2.2 CS2CD (the main benchmark)

- Hugging Face `CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection`, CC BY 4.0, pinned revision `44e5129…`; 795 matches, 52.6 GB raw. [read] `DATA_SOURCES.md` L5–11; `METHODS.md` L47
- Extraction: 734,975,247 player-tick rows → 141,633 encounters; 7,949 player-matches, 7,508 with at least one eligible encounter (94.5%; 98.9% of positives, 93.6% of negatives). [read] `cs2cd_v1/data_audit.json`; `METHODS.md` L81
- **Label origin** [checked from `cs2cd_v1/players.csv`]:

  | Source folder | Label 0 | Label 1 |
  |---|---|---|
  | `no_cheater_present` | 4,779 | 0 |
  | `with_cheater_present` | 1,861 | 1,309 |

  Every positive label comes from one source folder. "Reviewed" in the code means
  `source == 'with_cheater_present'` (`research/scripts/select_cs2cd.py` L19). The negative-only
  collection is unreviewed; the reviewed one "is not guaranteed error-free" (`METHODS.md` L79).
  The journal adds that only matches with a VAC-banned player were manually verified (J:E24 L1160).
- **Identity**: player IDs are remapped per match, so separation of real people across splits cannot be verified. [read] `protocol.json` → `identity_limit`
- **Split**: 60/15/10/15% by match-connected group, stratified on presence of a positive label; `train_test_split` seeds 9072026 / 9072027 / 9072028 (`research/scripts/audit_cs2cd.py` L65–67). Duplicate grouping by full hurt-event signatures and 32-event subsequences. [read]

  | Split | Matches | Eligible player-matches | Encounters (label 0 / 1) |
  |---|---|---|---|
  | train | 477 | 4,523 | 71,981 / 14,041 |
  | validation | 118 (119 assigned, one has no eligible encounter) | 1,112 | 16,749 / 3,330 |
  | calibration | 79 | 743 | 12,099 / 2,261 |
  | test | 120 | 1,130 (214 positive) | 17,975 / 3,197 |

  [read `data_audit.json` → `split_counts`; encounter label counts checked from `events.parquet`]
- **Weapon balance, encounters, cheating / not cheating** [checked from `events.parquet`; matches `research/opencs2/SCOUT_AWP_BALANCE.md` §2 and `OTHER_WEAPONS.md` §2]:

  | Weapon | Train | Validation | Calibration | Test |
  |---|---|---|---|---|
  | Scout (SSG 08) | 5,933 / 2,925 | 1,525 / 837 | 969 / 643 | 1,374 / 760 |
  | AWP | 1,717 / 4,776 | 366 / 1,077 | 325 / 898 | 371 / 1,242 |

  Overall cheating share of training encounters is 16.3%; the scout is 67%, auto-sniper 44%,
  Deagle/R8 28%, AWP 26%, USP-S/P2000 11%, rifles 6%. Full table: `OTHER_WEAPONS.md` §2. [read]
- **Local**: only metadata (`research/data/cs2cd/metadata/`), the event index (`events.parquet`), player tables, saved predictions and models. **The raw data and the window cache are not on this machine.** [checked: directory listing; also stated in J:E16 L878 and `OTHER_WEAPONS.md` L26]

### 2.3 OpenCS2 professional demos (legitimate-only, staged, not trained on)

- Hugging Face `blanchon/opencs2_dataset_demo`, revision `912a363…`, CC BY 4.0; 1,988 HLTV demos. All players labelled 0 on the assumption that professional LAN play is legitimate ("strong but not certain"). [read] `research/opencs2/PLAN.md` L38–51
- Metadata says 128 tick; the demo files are 64.00 Hz. [read] `PLAN.md` L44–46
- Two staged additions, both git-ignored under `research/data/opencs2/` [checked against each `manifest.json` → `totals` and the parquet files]:

  | Set | Demos processed / failed | "Training addition" (`events.parquet`, split `train`) | Reserve (`reserve_events.parquet`, split `reserve`) |
  |---|---|---|---|
  | `awp_v1` | 127 / 10 | 100 player-maps, 78 players, 2,356 encounters (1,330 AWP, 129 scout) | 25,632 encounters (1,480 AWP, 172 scout), 1,070 player-maps |
  | `scout_v1` | 238 / 29 | 183 player-maps, 4,320 encounters (781 scout, 1,168 AWP) | 41,737 encounters (362 scout, 2,398 AWP) |

  Zero duplicate windows against CS2CD in both. Selection for `awp_v1` is seeded (20261019) and
  outcome-blind; `scout_v1` is ordered by metadata scout kills.
- The reserve shares matches with the addition, so it can never be an independent evaluation cohort for a model trained on the addition. [read] `PLAN.md` L81–85
- Raw demos are on `D:\opencs2\demos\` (outside the repo; about 65 GB + 106 GB by the manifests). [read; folder existence checked, contents not]
- **Not in `DATA_SOURCES.md`.** The attribution update is plan step 4, unapproved. [checked]

### 2.4 Self-collected data

- **Live server captures**: one author-vs-bots human session (8 rounds, 51 windows) and one author-vs-hard-bots live session (18 encounters), both 2026-10-11. [read] J:E21–E23. Bot victims are out of distribution (training never had bots).
- **Friend sessions: none recorded.** The protocol exists (`research/realtime/server/FRIEND_SESSIONS.md`); J:E27 L1326 says "friend sessions (0, not started)"; the plugin folder on `D:\cs2server` holds only captures dated 2026-10-11 and `consent.txt` is 19 bytes (one ID). [read + checked]
- **Demos scored offline** (`local_runs/`, git-ignored) [checked from each run's `extraction_audit.json` header and `scores.csv`]:
  - two maps of one pro match on a StarLadder server (documented, J:E14);
  - two maps of a second pro match on a BLAST server (`local_runs/ancient`, `cache`, 2026-09-22; demos in a second match folder at the repo root); no journal entry;
  - five Valve matchmaking demos, the author's and a friend's (09-24 to 09-29); only referenced indirectly as "MM demos (2 demos, 106 windows)" in J:E20 L1006;
  - two demos recorded on FACEIT servers, scored as `cheater_demo_1` / `cheater_demo_2` (09-22).
  - `cheaters_demos/` holds all 7 of the matchmaking and FACEIT `.dem` files (1.2 GB), despite its name. See section 6, gap G1.

### 2.5 Sources considered and rejected

ESTA (2 Hz) and Kaggle (5 channels, 32 Hz) are unusable for the 39-channel model (J:E27 L1304). No
public dataset with cheat-type labels was found (J:E24 L1160). [read]

---

## 3. Pipeline and models as implemented

### 3.1 Code map

| Stage | Where | Note |
|---|---|---|
| CS2CD extraction and features | `research/anticheat/cs2_data.py`, `cs2_features.py`; `scripts/prepare_cs2cd.py`, `cache_cs2cd.py` | Hash-pinned, frozen |
| Split and audit | `scripts/audit_cs2cd.py` | Refuses to overwrite an existing protocol |
| Training | `scripts/train_cs2cd_neural.py` (7 configs, needs CUDA), `train_cs2cd_tabular.py` (3 LightGBM) | |
| Selection, calibration, test | `scripts/select_cs2cd.py`, `evaluate_cs2cd.py` (`--test`), `analyze_cs2cd.py` | |
| Inference | `anticheat/cs2_inference.py`, `scripts/predict_cs2cd.py` | NPZ `x [N,256,39]`, `player`, `tick` |
| `.dem` extraction | `anticheat/dem_extractor.py`, `demo_timing.py`, `scripts/score_private_match.py`, `anticheat/private_scoring.py` | 64 Hz only |
| Real-time | `research/realtime/plugin/AnticheatTelemetry.cs`, `capture_schema.py`, `check_capture.py`, `demo_parity.py`, `capture_windows.py`, `live_scoring.py` | |
| OpenCS2 additions | `scripts/add_opencs2_awp.py`, `add_opencs2_scout.py` | |
| Weapon analyses | `scripts/count_scoped_encounters.py`, `analyze_weapon_groups.py` | Saved scores only |

### 3.2 Training configuration (CS2CD)

- Neural: AdamW, lr 5e-4, weight decay 1e-3, up to 40 epochs, patience 8, seeds 42 and 123; unweighted player-match BCE on bags of up to 16 sampled encounters, optional encounter auxiliary loss 0.25; evaluation uses all eligible encounters. [read] `protocol.json` → `neural_training`
- LightGBM: `n_estimators=2000`, `learning_rate=0.025`, `random_state=42`, early stopping 100. [read] `scripts/train_cs2cd_tabular.py` L33–38
- **Natural sampling, no class, sample or weapon weights anywhere.** [read] J:E27 L1290; `ARCHITECTURE_CHOICES.md` L46–48
- Normalisation: fixed physical scales + asinh compression, then z-score fitted on training ticks only. [read] `ARCHITECTURE_CHOICES.md` L27–28; `METHODS.md` L67
- Selection rule (written before training): mean of validation AUC on all players and on reviewed matches. Reason recorded in `protocol.json` → `pretraining_amendment`: every positive comes from reviewed matches, so collection source is a shortcut.
- Artifact environment: Python 3.13.5, torch 2.9.0, scikit-learn 1.6.1, lightgbm 4.6.0 (`cs2cd_v1/environment.json`). [read]

### 3.3 Validation selection (`cs2cd_v1/selection.json` → `validation_models`, `winner`) [read]

| Model | Selection score |
|---|---|
| LSTM 5 ch / 39 ch | 0.908 / 0.929 |
| Patch Transformer | 0.931 |
| Hierarchical CNN + Transformer | 0.932 |
| LightGBM leaves 15 / 31 | 0.938 / 0.938 |
| TCN seed 42 / seed 123 / reviewed-only | 0.940 / 0.944 / 0.942 |
| **Blend 0.75 TCN + 0.25 LightGBM** | **0.946** (all 0.9425, reviewed 0.9496) |

### 3.4 Calibration and thresholds (`cs2cd_v1/calibration.json` → `models.selected`) [read]

Platt slope 0.7687, intercept 0.2383, fitted on the calibration split (743 player-matches: 625
negative, 118 positive).

| Policy | Threshold | Calibration-split result |
|---|---|---|
| `accuracy` | 0.2572 | precision 0.695, recall 0.831, FPR 6.88% |
| `f1` | 0.2114 | precision 0.680, recall 0.864, FPR 7.68% |
| `default_0_5` | 0.5 | precision 0.673, recall 0.593, FPR 5.44% |
| `fpr_1pct` (strict) | 0.8528 | 20 TP, 6 FP of 625 (FPR 0.96%), recall 0.169 |
| `fpr_0_1pct` | 0.8895 | 12 TP, 0 FP of 625, recall 0.102 |

### 3.5 Headline test result (`cs2cd_v1/test_results.json` → `cohorts.all.models.selected`) [read]

Test: 1,130 player-matches, 214 positive, 916 negative, 120 matches.

- ROC-AUC **0.9727**, 95% interval 0.9560–0.9854 (bootstrap by match, 2,000 resamples, 120 clusters); average precision 0.8829.
- Advantage over the best tree: 0.0049, paired interval −0.0015 to 0.0124, so not established. [read] `METHODS.md` L73; `test_results.json` → `cohorts.all.bootstrap.selected_minus_other_95`

| Policy | TP | FP | FN | TN | Precision | Recall | FPR | Accuracy |
|---|---|---|---|---|---|---|---|---|
| `accuracy` | 184 | 49 | 30 | 867 | 0.790 | 0.860 | 5.35% | 93.0% |
| `f1` | 193 | 58 | 21 | 858 | 0.769 | 0.902 | 6.33% | 93.0% |
| `default_0_5` | 154 | 28 | 60 | 888 | 0.846 | 0.720 | 3.06% | 92.2% |
| `fpr_1pct` (strict) | 67 | 4 | 147 | 912 | 0.944 | 0.313 | 0.44% | 86.6% |
| `fpr_0_1pct` | 45 | 3 | 169 | 913 | 0.938 | 0.210 | 0.33% | 84.8% |

- Reviewed-only cohort (419 player-matches, 214 positive): AUC 0.9710; strict policy 67 TP, 0 FP of 205.
- Source-only diagnostic: AUC 0.888 overall, 0.500 within reviewed matches. [read] `METHODS.md` L71
- Precision at an assumed 1% prevalence, strict policy: 0.42. [read] `test_results.json`; `METHODS.md` L83
- Observation horizon, first N encounters, same calibrator and thresholds: AUC 0.834 (N=1), 0.946 (N=5), 0.962 (N=10). [read] `test_results.json` → `horizons`
- Always-negative baseline accuracy: 81.1%. [read] `README.md` L25

### 3.6 Kaggle results (secondary)

- Journal prototype, engagement level: AUC 0.713–0.718, precision 0.22–0.25, recall 0.72–0.82 at threshold 0.5 (J:E10 L361–398). No checkpoint or prediction arrays published for those runs (`METHODS.md` L19). [read]
- Audited split, record level: retrained original recipe AUC 0.8351, revised LSTM 0.8432 (88.5% accuracy), CNN pair 0.8586 (89.3%), constant baseline 84.5%. Gain of the revision: 0.0081, interval −0.0044 to 0.0208. [read] `METHODS.md` L35–43; `research/experiments/lstm_revision/test_results.json`
- That test set was already examined; the docs call it "a controlled diagnostic that needs fresh confirmation" (`METHODS.md` L43). [read]

### 3.7 Real-time pipeline status

| Phase | Result | Source |
|---|---|---|
| 0 parity gate | GO with conditions C1–C3 | J:E16; `research/realtime/PHASE0_PARITY.md` |
| 1 capture | Plugin vs GOTV demo of the same session: yaw, pitch, x, y, health, punch agree on 35,771 ticks. Plugin event stamps are one tick early in about 95% of events, so windows are anchored on the victim's health drop | J:E21 (L1027–1058) |
| 2 window builder | Causality and determinism tests; 88 hits → 51 windows | J:E22 |
| 3 live scoring | Median 0.40 s per hit (first hit 4–7 s, model load); flags shown as "provisional" below 10 encounters; thresholds unchanged | J:E23; `research/realtime/live_scoring.py` L151–158, default `min_encounters=10` |
| 4 evaluation and consent | **Not started** | J:E23 L1130 |

Capacity: up to 10 humans supported; over 20 not supported. Known listener defects: hits on
non-consenting victims dropped, backlog over about 64 s silently loses hits, rejected windows not
logged, player score never resets between maps, a reused slot can be credited to the wrong player.
[read] J:E24 L1138–1148; `research/realtime/server/CAPACITY.md`

---

## 4. Known data issues and risks

### 4.1 Label noise and label granularity

- Labels are per player-match; a "cheating encounter" is any encounter by a labelled player, not a cheating shot. [read] `SCOUT_AWP_BALANCE.md` L12–14
- Negatives in `no_cheater_present` are unreviewed; a review queue of strict-policy negatives exists (`cs2cd_v1/strict_policy_negative_label_review_queue.csv`), labels left unchanged. [read] `METHODS.md` L73, L79
- OpenCS2 "legitimate" is an assumption about pro play. [read] `PLAN.md` L50–51
- A later-ban labelling route (`GetPlayerBans`) would give noisy negatives ("not banned" is not "legitimate"). [read] J:E27 L1296–1301

### 4.2 Source shortcut

All positives come from one source folder (section 2.2). Source-only AUC is 0.888 overall. The
selection rule and the reviewed-only cohort were added to control for it, and the docs still say
"collection artifacts may still remain" (`METHODS.md` L71). [read + checked]

### 4.3 Weapon imbalance (scout), with three different definitions in use

The numbers below are not interchangeable. Keep the unit and definition with each one.

| Definition | Numbers | Source |
|---|---|---|
| Encounters, train | Scout 5,933 cheating vs 2,925 not; AWP 1,717 vs 4,776 | [checked] |
| "Sniper-heavy" player-match (over 50% of encounters with any sniper), train | 418 cheating vs 207 legitimate; by main sniper: scout 364 vs 122, AWP 54 vs 85 | [read] J:E24 L1154, J:E27 L1264 |
| "Main weapon group" player-match (group with most encounters), train | Scout 397 vs 170; AWP 78 vs 225 | [read] `OTHER_WEAPONS.md` L49–50 |

False positives and recall on the **already-used test split** (descriptive, nothing tuned on it):

- Strict threshold, legitimate players: sniper-heavy 4/42 vs 0/874 others (J:E24 L1153); by main sniper, scout-heavy 4/25, AWP-heavy 0/17 (J:E27 L1277); by main weapon group, scout 4/36 (`OTHER_WEAPONS.md` §3).
- `accuracy` threshold, legitimate players: scout-heavy 18/25, AWP-heavy 12/17, others 19/874 (J:E27 L1277).
- Strict recall on cheater-labelled players: scout-heavy 0.55 (n=102), AWP-heavy 0.14 (n=22), others 0.09 (n=90); about 56 of the 67 strict detections are scout-heavy (J:E27 L1278).
- **Stated consequence**: "Part of the headline result may come from a scout-style shortcut in CS2CD's labels. Reducing that shortcut would likely lower strict recall" (J:E27 L1279).
- Weapon-only baseline (logistic regression on weapon-group shares): test AUC 0.874 vs detector 0.973 (`OTHER_WEAPONS.md` §4.4).
- Mechanism is unproven: `weapon_sniper` carries 0.9% of tree gain, `shots_in_burst` 38% (J:E24 L1155). The first-shot hypothesis for pistols is untested because the window cache is not local (`OTHER_WEAPONS.md` L143–146).

### 4.4 Domain shift between sources

| Source | Velocity | Flash / footstep channels | Skill and setting |
|---|---|---|---|
| CS2CD (training) | present; whether networked or derived is unknown (`PLAN.md` L114–115) | present | matchmaking-type players (J:E14 L727 says "primarily pub/FPL", not otherwise sourced) |
| GOTV / HLTV demos (pro, matchmaking) | derived from position differences, clamped ±5000 u/s (`anticheat/dem_extractor.py` L94–100, L401–410) | footstep and blind events absent in the two pro GOTV demos, both channels all zero (J:E14 L745–756); flash present in the OpenCS2 probe (J:E25 L1178) | professional |
| Live plugin | logged `AbsVelocity`, agrees with position differences (J:E19) | events available; footstep and blind timing unvalidated (J:E22 L1092) | friends / bots on a private server |

Measured effect on legitimate pro players, frozen model and thresholds (J:E27 L1266–1273,
`local_runs/opencs2_pro_fpr_scores.csv`):

| Group | n | `accuracy` | `f1` | `fpr_1pct` |
|---|---|---|---|---|
| Pro AWPers (`awp_v1` addition) | 100 | 13% [7.1–21.2] | 21% [13.5–30.3] | 0% [0–3.6] |
| Pro riflers, same matches (reserve) | 1,069 | 0.2% | 0.3% | 0% [0–0.3] |

One pro AWPer was flagged at `accuracy` and `f1` on a real match (calibrated 0.42), with AWP
encounter scores up to 0.97 and AK-47 scores near 0 (J:E14 L690–735).

### 4.5 Same-person leakage and people as the unit

- CS2CD cannot guarantee person-disjoint splits (section 2.2).
- OpenCS2: at most 3 maps per player in `awp_v1`; 100 player-maps come from 78 people; "pro AWPers are a small group" (`PLAN.md` L116). `scout_v1` has no per-player cap that I found in its arguments (`scripts/add_opencs2_scout.py` L76–86). [read + inferred]
- Friend data: "Many distinct people matter more than many matches: a few friends repeated many times teach the model their style" (J:E26 L1251). The protocol treats people as the unit: at least 12 people, first 6 player-matches per person per mode, person-level bootstrap (`FRIEND_SESSIONS.md` L42–54).

### 4.6 Small samples and calibration transfer

- Calibration split has 625 negatives, so a 0.1% FPR target cannot be resolved: `fpr_0_1pct` gave 0/625 on calibration and 3/916 on test. [read] `calibration.json`, `test_results.json`; the same caveat is written for Kaggle in `reports/protocol.md` L25
- Strict recall was 16.9% on calibration and 31.3% on test: operating points move between splits. [read]
- Strict test FPR rests on 4 false positives; all 4 are scout-heavy. [read]
- **All 4 are in the unreviewed `no_cheater_present` source, and 3 of the 4 are players of one match** (`no_cheater_present/84`; the fourth is in match 64). [checked] `cs2cd_v1/strict_policy_negative_label_review_queue.csv`; consistent with 0 FP of 205 in the reviewed cohort. So these four cannot be told apart from unlabelled cheaters, and the strict FPR is effectively two matches' worth of evidence. No document mentions the same-match clustering. [inferred]
- Per-weapon thresholds were rejected for sample size: 14 AWP-heavy and 30 scout-heavy legitimate player-matches in calibration (J:E27 L1288).
- Tiny groups: 49 legitimate USP-S-main players in validation + test, 7 auto-sniper; 21 cheating scoped-rifle encounters in training (`OTHER_WEAPONS.md` §3; J:E26 L1243).
- Early scores are unstable: live player score 0.73 after 1 encounter, 0.12 after 18 (J:E23 L1123, J:E29 L1369).

### 4.7 Test-set reuse

The CS2CD test set has been used for the headline result and then for several post-hoc weapon
breakdowns (J:E24, E27, E29), each stated as descriptive. Validation is not independent either: it
chose the checkpoint, early stopping and blend weights (`OTHER_WEAPONS.md` L23–25). Any new claim
needs a new cohort (`REPRODUCING.md` L64). [read]

### 4.8 Coverage and detection limits

- No score for players without a damage encounter (5.5% of player-matches). [read] `METHODS.md` L81
- No line of sight, smoke or hitbox data; eye height is approximated. [read] `METHODS.md` L51
- The author's list of unsolved problems: sniper flick false positives, silent aim / low-FOV cheats, bhop, wallhacks (J L1395–1414). The silent-aim statement has no measurement behind it in the repo. [read + inferred]

### 4.9 What the tests guard (names read, not run)

- `tests/test_cs2_pipeline.py`: no post-impact information in windows, causal time-since features, padding invariance.
- `tests/test_pipeline.py`: no duplicate groups across Kaggle splits, pre-features cannot see post-engagement, non-finite input rejected, saved weights restore.
- `tests/test_thresholds.py`: low-FPR threshold with ties and small samples; calibration preserves ranking.
- `tests/test_capture_windows.py`, `test_capture_checks.py`, `test_live_scoring.py`: anchoring on the health drop, nothing at or after the hit changes a window (with a negative control), determinism, bots never scored, live windows equal batch windows, unreadable punch is reported and never zero-filled.
- `tests/test_dem_extractor.py`, `test_demo_parser.py`, `test_demo_timing.py`, `test_private_scoring.py`: extractor rejection rules, 64 Hz timing, feature parity with the training extractor, checksum checks. (Described from J:E13 L520; I did not read these four files' test names.)
- `tests/test_add_opencs2.py`, `test_add_opencs2_scout.py`: candidate rule, deterministic ordering, yield stop rule.
- **Not guarded**: person-level separation, source or weapon shortcut, anything about merged datasets or weights (no such code exists). [inferred]

---

## 5. Open decisions and unfinished threads

The four threads named in recent commit messages, with where each is written down.

### D1. Friend data: evaluation, training, or split by person (commit `fb5f39b`)

- **Decided on paper**: `FRIEND_SESSIONS.md` (draft v1) fixes friend sessions as an evaluation group only: within-person design (normal, scout-only, AWP-only), at least 12 people, blind collection, offline scoring, one third of the people held out for a single final use.
- **Reopened**: `SCOUT_AWP_BALANCE.md` §5 and §7.2, and J:E30 L1385 ("decide whether friend scout data is evaluation, training or split by person before the first session; ask for scout-only players, who are the missing group").
- **The commit's wording**, in full: "next decision, is about friend data, since i cannot use encounters from the same player. Same style of the encounter. Need a strategy". It appears in no document. The nearest text is J:E26 L1251 (a few friends repeated many times teach the model their style) and the people-as-unit rules in `FRIEND_SESSIONS.md`. No strategy is written. [checked by search]
- **Options the author has listed**: (a) evaluation only; (b) training data (estimated ceiling about 1,400 scout encounters from 12 people × 6 matches × ~20, `SCOUT_AWP_BALANCE.md` L92); (c) split by person.
- **Status**: no session recorded; consent wording not approved (`FRIEND_SESSIONS.md` L5–6); the mode cvars are "not yet verified in CS2" (L32–34).

### D2. Scout imbalance (commit `9152d96`)

Supersession chain, so the latest position is clear:

1. J:E24 (10-19): problem framed as sniper / AWP false positives.
2. J:E25–E26 (10-19/20): `awp_v1` built against a target of +211 sniper-heavy player-maps; then found that the imbalance is the scout and the target had been computed on a count that included it (J:E26 L1241).
3. J:E27 (10-22): correction; recommendation (d) keep the AWP addition out of training and use it as an evaluation group, choose the strict operating point, report lenient AWP false positives as a limitation.
4. J:E30 (10-27/28): pro scout run done (`scout_v1`), stopped by the yield rule. Covers 42% of the 1:1 encounter gap; about 1,560 still missing (1,926 without the reserve). Only 2 of 183 added player-maps are scout-main (median scout share 17%), so "at the player level the scout imbalance is barely changed, so reweighting is likely needed".

- **Open**: balancing target (1:1 for the scout only, or weapon-neutral weights for scout, auto-sniper, Deagle, AWP; `OTHER_WEAPONS.md` §8.2); whether reserve scout encounters go to training (`SCOUT_AWP_BALANCE.md` §7.3); whether to download the remaining 253 one-scout-kill demos (about 450 encounters, about 120 GB; J:E30 L1380); where legitimate scout-main players come from at all.
- **Other sources considered**: matchmaking demos with later-ban labels (about 900 matches needed, GDPR and ethics; J:E27 L1296–1301); FACEIT API (terms unclear; L1302); friends (D1).
- **Next step the author named**: "a written reweighting protocol (LightGBM on validation, CPU) before any retraining" (J:E30 L1385). Not written.

### D3. OpenCS2 data: use it for training or not (commit `df48377`)

- **Done**: extraction of `awp_v1` and `scout_v1` (section 2.3).
- **Not done, each needing approval** (`PLAN.md` L96–108): merged dataset version (`cs2cd_v2`) with a new protocol; channel handling across sources (masking flash/footstep, derived velocity everywhere); retraining and recalibration; evaluation on new data only.
- **Tension to resolve**: the files are named and split-tagged `train`, but the current recommendation is to keep `awp_v1` out of training (PLAN update L6–10), and it has already been used once as an evaluation group (J:E27 L1283). `scout_v1`'s role is not stated beyond "training addition".

### D4. Per-weapon expert model vs retraining on everything (commit `df48377`)

- **Appears only in the commit message**, whose last clause is "next:thining about an expert or retrain the entire dataset". No document uses the word "expert" or analyses this choice. The same message mentions "researched about multiple detection of cheats"; the only matching text is J:E24 L1158–1161 (no public cheat-type labels; `GetPlayerBans` route). [checked by search]
- Nearest text: J:E26 L1247 lists "sampling or weighting by weapon, a weapon-conditioned model, or no change with the limitation reported"; J:E27 L1281–1290 evaluates (a) report only, (b) weighting, (c) per-weapon calibration (rejected, too few samples), (d) evaluation-only (recommended) for the AWP. A weapon-conditioned or expert model was never evaluated.
- **Constraint on any retraining** [checked]: the local environment is Python 3.11, torch 2.13 CPU, `torch.cuda.is_available()` is False; the neural trainer requires CUDA (`REPRODUCING.md` L7, L41); the CS2CD raw data (52.6 GB) and window cache are not on this machine. LightGBM can be retrained on CPU, but only with the cached player features, which are also not local as far as I can see. [last clause inferred]

### Other open items

| # | Item | Options or state | Source |
|---|---|---|---|
| D5 | Thesis operating point | Strict `fpr_1pct` recommended; whether the scout-shortcut finding is reported as a main result | J:E27 L1330 |
| D6 | Live flag rule | Show no flag below about 10 encounters, or require hits from more than one weapon group; changes `live_scoring.py` | `OTHER_WEAPONS.md` §7, §8.3 |
| D7 | Listener defects before larger sessions | Section 3.7 list | J:E24 L1163 |
| D8 | Ethics, consent, GDPR | Supervisor / ethics approval for the data plan; consent wording | J:E24 L1163; `FRIEND_SESSIONS.md` L5–6 |
| D9 | Pistol-only friend rounds | To measure live USP-S and Deagle false positives | `OTHER_WEAPONS.md` §8.4 |
| D10 | Phase 4 of the real-time plan | Session logging, consent procedure, evaluation against verified labels | J:E23 L1130 |
| D11 | Further model ideas | User-command features (present in CS2CD, never tested), line-of-sight features, masked pretraining, image-based detection | `ARCHITECTURE_CHOICES.md` §7.5; J L1411–1414 |
| D12 | Two failing tests | `test_private_scoring.py`, `KeyError: 'steamid'` at `scripts/score_private_match.py:159`; reported since J:E17, unfixed | J:E17 L909; line 159 still reads `p['steamid']` [checked] |

---

## 6. Gaps noticed

### Undocumented steps and results without a recorded script

- **G1. Scored demos with no write-up.** The runs listed in section 2.4 other than the StarLadder match have no journal entry; commit `ff6931d` only says "added some demos and faciet demos". From the `scores.csv` files, at the saved thresholds:
  - `cheater_demo_1` / `cheater_demo_2` (FACEIT servers, 10 players each): nobody flagged at `accuracy` or `fpr_1pct`; one player flagged at `f1` in the first (highest calibrated scores 0.255 and 0.063). There is no recorded ground truth for which player, if any, was cheating, so this is not a recall measurement.
  - Second pro match (2 maps, 20 player-maps) and the five matchmaking demos (38 player-matches): nobody flagged at any threshold (highest 0.184).
  - [checked; reading the folder name as "these demos contain cheaters" is inferred]
- **G2. No script for the pro false-positive table.** `local_runs/opencs2_pro_fpr_scores.csv` (1,170 rows) backs J:E27's table, and its flag counts match the table when the saved thresholds are applied (13, 21, 0 of 100; 2, 3, 0 of 1,069), but no script in `research/scripts/` produces it. Same for the "sniper-heavy" player-match counts and test breakdowns in J:E24 and J:E27. `analyze_weapon_groups.py` covers only the E29 tables; `count_scoped_encounters.py` only encounter counts. [checked by search]
- **G3. `scout_v1` has no plan document**; its selection rule exists only in the script and manifest. Its manifest lists 10 download failures that J:E30 does not mention. [checked]
- **G4. Journal prototype runs** (J:E10) have no saved checkpoint tied to them, no recorded seed in the journal, and no prediction arrays (`METHODS.md` L19). `models/best_lstm.pt` exists locally with no provenance note. [read + checked]
- **G5. Live-session and capacity measurements** (J:E19–E24) rest on capture files partly deleted (J:E21 L1055) and on `local_runs/` logs; no config or seed record beyond the journal text. [read]

- **G5b. Test counts do not add up.** The per-suite rows in J:E14's table (L650–657) sum to 199, the stated total is 217. The test files currently contain 132 `def test_` functions, against "248 pass; 2 failures" in `ARCHITECTURE_CHOICES.md` L102 (parametrised tests may account for the difference; not checked). [checked the arithmetic and the count only]

### Contradictions and stale statements

- **G6. `ARCHITECTURE_CHOICES.md` vs J:E30, same commit.** §5 lists only the AWP addition and §7 says "nothing here has been started" about scout data, quoting the pre-run estimates (840 to 1,950 encounters). J:E30 records the scout run as finished with 1,143 encounters. `SCOUT_AWP_BALANCE.md` §7.1 and `OTHER_WEAPONS.md` §8.1 still ask whether to download the best 100 scout demos. [read]
- **G7. Scout gap figures differ by baseline**: 3,008 (`SCOUT_AWP_BALANCE.md` §4), 2,879 (J:E27 L1294, after the 129 from `awp_v1`), about 2,700 "still needed" (after 301 staged), about 1,560 (J:E30). All consistent once the baseline is stated; none is labelled as superseding the others. [read]
- **G8. Journal tail still blames the AWP** (J L1399–1400, "flags most AWP shots as suspect"), contradicted by J:E27 (scout, not AWP; 0/100 pro AWPers at strict). J:E23 called live AWP scores low; J:E24 L1156 corrects that. [read]
- **G9. `README.md` is at the September state**: "Live integration … remain to be tested" (L9) and no mention of the real-time pipeline, OpenCS2 or the weapon findings. `.claude/CLAUDE.md`'s repository map omits `research/realtime/` and `research/opencs2/`. [read]
- **G10. Small count differences, both explained**: reserve player-maps 1,070 (manifest) vs 1,069 riflers scored (J:E27): the scores file has a 1,070th reserve row in its own group, "pro sniper-heavy (reserve)", which the journal table leaves out (it is the single pro scout-heavy player-map mentioned in `SCOUT_AWP_BALANCE.md` §6). Protocol counts 219 test positives and 119 validation matches vs 214 and 118 eligible, from the eligibility filter (`METHODS.md` L65). [checked]
- **G11. Hash chain inconsistency.** `protocol.json` → `extraction_source_sha256` and `cache.json` give `cs2_data.py` as `08043b…`; `selection.json` → `code_sha256` and the file on disk are `0e25c8…`. The file has a single commit in git (`b62c571`). Commit `f2de563` ("solved metadata of the files") rewrote hashes in six JSON files after the inference guard rejected the artifacts (J:E11 L431–435). `research/results/test_metrics.json` and `research/models/selection.json` also carry different hashes for the same paths. Cause not confirmed (line-ending conversion on checkout is a candidate). The frozen-artifact guarantee therefore depends on a manual metadata edit. [checked; cause inferred]
- **G12. Environment drift**: artifacts built with scikit-learn 1.6.1 / Python 3.13, run locally with 1.9.0 / 3.11; five `InconsistentVersionWarning`s (J:E13 L523). [read + checked versions]
- **G13. Weapon identity**: `player_hurt` reports USP-S and P2000 both as `hkp2000` and M4A1-S as `m4a1`; J:E23 calls the first live hit a "P2000 hit", J:E29 a "USP-S hit". [read]
- **G14. Entry order**: J:E28 contains a block headed "Earlier run status (paused 2026-10-19)" that predates E25's completion note; Entry 1 cites `implementation_plan.md`, deleted in `788fe64`. [read + checked]
- **G15. Committed build files**: plugin `obj/` files with local paths were committed in `9257540` and removed in `8debe31` (still in history). [read] J:E21 L1056; `git log --stat`

---

## 7. Index of the most important files

| File | Why |
|---|---|
| `PROJECT_JOURNAL.md` | Chronological record; Entries 24–30 (L1134–1385) hold the current data problem and decisions |
| `research/METHODS.md` | Definitions, evaluation units, limits of the benchmark |
| `research/ARCHITECTURE_CHOICES.md` | One-page design summary (stale on scout data, see G6) |
| `research/experiments/cs2cd_v1/protocol.json` | Pre-registered protocol, split rule, feature list |
| `research/experiments/cs2cd_v1/selection.json`, `calibration.json`, `test_results.json` | Validation choice, thresholds, all test metrics |
| `research/experiments/cs2cd_v1/events.parquet`, `players.csv`, `test_predictions.csv` | Per-encounter and per-player tables for any re-analysis without raw data |
| `research/opencs2/OTHER_WEAPONS.md` | Most complete weapon analysis; reproducible with `scripts/analyze_weapon_groups.py` |
| `research/opencs2/SCOUT_AWP_BALANCE.md`, `PLAN.md` | Scout gap, OpenCS2 plan and its unapproved steps |
| `research/realtime/server/FRIEND_SESSIONS.md` | Friend-session protocol (draft) |
| `research/realtime/server/CAPACITY.md`, `research/realtime/PHASE0_PARITY.md` | Live limits and capture parity |
| `research/anticheat/cs2_data.py`, `dem_extractor.py` | Feature definitions; demo approximations |
| `research/realtime/capture_windows.py`, `live_scoring.py` | Live window builder and scorer |
| `research/scripts/add_opencs2_awp.py`, `add_opencs2_scout.py` | How the staged pro data was selected |
| `research/data/opencs2/*/manifest.json` (local only) | Exact counts and selection parameters of the staged data |
| `local_runs/` (local only) | Pro FPR scores, live logs, demo scoring outputs including G1 |
| `.claude/CLAUDE.md` | Project invariants any analysis must respect |

---

## 8. Not verified

- **Test results.** No test was run. Pass counts (217, 239, 248) and the two `KeyError: 'steamid'` failures are journal statements; I only confirmed that line 159 still reads `p['steamid']`.
- **Artifact integrity.** `scripts/verify_artifacts.py` was not run; G11 is from reading hashes.
- **Weapon and FPR tables.** I recomputed only the scout and AWP encounter counts per split, the label-by-source table, the OpenCS2 totals and the flag counts in the pro FPR table (from its saved scores; the scoring itself was not re-run). The player-match breakdowns, Spearman values, weapon-only baseline and gap projections are as written in the documents.
- **Real-time measurements.** Parity, latency and capacity numbers are from the journal; captures were not opened. `CAPACITY.md`, `PHASE0_PARITY.md`, `server/README.md` and `realtime_detector_research.md` were read by heading and through the journal's summaries only.
- **PDFs.** `AntiCheat_Notes_and_Issues.pdf` was read as text (same content as J L1395–1414). The first lines of the two background reports and `ARCHITECTURE_CHOICES.pdf` were read; `docs/anticheat_review.pdf` was not opened.
- **`src/`, `notebooks/`, `docs/` build scripts, the plugin C# source and the Kaggle scripts** were not read line by line.
- **D: drive.** `D:\opencs2` and `D:\cs2server` exist; demo counts and sizes there were not checked.
- **Claims with no source in the repo**: that CS2CD is "primarily pub/FPL" players (J:E14 L727); that the model "successfully detects standard aimbot patterns" but misses silent aim (J L1403); the 30-day matchmaking demo retention (J:E27 L1297, marked unofficial there).
- **Ground truth for `cheaters_demos/`** is not recorded anywhere I found.
- **Whether anything happened after 2026-10-28** outside the repository (sessions, supervisor decisions): no evidence either way beyond the plugin folder dates.
