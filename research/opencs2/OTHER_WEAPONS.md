# All weapon groups: statistics, false positives and data needs

**Date:** 2026-10-27 · **Status:** research only. No training, download or change to any dataset.
**Reproduce every table:** `python scripts/analyze_weapon_groups.py` (run from `research/`, about
1 minute on a CPU). The script uses saved scores only: the validation score files, `test_predictions.csv`,
the CNN's test encounter scores, and the calibrator and thresholds frozen on the calibration split.
It asserts that it reproduces the published strict test result (67 true positives, 4 false
positives). The only model it fits is a weapon-share baseline, on the training split.
Companion to `SCOUT_AWP_BALANCE.md` (AWP and scout only).

## 1. Definitions and caveats

- An **encounter** is the first hit of an attacker on a victim in a burst, not a kill. Labels are
  **player-match** labels, and "not cheating" labels may include unreviewed cheaters.
- **Weapon groups:** AWP; Scout (SSG 08); Auto-sniper (SCAR-20, G3SG1); Scoped rifle (AUG,
  SG 553); Rifle (AK-47, M4A4/M4A1-S, Galil, FAMAS); Machine gun; SMG; Shotgun; **USP-S/P2000**;
  **Deagle/R8**; Other pistol (Glock, Tec-9, P250, Five-SeveN, Dual Berettas, CZ75).
- **The USP-S cannot be separated from the P2000 in CS2CD.** CS2CD and our extractor take the
  weapon from `player_hurt`, which reports both as `hkp2000` (and the M4A1-S as `m4a1`). In the
  OpenCS2 kill metadata, 94% of this group is the USP-S (13,858 `usp_silencer` against 959
  `hkp2000` kills), so the group is mostly USP-S.
- **Main weapon group** of a player-match = the group with most of its encounters.
- **Validation is not independent of the detector:** it chose the CNN checkpoint, LightGBM early
  stopping and the blend weights. Pooled validation + test numbers are descriptive. **Only the
  test-only columns are held-out evidence.** Thresholds come from the calibration split.
- The CS2CD window cache (`data/cs2cd/processed/`) is not on this machine, so per-channel checks
  on the windows could not be run (section 4.4).

## 2. CS2CD encounters by weapon group

Cheating / not cheating encounters. "Cheating share" is for the training split.

| Group | Train | Validation | Calibration | Test | Cheating share |
|---|---|---|---|---|---|
| AWP | 1,717 / 4,776 | 366 / 1,077 | 325 / 898 | 371 / 1,242 | 26% |
| **Scout** | **5,933 / 2,925** | 1,525 / 837 | 969 / 643 | 1,374 / 760 | **67%** |
| Auto-sniper | 289 / 366 | 129 / 113 | 46 / 76 | 53 / 96 | 44% |
| Scoped rifle | 21 / 1,072 | 6 / 240 | 3 / 165 | 13 / 272 | 2% |
| Rifle | 2,378 / 34,835 | 482 / 7,999 | 322 / 5,776 | 438 / 8,725 | 6% |
| Machine gun | 24 / 538 | 0 / 88 | 18 / 61 | 1 / 118 | 4% |
| SMG | 337 / 8,298 | 66 / 2,023 | 51 / 1,402 | 70 / 2,056 | 4% |
| Shotgun | 110 / 2,170 | 25 / 404 | 23 / 233 | 74 / 465 | 5% |
| **USP-S/P2000** | 621 / 5,119 | 159 / 1,188 | 127 / 847 | 165 / 1,273 | **11%** |
| **Deagle/R8** | **1,962 / 4,973** | 426 / 1,175 | 303 / 828 | 478 / 1,271 | **28%** |
| Other pistol | 649 / 6,909 | 146 / 1,605 | 74 / 1,170 | 160 / 1,697 | 9% |
| Pistol (all) | 3,232 / 17,001 | 731 / 3,968 | 504 / 2,845 | 803 / 4,241 | 16% |
| **All weapons** | 14,041 / 71,981 | | | | **16.3%** |

Player-matches by main group, cheating / not cheating (train · validation · calibration · test):
AWP 78/225 · 23/44 · 16/41 · 23/48; Scout 397/170 · 114/51 · 55/42 · 111/36; Rifle 157/2,532 ·
30/585 · 23/405 · 39/613; USP-S/P2000 13/109 · 7/27 · 3/14 · 8/22; Deagle/R8 71/163 · 16/42 ·
13/29 · 18/47; Other pistol 12/122 · 7/42 · 0/27 · 8/34. All other groups have under 300 player-matches
in total.

**Finding.** Only the scout has more cheating than not-cheating encounters. But measured against
the overall cheating share (16.3%), four groups carry label information by themselves: **scout
(67%), auto-sniper (44%), Deagle (28%) and AWP (26%)**. The USP-S group (11%) is *below* average.

## 3. False positives by main weapon group

Legitimate player-matches, validation + test (pooled) and test only. Exact 95% intervals in
brackets. Thresholds (calibrated score): `accuracy` 0.257, `f1` 0.211, strict `fpr_1pct` 0.853,
all chosen on the calibration split.

| Main group | Legit players | FPR `accuracy` (pooled) | FPR strict (pooled) | FPR `accuracy` (test only) | FPR strict (test only) | Cheaters | Recall `f1` / strict | Precision `f1` / strict |
|---|---|---|---|---|---|---|---|---|
| AWP | 92 | 22/92 = 23.9% [15.6–33.9] | 0/92 [0–3.9] | 13/48 = 27.1% | 0/48 | 46 | 0.87 / 0.09 | 40/65 / 4/4 |
| **Scout** | 87 | **56/87 = 64.4%** [53.4–74.4] | **15/87 = 17.2%** [10.0–26.8] | 20/36 = 55.6% | 4/36 = 11.1% | 225 | 0.99 / 0.48 | 223/281 / 109/124 |
| Auto-sniper | 7 | 2/7 | 0/7 | 1/5 | 0/5 | 10 | 1.00 / 0.10 | 10/12 / 1/1 |
| Scoped rifle | 20 | 1/20 | 0/20 | 0/11 | 0/11 | 0 | — | — |
| Rifle | 1,198 | 14/1,198 = 1.2% [0.6–2.0] | 0/1,198 [0–0.3] | 6/613 = 1.0% | 0/613 | 69 | 0.61 / 0.06 | 42/59 / 4/4 |
| SMG | 154 | 2/154 = 1.3% | 0/154 | 1/74 | 0/74 | 1 | 0 / 0 | 0/2 / 0/0 |
| Shotgun | 34 | 0/34 | 0/34 | 0/20 | 0/20 | 5 | 0.80 / 0 | 4/4 / 0/0 |
| **USP-S/P2000** | 49 | 4/49 = 8.2% [2.3–19.6] | 1/49 = 2.0% [0.1–10.9] | **0/22** [0–15.4] | **0/22** [0–15.4] | 15 | 0.93 / 0.07 | 14/19 / 1/2 |
| **Deagle/R8** | 89 | 12/89 = 13.5% [7.2–22.4] | 0/89 [0–4.1] | 5/47 = 10.6% [3.5–23.1] | 0/47 [0–7.5] | 34 | 0.88 / 0.15 | 30/45 / 5/5 |
| Other pistol | 76 | 5/76 = 6.6% [2.2–14.7] | 0/76 [0–4.7] | 2/34 = 5.9% | 0/34 | 15 | 0.67 / 0.13 | 10/17 / 2/2 |
| Pistol (all) | 214 | 21/214 = 9.8% [6.2–14.6] | 1/214 = 0.5% [0.0–2.6] | 7/103 = 6.8% [2.8–13.5] | 0/103 [0–3.5] | 64 | 0.84 / 0.12 | 54/81 / 8/9 |

Machine-gun players (16) are omitted: 1 flagged at `accuracy`, none at strict. Overall strict
FPR: validation 12/906 = 1.3%, test 4/916 = 0.4%.

**Finding.** At the strict threshold, only **scout** players are falsely flagged in any number.
At the lenient thresholds, scout (64%), AWP (24%) and **Deagle (13.5%)** players are flagged far
more often than rifle and SMG players (about 1%). USP-S-main players (8.2% pooled, 0 of 22 on test)
sit with the other pistols.

## 4. The USP-S question

### 4.1 Player level: no USP-S effect

- **USP-S share does not predict a higher score** for legitimate players: Spearman ρ = −0.03
  (p = 0.18). For comparison: scout +0.34, AWP +0.25, **Deagle +0.25**, rifle −0.34, SMG −0.40.
- Legitimate players binned by weapon share, FPR at `accuracy` (validation + test; bin edges fixed
  before the first run):

| Share of encounters | USP-S/P2000 | Deagle/R8 | Scout (reference) |
|---|---|---|---|
| 0 | 79/681 = 11.6% | 38/987 = 3.9% | 37/1,458 = 2.5% |
| 0–10% | 18/612 = 2.9% | 19/342 = 5.6% | 7/157 = 4.5% |
| 10–20% | 9/359 = 2.5% | 22/247 = 8.9% | 7/85 = 8.2% |
| 20–35% | 8/112 = 7.1% | 15/134 = 11.2% | 10/49 = 20.4% |
| >35% | 5/58 = 8.6% [2.9–19.0] | **25/112 = 22.3%** [15.0–31.2] | **58/73 = 79.5%** |

  USP-S has no steady trend. The "0" row is high, probably because it holds most of the
  sniper-heavy players (not checked separately). The
  highest USP-S bin (8.6%) is 0 of 27 on test only. The **Deagle rises steadily with share,
  the same shape as the scout but weaker**, and stays at 0 at the strict threshold (0/112).

### 4.2 Single encounters: high scores are rare, but they happen

The CNN's per-encounter score on test (player labels copied to encounters, so these are clustered
and only indicative):

| Group | Legit encounters | Median | Legit share > 0.5 | Cheating median | AUC within group |
|---|---|---|---|---|---|
| Rifle | 8,725 | 0.004 | 1.2% | 0.053 | 0.76 |
| **USP-S/P2000** | 1,273 | 0.005 | **3.2%** | 0.241 | 0.83 |
| Other pistol | 1,697 | 0.002 | 2.1% | 0.196 | 0.84 |
| Deagle/R8 | 1,271 | 0.015 | 7.5% | 0.369 | 0.82 |
| AWP | 1,242 | 0.123 | 17.2% | 0.565 | 0.76 |
| Scout | 760 | 0.361 | 40.3% | 0.948 | 0.77 |

About 1 in 30 legitimate USP-S hits scores above 0.5 by itself.

### 4.3 This explains the live observation

In the 11 October live session (`local_runs/live_20261011.jsonl`), the **first hit of the
session was a USP-S hit** with an encounter score of 0.93. With 1 encounter, the provisional player
score was 0.73, flagged at `accuracy`, `f1` and `default_0_5`. After 3 encounters it was 0.26 and
after 18 it was 0.12 (not flagged). The other two USP-S hits scored 0.05 and 0.03.
Every half starts with a pistol round, so **the first provisional scores of every player come
from pistols**, and one high pistol hit is enough for a provisional flag.

### 4.4 Is it the scout-style shortcut?

- **Weapon-only baseline** (logistic regression on each player's weapon-group shares, fitted on
  train): AUC 0.876 validation / 0.874 test. Source-only: 0.881 / 0.888. Detector: 0.943 / 0.973.
  Among players with at least 20% pistol encounters (664 on test, 127 cheating) the detector keeps
  AUC 0.974 while weapon-only drops to 0.837, so pistol-heavy players are not ranked by weapon style.
- Weapon-only coefficients (positive = more cheating): scout +3.97, AWP +1.68, auto-sniper +1.48,
  **Deagle +1.42**, USP-S/P2000 −0.45, other pistol −0.85. The labels make a Deagle-heavy player
  look more suspicious; a USP-S-heavy player does not.
- Inputs: in LightGBM, `shots_in_burst` carries 38% of the gain (rank 1); `weapon_pistol` 0.5%,
  `scoped` 0.9%, `recoil_index` 0.4%, punch about 1% each. A plausible but **untested** mechanism:
  semi-automatic pistols often hit on the first shot, which matches the first-shot pattern of
  cheaters. Testing it needs the CS2CD window cache, which is not on this machine.

**USP-S conclusion (confidence: medium).** There is no player-level USP-S false-positive problem
in CS2CD: 0 of 22 USP-S-main legitimate players flagged on test, no trend with USP-S share, and a
below-average cheating share. The observed "false detections" are **provisional flags from single
pistol hits early in a session**. That is a live-scoring rule issue, not a data gap. The limit:
only 49 USP-S-main legitimate players (test-only upper bound 15%).

## 5. Gap analysis (training split)

Two targets. **1:1** = as many not-cheating as cheating encounters (the target in
`SCOUT_AWP_BALANCE.md`). **Weapon-neutral** = every group at the overall cheating share (16.3%),
so the weapon alone tells the model nothing. Staged pro encounters (training addition + reserve)
are subtracted.

| Group | Cheating share | Staged pro | Needed for 1:1 | Needed for weapon-neutral |
|---|---|---|---|---|
| Scout | 67% | 301 | **2,707** | **27,338** |
| Deagle/R8 | 28% | 627 | 0 | **4,507** |
| AWP | 26% | 2,810 | 0 | **1,259** |
| Auto-sniper | 44% | 0 | 0 | **1,123** |
| USP-S/P2000 | 11% | 1,828 | 0 | 0 |
| All other groups | 2–9% | — | 0 | 0 |

## 6. Pro-demo yield (estimates; method of `SCOUT_AWP_BALANCE.md` §3.2)

Encounters per kill from the 117 usable processed demos; 8% extraction failures; 1,861 demos still
unprocessed. Demos ranked by that group's kills.

| Group | Enc. per kill | Still obtainable (est.) | Demos for 1:1 | Demos for weapon-neutral |
|---|---|---|---|---|
| Scout | 2.32 | about 1,950 | not enough | not enough |
| Deagle/R8 | 1.80 | about 9,270 | 0 | **402 demos, 193 GB** |
| AWP | 1.08 | about 25,300 | 0 | **31 demos, 24 GB** |
| Auto-sniper | — | about 0 (5 kills in all pro demos) | 0 | not obtainable |
| USP-S/P2000 | 2.04 | about 26,100 | 0 | 0 |

The demos that are best for the scout also carry other weapons. **The best 100 scout demos
(45 GB)** would bring about 840 scout, **1,560 AWP**, 560 Deagle and 1,440 USP-S encounters; the
best 200 (88 GB) about 1,270 scout, 2,980 AWP and 1,110 Deagle.

## 7. Recommendation

| Group | What it needs | Why |
|---|---|---|
| **USP-S/P2000** | **No data.** A live-scoring rule: show no flag until a player has at least 10 encounters, or require hits from more than one weapon group | The false detections are single-hit provisional flags (§4.3); the player-level rate is not raised |
| **Scout** | Data **and** reweighting | Pro data covers at most about 72% of the 1:1 gap; the weapon-neutral gap is not reachable with data |
| **AWP** | Comes free with the scout download | The best 100 scout demos close the weapon-neutral AWP gap |
| **Deagle/R8** | **Reweighting**, not a download | Neutral needs 193 GB; the scout download gives about 12% of it. The effect is lenient-threshold only |
| **Auto-sniper** | Reweighting or nothing | No pro supply; 7 legitimate players in val + test, too few to measure |
| Rifle, SMG, shotgun, other pistols, scoped rifle | Nothing | False-positive rates about 1% or lower at lenient thresholds, 0 at strict |

**Smallest download that covers the data-fixable gaps:** the **best 100 scout demos, about 45 GB**
(unchanged from `SCOUT_AWP_BALANCE.md`). It covers about 31% of the scout 1:1 gap and all of the
weapon-neutral AWP gap. More demos give less per GB: the best 200 (88 GB) add about 430 scout
encounters. Deagle and auto-sniper are better handled by per-group weights.

## 8. Open decisions

1. Whether to download the best 100 scout demos (45 GB). Nothing in this analysis needs more.
2. Balancing target: 1:1 for the scout only, or weapon-neutral weights for scout, auto-sniper,
   Deagle and AWP (the latter fixes the lenient-threshold Deagle and AWP rates too).
3. Live scoring: the minimum number of encounters before any flag is shown (changes
   `realtime/live_scoring.py`; needs approval).
4. Whether friend sessions should include pistol-only rounds, to measure live USP-S and Deagle
   false-positive rates on new players (the only way to get independent evidence).
5. Retraining is still not approved; any data or weight change needs recalibration and a new
   independent test group.
