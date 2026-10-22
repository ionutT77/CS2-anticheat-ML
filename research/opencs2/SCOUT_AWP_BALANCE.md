# Scoped-weapon data: statistics and balance plan

**Date:** 2026-10-22 · **Status:** statistics only; no training, no dataset merge.
**Reproduce the CS2CD and staged-pro tables:** `python scripts/count_scoped_encounters.py`
(run from `research/`). The pro-demo projections in section 3 were computed from the OpenCS2
metadata in `D:\opencs2\metadata` and the extraction manifest; the method is described there.

## 1. Definitions and caveats

- An **encounter** is the first hit of an attacker on a victim in a burst. It is **not a
  kill**: it counts whether or not the victim died.
- Labels belong to the **player-match**. A "cheating" encounter comes from a player labelled
  as a cheater, not necessarily from a cheating shot. "Not cheating" labels may include
  unreviewed cheaters (CS2CD's negatives are only partly verified).
- Weapon groups: **AWP**; **Scout** (SSG 08); **auto-sniper** (SCAR-20, G3SG1); **scoped
  rifle** (AUG, SG 553).
- All OpenCS2 data is from professional matches and is labelled **not cheating**.

## 2. CS2CD (the original dataset)

| Weapon | Training split: cheating / not cheating | All splits: cheating / not cheating |
|---|---|---|
| **AWP** | 1,717 / 4,776 | 2,779 / 7,993 |
| **Scout (SSG 08)** | **5,933 / 2,925** | **9,801 / 5,165** |
| Auto-sniper | 289 / 366 | 517 / 651 |
| Scoped rifle | 21 / 1,072 | 43 / 1,749 |

("All splits" includes validation, calibration and test. Only the training split is used to
train.)

**Player-match view** (sniper-heavy = more than 50% of a player-match's encounters use a
sniper weapon; main weapon = the more frequent of scout and AWP): training split, legitimate
vs cheating, AWP-main **85 vs 54**, scout-main **122 vs 364**.

**Finding:** the AWP is not imbalanced (more not-cheating than cheating). The imbalance is
the **scout**: about 2 cheating encounters per not-cheating one.

## 3. Professional matches (OpenCS2, HLTV demos)

### 3.1 Already extracted (127 demos processed, 117 usable, 10 skipped as damaged or incomplete)

| Weapon | Training addition | Reserve | Total | Label |
|---|---|---|---|---|
| AWP | 1,330 | 1,480 | 2,810 | not cheating |
| Scout | **129** | **172** | **301** | not cheating |
| Scoped rifle | 0 | 6 | 6 | not cheating |

The addition holds 100 AWP player-maps from 78 players. The reserve holds every other player
on those maps; it comes from the same matches, so it is **not** an independent test set.

### 3.2 Still obtainable from the 1,871 unprocessed demos (*estimates*)

From the metadata: 25,655 AWP kills and **914 scout kills** (in 491 demos) remain.

- **Encounters per kill**, measured on the processed demos: AWP **1.08** (2,596 kills → 2,810
  encounters); scout **2.32** (130 kills → 301 encounters). The scout ratio is high because
  scout hits often do not kill, and it rests on only 130 kills, so treat it as ±30%.
- About 8% of demos fail extraction (10 of 127).
- Disk cost: about 0.52 GB per demo.

**Scout: demos ranked by scout kills (best first)**

| Demos downloaded | Scout kills | Scout encounters (estimate) | Download |
|---|---|---|---|
| 25 | 146 | about 310 | 12 GB |
| 50 | 242 | about 520 | 24 GB |
| 100 | 392 | about 840 | 46 GB |
| 200 | 594 | about 1,270 | 93 GB |
| 300 | 723 | about 1,540 | 146 GB |
| all 491 | 914 | about 1,950 | 228 GB (about 28 hours at 25 Mbps) |

**AWP:** about 25,600 more encounters exist, but would cost roughly 1 TB. They are not
needed: AWP is already balanced.

## 4. How many encounters are needed to balance (1:1, training split)

| Weapon | Gap | Staged pro data | Still needed |
|---|---|---|---|
| AWP | none (4,776 not cheating vs 1,717 cheating) | — | **0** |
| **Scout** | 5,933 − 2,925 = **3,008** | 301 | **about 2,700** |
| Auto-sniper, scoped rifle | none | — | 0 |

Only the 129 scout encounters in the training addition are currently on the training side;
the 172 in the reserve could be added to training later.

## 5. Where the scout data can come from

| Source | Scout encounters | Share of the 2,700 still needed |
|---|---|---|
| Best 100 pro demos (46 GB) | about 840 | about 31% |
| All remaining pro scout demos (228 GB) | about 1,950 | about 72% |
| Friends, scout-only sessions | about 1,400 at most (12 people × 6 matches × ~20 encounters, estimate) | about 52% |

- Pro data alone **cannot** close the gap: even all of it covers about 72%, and the estimate
  has a wide margin.
- Using the friend data for training removes it as an evaluation group
  (`research/realtime/server/FRIEND_SESSIONS.md`). That choice must be made **before the first
  session**.
- Whatever is added changes the class balance, so calibration and thresholds must be refitted
  and a new independent test group is needed.

## 6. Risks of adding pro scout data

- **Source shortcut:** the pro data is entirely not-cheating and comes from GOTV demos
  (velocity derived from position, flash and footstep channels mostly empty). The model could
  learn "this source means legitimate". Check with the source-only diagnostic after merging.
- **Skill and context:** pro scouts are mostly used in particular rounds, so their play may
  not resemble the scout players in CS2CD.
- **Fast, precise flicks are legitimate in the pro data,** but the effect on this model is
  **not yet measured for the scout:** only 1 pro scout-heavy player-map was scored (score
  0.303, above the `accuracy` threshold and below the strict one). Measure the current model
  on the staged pro scout players before and after any retraining.
- Reweighting can cover what the added data does not, without deleting any data.

## 7. Open decisions

1. How many pro scout demos to download (suggested first step: the best 100, 46 GB).
2. Whether friend scout data is for training, evaluation, or split by person.
3. Whether to add the 172 reserve scout encounters to training.
4. Retraining is not approved; any retraining needs a new protocol, a new test group,
   recalibration, and a CUDA GPU for the neural models (the LightGBM part runs on CPU).
