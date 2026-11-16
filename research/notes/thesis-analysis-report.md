# Thesis analysis report: CS2 anti-cheat (2026-11-16)

Code revision `fb5f39b` plus `research/experiments/thesis_analysis_v1/` (branch `thesis-analysis`).
Nothing in the frozen `cs2cd_v1` experiment, the pipeline code or the staged data was changed.

**Evidence tags.** **[V]** verified here by running code on files in this repository.
**[I]** inferred: my conclusion from the evidence, not directly measured. **[E]** external source, linked.

Nothing in this report is validated live anti-cheat performance. The project has no independently
collected deployment data with verified labels; section 8 says what would be needed.

---

## 1. Conclusions first

**1. The ranking result is real, but the dataset's cheaters are mostly not the cheaters a server admin
needs to catch.** 80% of all cheater labels are in matches with four or more labelled cheaters, and 55%
are in "scout lobbies" (matches where at least three players mainly use the scout), with 5.9 labelled
cheaters per such match on average [V]. The weapon mix of the positive class (scout, auto-sniper,
Deagle/R8) and the number of cheaters per match are what hack-versus-hack lobbies look like [I]. A lone
cheater in an otherwise normal match, which is the deployment case, is rare in the data: 47 of 317
reviewed matches, and 6 to 9 such players per evaluation split [V].

**2. The headline AUC survives every control I could run; the strict operating point does not.**
AUC stays at 0.94 to 0.97 inside reviewed matches, between players of the same match, outside scout
lobbies, and with all sniper encounters removed from the neural component [V]. But strict-threshold
recall (the reported 31%) is 46% inside scout lobbies and **11% outside them** on test (35% and 9% on
validation + calibration) [V]. For a cheater who is alone in their match it is 0 of 6 on test and 0 of 7
on calibration [V]. The strict policy is, in practice, a scout-lobby detector. The number that describes
ordinary matches is "about 10% recall at the strict threshold", not 31%.

**3. Every strict "false positive" in every split is in the unreviewed folder, in a scout-heavy match,
and those matches are bunched together in the publisher's numbering.** 22 label-0 players are above the
strict threshold across validation, calibration and test; all 22 are in `no_cheater_present`, 0 of 574
reviewed negatives are, 20 of 22 are scout-main, and they sit in 12 matches [V]. 22 of the 29 unreviewed
scout lobbies are numbered 46 to 104 [V]. The most likely reading is unlabelled hack-versus-hack lobbies
collected from the same accounts' match histories [I]. Two consequences: the six calibration negatives
that set the strict threshold are in three such matches, so the threshold itself rests on probable label
errors; and the same accounts probably recur across train, validation, calibration and test, which the
match split cannot prevent.

**4. On 1,132 professional players (3,260 player-maps) the frozen model raises no flag at 0.5 or at the
strict threshold** (highest score 0.496) [V]. Lenient flags are confined to AWP and scout users (8% of
AWP-main player-maps at the `accuracy` threshold, 0 of 2,723 rifle-main) [V]. This is the best evidence
the project has that the model does not simply flag skill. It is not a deployment result: one input
channel is missing in these demos and the players are all professionals.

**What this means for the thesis claim.** "AUC 0.973 on CS2CD" is defensible as a benchmark ranking
result. "31% recall at 0.44% false-positive rate" is not defensible as a description of what the detector
would do on a normal server; it should be reported together with the lobby split, and the claim should be
restated as: good ranking of cheater-labelled players within CS2CD, high-confidence detection mostly of
blatant sniper-lobby cheating, untested on subtle cheats and on independent labelled data.

---

## 2. Verified state of the project

| Check | Result |
|---|---|
| Headline numbers | Reproduced from saved predictions: test AUC 0.9727; strict 67 TP / 4 FP at 0.8528; validation blend 0.9425 / 0.9496; calibration strict 20 TP / 6 FP of 625 [V] (`00_gates.py`) |
| Artifact chain | `verify_artifacts.py`: 41 files and the selection/calibration chain verify [V] |
| Tests | 252 pass, 2 fail (`test_private_scoring.py`, `KeyError: 'steamid'`), as the journal says. The survey's count of 132 test functions is consistent once parametrised cases are counted [V] |
| CPU re-scoring parity | Re-scoring `awp_v1` on this machine reproduces `local_runs/opencs2_pro_fpr_scores.csv` to 3e-8 on all 1,170 rows [V] (`02_score_pro.py`) |
| Per-encounter file | `tcn39_seed123_test_events.npz` holds probabilities, not logits; the mean of their logits rebuilds the player column to 4e-4 [V] |

**Where the survey note or the documents were wrong or incomplete**

- The survey says the 39-channel window cache is not hosted; `data/cs2cd/metadata/windows_repo.json`
  points to the publisher's own 256-tick windows (`CS2CD/Context_window_256`, 9.4 GB), which is a
  different product (kill-centred, includes 32 ticks after the kill) and is not used by any script [V].
- The survey says three of the four strict test false positives are in one match. Correct, and the same
  pattern holds on calibration and validation, which no document says (section 4.3) [V].
- The survey and journal treat OpenCS2 pro windows as "staged tables"; the full raw windows of 326 pro
  maps (74,045 encounters, with stable player identities) are on disk, so the frozen model can be run
  on them locally [V].
- `PLAN.md` describes `scout_v1` as never scored. It has now been scored once (section 5.4) and is no
  longer a fresh cohort.
- `METHODS.md` says the reviewed cohort supports "a signal beyond source membership". True, but it does
  not mention that weapon choice alone reaches AUC 0.85 to 0.88, including inside reviewed matches [V].

---

## 3. What the data is (structure nobody had written down)

All from labels, weapons and match numbers; no model scores involved [V] (`01`, `03`, `05`).

| | Matches | Player-matches | Cheater labels | Mean cheaters per match |
|---|---|---|---|---|
| Reviewed, scout lobby | 121 | 1,044 | 711 | 5.9 |
| Reviewed, other | 196 | 1,807 | 583 | 3.0 |
| Unreviewed, scout lobby | 29 | 264 | 0 | 0 |
| Unreviewed, other | 448 | 4,393 | 0 | 0 |

- Cheater labels per reviewed match: 1 in 47 matches, 2 in 41, 3 in 39, 4 in 51, 5 in 59, 6 to 10 in 80.
- Scout share of a cheater's encounters rises with the number of cheaters in the match: 17% with one
  cheater, 41% with five, 68% with nine.
- Maps differ by folder: dust2 and mirage are 173 of 317 reviewed matches and 85 of 478 unreviewed.
- Neighbouring match numbers resemble each other far more than chance (number of scout-main players,
  lag-1 correlation 0.38 unreviewed and 0.28 reviewed, shuffled 0.00, p < 0.001; same map 25% vs 7%).
  10 of 14 held-out unreviewed lobby matches have a training lobby match within five numbers (14% for
  other held-out matches).
- [E] The publisher collected matches from a match-sharing site that lists each account's 50 most recent
  matches and flags VAC-banned participants; labels in the 317 ban-present matches were manually
  reviewed, with "uncertain" labelled non-cheater; the 478 other matches were not reviewed. In a
  50-match sample the non-cheater label was 97.2% precise in no-ban matches, and the VAC flag alone had
  24.7% recall in ban-present matches
  ([AntiCheatPT, arXiv 2508.06348](https://arxiv.org/html/2508.06348v1)). So about 2.8% of unreviewed
  "legitimate" players are expected to be cheaters, and reviewed negatives are not clean either.

---

## 4. The data issues, one by one

### 4.1 Source folder shortcut: small. Confidence high.

- Among label-0 players, the score separates the two folders weakly: AUC 0.60 (validation), 0.61
  (calibration), 0.69 (test); median score 0.023 unreviewed vs 0.036 to 0.041 reviewed [V].
- Inside reviewed matches the detector's AUC is 0.950 / 0.962 / 0.971, and between players of the same
  match 0.938 / 0.961 / 0.967 (match-cluster intervals in `records/01_cs2cd_hypotheses.txt`) [V]. A
  folder signal cannot produce within-match ranking.
- Share of performance it could explain: little of the ranking. It does not explain the strict flags
  either: all strict false positives are in the *unreviewed* folder, the opposite direction.

### 4.2 Scout: the model is partly a weapon-context detector. Confidence high for the size, medium for the mechanism.

- Weapon shares alone (logistic regression fitted on train): AUC 0.880 validation, 0.846 calibration;
  0.883 / 0.857 inside reviewed matches. Detector: 0.943 / 0.941 [V].
- If legitimate players had the same main-weapon mix as cheaters (post-stratification), AUC falls from
  0.942 to 0.825 on validation + calibration (0.973 to 0.904 on test); strict recall from 24% to 10%
  (31% to 13% on test); strict FPR rises from 1.2% to 2.6% (0.44% to 1.5%) [V].
- Scout-main players: AUC 0.74 (validation + calibration), 0.85 (test); strict false positives 16 of 93
  and 4 of 36 legitimate scout-main players [V].
- Players with no scout encounter at all: AUC 0.927 / 0.964; strict flags 5 of 98 cheaters and 0 of
  1,208 legitimate (6 of 61 and 0 of 732 on test) [V].
- Neural component with scout encounters removed (test, attribution only): AUC 0.943 vs 0.959 for the
  same players; with all sniper encounters removed 0.936 vs 0.960. The median cheater score drops from
  0.72 to 0.30 to 0.40 [V]. So the ranking does not depend on sniper encounters; the *confidence* does.
- Encounter level (test): a legitimate player's scout encounter scores at least 0.5 in 40% of cases, AWP
  17%, rifle 1.2%; within-weapon encounter AUC is 0.71 to 0.84 for every group [V].
- How much of the reported performance it explains: most of the strict-threshold recall (57 of 67 strict
  detections on test are inside scout lobbies), and roughly 0.07 to 0.12 of AUC.

### 4.3 Strict false positives: almost certainly not ordinary false positives. Confidence medium-high.

- All 22 across the three splits are unreviewed-folder players; 20 are scout-main; 12 matches; in most of
  those matches five to nine of the ten players score above 0.5 (`records/01_cs2cd_hypotheses.txt`, T6) [V].
- Outside scout lobbies the strict threshold flags 5 of 1,309 legitimate players on validation +
  calibration and 0 of 847 on test [V].
- Three readings: (a) unlabelled cheaters in hack-versus-hack lobbies nobody was banned in; (b) a
  legitimate scout-only community mode; (c) a real model failure on scout play. (a) fits the
  match-number clustering and the publisher's own 2.8% estimate; (c) is partly true as well, because pro
  scout users also score higher (section 5.4), but no pro reaches 0.5. I cannot separate (a) from (b)
  without the demos. [I]

### 4.4 Thresholds: nothing below about 0.5% FPR is resolvable, and the strict threshold is unstable. Confidence high.

- Strict FPR with match-cluster intervals: calibration 0.96% [0.0, 2.2]; test 0.44% [0.0, 1.2];
  validation 1.32% [0.4, 2.5] [V].
- Zero false positives among 625 negatives bounds FPR at 0.48%; among 916 at 0.33%. Showing FPR below
  0.1% needs about 3,000 clean negatives, and they must be independent matches, not players of the same
  few lobbies [V].
- Re-deriving the strict threshold on resampled calibration matches gives 0.69 to 0.88 (95%), with test
  FPR 0.33% to 1.6% and test recall 24% to 57% [V]. The reported 31% is one draw from that range.
- The threshold is set by three lobby matches. Derived on calibration negatives outside scout lobbies it
  would be 0.57, which on validation outside lobbies gives recall 42% at FPR 2.6% (calibration and
  validation only; test not used) [V]. The 1% target does not transfer between splits at this sample size.
- The `fpr_0_1pct` policy is not supported by the data and should be dropped from the thesis.
- The Platt map under-states risk in the middle: players scored 0.2 to 0.4 are cheater-labelled 45% to
  56% of the time in all three splits [V].

### 4.5 Same-person leakage: likely present, size unknown. Confidence medium that it exists, low on size.

- Match numbering shows collection in runs of similar matches (section 3), and the publisher collected
  from per-account match histories [V + E]. A random match split therefore very probably puts the same
  accounts on both sides, most of all the lobby players [I].
- Mechanism check on professionals, where identity is known: once the main weapon is accounted for, the
  person explains about 10% of score variance (ICC 0.10; 0.54 before, of which weapon role is nearly
  all); a player is re-identified across series from the window summaries 0.8% of the time (chance 0.1%)
  [V]. So for legitimate play, "style" is mostly weapon role, not a personal fingerprint. This says
  nothing about cheaters, whose "style" is their cheat configuration.
- Mouse-sensitivity step as an account linker: stable for 62% of professionals across maps and shared by
  a median of 22 people, so too weak alone [V].
- How to bound it: a split by contiguous blocks of match numbers (Kaggle spec K2). If AUC drops under the
  blocked split, the drop is an upper bound on what adjacency (people, lobbies, cheat builds) contributes.

### 4.6 Noisy negatives: they make the model look worse at the lenient thresholds and hide nothing at the strict one. Confidence medium.

- With the publisher's 2.8%, about 20 of the 711 unreviewed test negatives are cheaters [E + V].
- Strict policy: measured FPR 0.44%; between 0% and 0.45% after correction; recall between 29% and 30%.
- `accuracy` policy: measured FPR 5.35%; between 3.3% and 5.5%; precision between 0.79 and 0.88 [V].
- On professionals, who are a clean negative set, FPR at 0.5 is 0 of 3,260 against 1.5% to 2.2% for CS2CD
  players outside lobbies [V]. Part of that gap is label noise and part may be domain shift.
- [E] The same pattern is reported elsewhere: a 5EPlay study estimates 1.5 to 2% uncaught cheaters among
  aimbot negatives, and a manual review of its top false positives found cheaters who were then banned
  ([Hawk, arXiv 2409.14830](https://arxiv.org/html/2409.14830v2)).

### 4.7 Professional demos and domain shift. Confidence high for the counts, medium for interpretation.

`scout_v1` was scored here for the first and only time; `awp_v1` for the second time (parity check).

| Cohort (all assumed legitimate) | Player-maps / people | `accuracy` 0.257 | `f1` 0.211 | 0.5 | strict 0.853 |
|---|---|---|---|---|---|
| `awp_v1` AWPers | 100 / 78 | 13.0% [7.1, 21.2] | 21.0% | 0 | 0 |
| `awp_v1` others | 1,070 / 650 | 0.3% | 0.4% | 0 | 0 |
| `scout_v1` with 3+ scout encounters (first use) | 183 / 136 | 4.9% [2.3, 9.1] | 7.7% | 0 | 0 |
| `scout_v1` others (first use) | 1,907 / 951 | 0.5% | 0.8% | 0 | 0 |
| All AWP-main | 405 / 178 | 8.4% [5.9, 11.5] | 12.6% | 0 | 0 |
| All rifle-main | 2,723 / 1,028 | 0 | 0 | 0 | 0 |
| All | 3,260 / 1,132 | 1.1% | 1.7% | 0 [0, 0.11] | 0 [0, 0.11] |

Exact 95% intervals by player-map; person-level bootstrap intervals are in `records/04_pro.txt` and are
close to these. 32 of 1,132 people are flagged at least once at `accuracy`.

- Only one professional player-map is more than half scout, so professionals cannot test the scout
  problem. Scout share still moves the score: 10% flagged at `accuracy` when 20 to 50% of encounters are
  scout, 0.4% with none [V].
- Channel shift against the CS2CD training statistics: `victim_footstep` is zero in every pro window; the
  other channels are within about 0.25 standard deviations for non-sniper encounters [V]. Whether the
  missing channel lowers scores cannot be tested here (needs CS2CD windows; Kaggle spec K3).
- What it says about shift: pros (clean, high skill) score *lower* than CS2CD negatives, not higher.
  Skill alone does not trigger the model; sniper use does, at the lenient thresholds. Ordinary players
  are still untested.

### 4.8 Local demos without ground truth

See section 5.5. They cannot measure recall or FPR. Use: pipeline smoke tests and unlabelled
domain-shift examples only.

### 4.9 Problems not on the list

- **The positive class is not representative** (section 1, conclusion 1). This is the largest threat to
  the thesis claim and is not in any document.
- **Lone-cheater recall is unmeasured for practical purposes**: 6 to 9 players per split.
- **The blend is not shown to beat LightGBM alone** (difference 0.005, interval includes 0, already in
  `METHODS.md`), and weapon shares alone reach 0.85 to 0.88. The honest baseline table for the thesis is:
  weapon shares 0.85 to 0.88, source 0.88, LightGBM 0.968 (test), blend 0.973.
- **Eligibility is label-dependent** (98.9% of cheaters vs 93.6% of others have an eligible encounter);
  the AUC is conditional on having hit someone.
- **Encounter count is not a shortcut** (AUC 0.37 to 0.47; cheaters have slightly fewer encounters) [V].
- **Validation, calibration and test disagree** more than the intervals suggest (strict recall 29%, 17%,
  31%; AUC 0.943, 0.941, 0.973). With lobby matches arriving in clusters, 79 to 120 matches per split is
  a small number of independent units.
- **Authorship and process.** The `research/` benchmark arrived in one commit from a second author, and
  several journal entries are written by an AI assistant. A committee will ask who designed, ran and
  checked what. The thesis needs an explicit contribution statement.
- **Hash chain** (survey G11): the frozen-artifact guarantee depends on a manual metadata edit; the
  artifacts verify today, but the history should be stated.

---

## 5. Experiments run today

All in `research/experiments/thesis_analysis_v1/`; outputs in its `records/` folder. Common record:
code revision `fb5f39b` + that folder; data `cs2cd_v1` (pinned CS2CD revision `44e5129…`) and the staged
OpenCS2 sets (`912a363…`); frozen match split; bootstrap 2,000 resamples of whole matches, seed
20261116; frozen Platt map and thresholds; nothing selected or refitted except the metadata baselines
(fitted on train). `PREREGISTRATION.md` lists every test-split and pro-set analysis and was written
before the per-encounter test file and `scout_v1` were touched; Addendum 1 was written after the first
results and before the lobby analysis.

| Script | What | Splits used | Output |
|---|---|---|---|
| `00_gates.py` | Reproduce headline, validation blend, per-encounter file | val, cal, test | console |
| `01_cs2cd_hypotheses.py` | Source, within-match, weapon post-stratification, strict FPs, thresholds, metadata baselines | val, cal first; test once | `records/01_*.txt`, `.json`, `scored_players.csv` |
| `02_score_pro.py` | CPU scoring of all staged pro windows, parity check | pro sets | `records/pro_players.csv`, `02_score_pro.log` |
| `03_events_lobbies.py` | Scout-lobby split, threshold sensitivity (cal to val only), label-noise bounds, per-encounter attribution | val, cal; test once | `records/03_*.txt`, `.json` |
| `04_pro.py` | Pro flag rates, person effects, re-identification, channel shift | pro sets | `records/04_pro.txt`, `.json`, `pro_channel_shift.csv` |
| `05_match_index_clustering.py` | Similarity of neighbouring match numbers | labels and weapons only | `records/05_*.txt`, `.json` |

Results are in sections 1, 3 and 4. Numbers not repeated there:

- Metadata baselines (logistic regression, fitted on train): weapon shares 0.880 / 0.846 (validation /
  calibration); adding log encounter count does not help (0.868 / 0.841).
- Detector vs metadata on validation + calibration: 0.942 vs 0.858; Spearman correlation 0.69.
- Components on test: LightGBM 0.9677, TCN alone 0.9607, blend 0.9727. The neural model alone is below
  the tree.
- Strict recall by cheaters in the match (test): 0 of 6 with one cheater, 12% with two or three (n = 24),
  35% with four or more (n = 184).

**Test-split uses added today** (all pre-declared, descriptive, nothing chosen): T1 to T8 and A1 on player
scores; T4 and T5 on per-encounter scores. **Pro sets:** `awp_v1` second use; `scout_v1` first and only
use. Neither can be called fresh any more.

### 5.1 Kaggle spec written (not run)

`kaggle/cs2cd_followups.py` and `kaggle/KAGGLE_SPEC.md`: K1 weapon-normalised scoring, K2 blocked-split
leakage bound, K3 footstep-channel masking, K4 weapon-neutral reweighted LightGBM. It rebuilds the window
cache by streaming the raw data, checks every window hash against the frozen experiment, never scores
test rows, and was smoke-tested locally on stand-in data. Extraction runtime and Kaggle disk behaviour
are unverified.

### 5.2 Local demos without ground truth

Twelve scored runs in `local_runs/` [V]: nobody is flagged at `accuracy` except one player in the
documented pro match (0.42); the two "cheater demo" runs peak at 0.255 and 0.063. No file records who,
if anyone, was cheating. Decision: they are pipeline smoke tests and unlabelled examples. They are not
evidence about recall, and 38 presumed-legitimate matchmaking player-matches are too few and too
unverified to count toward a false-positive rate. If you know which account was cheating in the two
FACEIT demos and how you know, write it down with the date; otherwise the folder name `cheaters_demos`
should not be read as a label.

---

## 6. The five decisions

The prompt says four and lists five; all five were asked and answered on 2026-11-16.
Constraints you gave: deadline more than six months away; 5 to 8 friends can record.

| # | Decision | You chose | Why (evidence) |
|---|---|---|---|
| 1 | Friend data | **Split by person**, fixed before the first session | Person identity explains about 10% of pro score variance after weapon; never splitting a person removes the problem. Friends cannot give a tight FPR (0 of 8 people bounds it only below 31%), so their value is the paired design and the real server pipeline |
| 2 | Scout imbalance | **Report now; Kaggle spec written; retraining optional** until you have read this report | Section 4.2 and conclusion 2. Step 1 costs nothing and is needed either way |
| 3 | OpenCS2 pro data | **Split by player, hold-out first** | 0 of 3,260 flags at 0.5; only one scout-main player-map, so no scout fix; `victim_footstep` is always zero in pro windows, a ready-made "pro = legitimate" shortcut if trained on as-is |
| 4 | Experts vs one model | **One model with weapon-normalised scoring**; experts stay possible with more data | Encounter AUC is 0.71 to 0.84 inside every weapon group already; the problem is label-weapon correlation, which an expert inherits; 78 AWP-main cheaters in training |
| 5 | Self-made cheating data | **Public data first, then a logged server-side assist** | Only way to get cheat-type and setting labels; no third-party cheat software, so no account, malware or ToS risk; needs supervisor and ethics approval |

**How each is implemented**

1. *Friends.* With 5 to 8 people: order people by a seeded shuffle of sign-up IDs before any recording;
   the last 2 (of 5 or 6) or 3 (of 7 or 8) are the hold-out, scored once at the end. All of a person's
   sessions go to their pool. Every person plays both conditions (clean, and assisted once decision 5 is
   approved), so each is their own control. Report people flagged at least once with exact intervals,
   and paired differences; do not report rates as if matches were independent. The existing
   `FRIEND_SESSIONS.md` (12 people, evaluation only) needs a new version saying this; I did not edit it.
2. *Scout.* Reporting changes are listed in roadmap item 2. The spec is K1 and K4.
3. *Pros.* `records/pro_person_split.csv`: hashed player ID, pool by parity of the hash. Development 585
   people / 1,713 player-maps; hold-out 547 / 1,547 (AWP-main 196 / 209). The hold-out is a regression
   check for every future model (zero flags at 0.5); the development half may enter training only with
   the footstep channel masked for all sources and a source-shortcut check.
4. *Experts.* Revisit when a weapon group has roughly 300 cheater player-matches with trustworthy labels
   and as many clean ones, and an expert beats the shared model on weapon-balanced validation AUC by
   more than the bootstrap interval. Today no group outside the scout comes close.
5. *Cheat data.* Design only; nothing was built. I will not help obtain or configure cheat software.

---

## 7. Enhancement roadmap

Ranked by expected effect on (a) detection performance and (b) deployability, per unit of effort.
Repository findings are tagged [V] or [I] above; everything from outside is tagged [E] here.

**1. Score the frozen model on an independent labelled CS2 cohort (XGuardian).**
- Problem: unrepresentative positives, reused test set, no independent labels.
- [E] 2,903 matches from the 5EPlay platform, 5,486 players, 350 cheaters, labels manually re-verified,
  demo-based, raw demos released; reported FPR 4.1 to 10.1% and recall about 0.89 for their own model
  ([XGuardian, arXiv 2601.18068](https://arxiv.org/html/2601.18068v1)). Artifacts: one 22 GB archive
  ([Zenodo 17845614](https://zenodo.org/doi/10.5281/zenodo.17845614), GPL-3.0-or-later) and 13 raw
  parts, the first 48.3 GB ([Zenodo 17838584](https://zenodo.org/doi/10.5281/zenodo.17838584)).
- Takes: your approval for the download; reading the archive README first (identity format, tick rate);
  about 44 s per demo with the existing extractor; a pre-registration (frozen thresholds, primary metric
  AUC and recall at `accuracy` and strict, intervals by match).
- Could go wrong: the raw archive is one split stream, so a subset may not unpack alone; demos may not
  be 64 Hz or may lack properties the extractor needs; identities may not map to the cheater list; the
  footstep channel will probably be missing; negatives are noisy there too.
- Worked if: an AUC with a match-cluster interval on at least 100 cheater player-matches. Around 0.85 or
  above supports generalisation beyond CS2CD; around 0.6 to 0.7 says the CS2CD result is specific to its
  lobbies. Either result is a thesis finding.

**2. Re-base the reported evaluation and the operating point (no training).**
- Problem: the strict policy's numbers describe scout lobbies; thresholds below 0.5% FPR are unsupported.
- Takes: days. Primary table = all / outside lobbies / weapon-balanced / lone-cheater rows with
  match-cluster intervals; a baseline ladder (weapon shares, LightGBM, blend); drop `fpr_0_1pct`;
  state the threshold's resampling range; contribution statement; update the stale README and
  `ARCHITECTURE_CHOICES.md`. For a certified operating point use clean negatives: with the 1,713
  development pro player-maps, a threshold with zero flags bounds pro FPR below 0.18%. [E] Distribution-
  free threshold selection with finite-sample guarantees is an established method
  ([Learn then Test, arXiv 2110.01052](https://arxiv.org/abs/2110.01052)).
- Could go wrong: nothing technical; the headline gets less impressive.
- Worked if: every number in the thesis survives the question "is that the lobbies?".

**3. Logged server-side aim assist and a paired friend study (decisions 1 and 5).**
- Problem: no labels for subtle cheats, weapons or settings; no independent data through the real-time
  path.
- Takes: supervisor and ethics approval, consent from each participant; plugin work for an assist mode
  with exact logs (when active, strength, field of view); 5 to 8 people, clean and assisted sessions at a
  few settings and weapons. [E] Comparable studies had ethics approval, informed consent, minimal data
  and controlled cheat execution through a plug-in (summarised from a search; primary papers not read).
- Could go wrong: a simulated assist is not a commercial cheat; assisted play by non-cheaters is not how
  cheaters behave; 5 to 8 people give dose-response curves, not rates.
- Worked if: detection probability rises monotonically with assist strength, and you can state the
  weakest setting detected at a threshold that flags none of the same people when clean.

**4. Hit-consistency evidence for silent aim and rage cheats.**
- Problem: silent aim leaves the view angles untouched, so the present aim channels cannot see it [I];
  it is one of your listed unsolved problems.
- Idea: for each bullet that hits, the angle between where the player was looking (with recoil) and the
  victim at the fire tick. Legitimate hits must lie within spread plus the target's angular size. The
  server knows both. This is a separate, explainable evidence stream scored after the shot, so it needs
  its own protocol: the current invariant (windows end before the first hit) stays for the existing
  model.
- Takes: a feature script on CS2CD raw data (Kaggle) and the same quantity in the plugin; pros as the
  clean reference.
- Could go wrong: lag compensation and the one-tick stamp offset the journal found make server-side
  geometry inexact; hitboxes are approximated; wall penetration.
- Worked if: the pro distribution has a tight tail and lobby cheaters in CS2CD sit far outside it. That
  would also test the hack-versus-hack reading of the lobbies.

**5. Reweighting, weapon-normalised scoring and cleaned negatives (K1, K4, then optional neural retraining).**
- Problem: section 4.2.
- Takes: one Kaggle run you launch; a new test cohort (item 1 or the friend hold-out) for any claim.
- [E] Retraining only the last layer on group-balanced data is reported to match heavier methods for
  spurious correlations ([Kirichenko et al., arXiv 2204.02937](https://arxiv.org/abs/2204.02937)), which
  is the cheap neural variant if K4 helps.
- Could go wrong: all-player AUC falls; with noisy scout negatives the weights amplify label errors,
  which is why the suspect lobbies are removed first.
- Worked if: weapon-balanced and outside-lobby validation AUC rise and the pro hold-out stays at zero
  flags at 0.5.

**6. Bunny-hop detection as a small rule-based module.**
- Problem: an original goal, never started.
- Idea: per jump, ticks between landing and the next jump; streaks of perfect hops and speed gain. A
  statistical test against human timing, from plugin data, no learned model at first.
- [E] I found no peer-reviewed work on this; public sources are community server plugins that compare
  air speed to ground speed ([example](https://forums.alliedmods.net/showthread.php?p=1260095)).
- Could go wrong: CS2 movement and input handling differ from older titles; legitimate scroll-wheel
  jumping produces good timing too; labels would come only from your own sessions.
- Worked if: scripted hopping on your server is separated from the same people hopping manually.

**Considered and rejected**

- *Image or video based detection.* Needs rendered player views, heavy compute, and is not server-side.
- *Per-weapon experts now.* Decision 4.
- *Training on all pro data.* Cannot fix the scout; shortcut risk (section 4.7).
- *Positive-unlabelled or noise-robust training as the main fix.* The suspect negatives are concentrated
  in identifiable matches, not spread at random, so removing those matches is simpler and does not need
  an assumption about how labelled positives were selected [I]; survey:
  [Bekker and Davis, arXiv 1811.04820](https://arxiv.org/abs/1811.04820).
- *Later-ban labels on matchmaking demos at scale.* [E] The VAC flag alone had 24.7% recall in the
  publisher's check, so negatives stay noisy; plus other players' IDs are personal data.
- *A bigger or different architecture.* Seven architectures are within 0.04 AUC, the blend is not shown
  to beat LightGBM, and weapon shares alone reach 0.85 to 0.88. Capacity is not the bottleneck; labels
  and population are.
- *Wallhack detection from the current windows.* Needs line of sight and map geometry. [E] One published
  system uses "elimination through occluder" features and still reports FPR above 25% at default
  settings ([Hawk, arXiv 2409.14830](https://arxiv.org/html/2409.14830v2)).
- *Masked pretraining on gameplay.* Plausible later; not before the evaluation is fixed.

**Ordered plan**

1. Now: roadmap item 2 (reporting), contribution statement, fix the two failing tests, live flag only
   after about 10 encounters (existing proposal D6).
2. Launch the Kaggle script (K1 to K4); bring back `results.json`; decide on retraining.
3. Read the XGuardian archive README, approve the download, pre-register, score.
4. In parallel: supervisor and ethics approval; rewrite the friend protocol for 5 to 8 people and the
   person split; fix the listener defects listed in the journal before any session.
5. Plugin: hit-consistency and jump logging, then the assist mode; pilot with one person.
6. Friend sessions, paired; evaluate on the development pool; hold-out once at the end.
7. Only then: optional neural retraining with weights, final numbers on the cohorts that are still fresh.

---

## 8. What would earn a "validated live" claim

None of this exists yet.

- Data collected independently of training, through the real-time path (plugin capture and live
  scoring), on a later date and game version.
- Verified labels for both classes: clean players known to be clean, and positives with known ground
  truth (controlled assist, or platform bans confirmed by review).
- Ordinary matches: one or two cheaters among otherwise legitimate players.
- Thresholds fixed in advance, results reported once.
- People as the unit. Showing FPR below 1% needs about 300 distinct clean people with no flag; 5 to 8
  friends cannot do that, whatever the number of matches. A recall estimate within plus or minus 10
  points needs about 100 positive player-matches.

Until then the accurate description is: an offline benchmark result on CS2CD, a clean bill on
professional demos at 0.5 and above, and a working real-time pipeline tested against bots.

---

## 9. Not checked, and still uncertain

- **That scout lobbies are hack-versus-hack matches is an inference** from weapons, cheater counts and
  match numbering. The demos are not local, so I could not look.
- **Same-person leakage is not measured.** Match-number clustering is indirect evidence; K2 would bound it.
- **The Kaggle script has not run on CS2CD.** Only the smoke test passed; runtime, disk use and the two
  integrity gates are untested there. The hash-chain note (survey G11) may matter if file hashes differ
  on Linux; the script does not depend on those hashes.
- **Footstep-channel effect on pro scores**: untested (K3).
- **XGuardian**: I read the paper's HTML and two Zenodo pages, not the archive. Tick rate (the paper's
  "7 ticks = 109.375 ms" implies 64 Hz), identity format and whether a subset can be unpacked are
  unknown.
- **Literature**: claims about VACNet, risk-control details and the ethics of cheat-data studies come
  from search summaries; I cite only pages I fetched, and for the survey and the two method papers only
  their abstracts.
- **Not read**: the plugin source, `realtime/` code, `src/`, notebooks, PDFs, `D:` drive contents.
  Real-time parity, latency and capacity numbers are the journal's.
- **Publisher's 97.2% figure**: taken from the paper; whether it is per player or per match is not
  stated in what I read.
- **Small cells**: lone-cheater recall (6 to 9 players per split), AWP-main and Deagle-main groups, and
  everything about 22 strict false positives.
