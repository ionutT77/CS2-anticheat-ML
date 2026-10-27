# Architecture choices, implementation status and further considerations

**Scope:** the CS2 anti-cheat detector as built in `research/` (CS2CD benchmark model, the
live pipeline on a private server, and the data work around them). Numbers come from
`research/experiments/cs2cd_v1/` and `PROJECT_JOURNAL.md`; file paths are given so each
claim can be checked.

## 1. Problem and decision unit

| Choice | Decision | Reason |
|---|---|---|
| Task | Rank players by how likely their aim and shooting behaviour is cheating | Only match-level labels exist; no cheat-activation intervals |
| Decision unit | **One player in one match** (player-match) | Labels are player-match labels; a single shot is too noisy to judge |
| Unit of evidence | **Encounter windows**: the 256 ticks (4 s at 64 Hz) before an attacker's first hit on a victim | Behaviour before the shot is what shows aim quality; nothing after the hit is used |
| Output | A calibrated score and review flags | A review signal, not a verdict; no automatic action |

## 2. Input

- **39 channels per tick** (`research/examples/input_schema.json`): aim changes and target
  error, distance and movement, player and weapon state, shooting and recoil, encounter
  context (victim fire, footsteps, stance, health).
- **Why not only the 5 original channels:** the 5-channel LSTM reached validation AUC 0.911,
  against 0.925 for the same LSTM on 39 channels.
- **Leakage rules:** every feature must exist before the prediction time; the victim is
  identified from the hit but nothing after it enters the window; player IDs, map, rank and
  absolute coordinates are excluded.
- **Normalisation:** fixed physical scales plus an asinh compression, then z-scoring with
  statistics fitted on the **training split only**.

## 3. Model

**Two models, blended 75% / 25%:**

| Component | What it is | Why it was chosen |
|---|---|---|
| **Temporal CNN** (`tcn39_seed123`, 160,641 parameters), weight 0.75 | Dilated residual 1-D convolutions (dilations 1, 2, 4) over the 256 ticks, with mean and max pooling, followed by an event head | Best single validation score. Rationale (not tested in isolation): aim snaps and recoil control are local patterns in time, which convolutions capture with few parameters |
| **LightGBM** (`lgbm39_leaves15`), weight 0.25 | Gradient-boosted trees on fixed summaries of each encounter, pooled per player | A second model family: the blend scored higher on validation than either model alone (0.946 against 0.944 and 0.938). It also trains on a CPU |

**From encounters to a player score:**
- Neural model: the average of the encounter logits over a bag of up to **16 sampled
  encounters** in training, and **all eligible encounters** at evaluation. An optional
  encounter-level auxiliary loss (weight 0.25) is part of the training protocol.
- Trees: per-encounter summaries, pooled to the player by mean, standard deviation and the
  90th percentile.

**Training:** AdamW (lr 5e-4, weight decay 1e-3), up to 40 epochs with early stopping
(patience 8), two seeds (42 and 123), unweighted player-match loss, natural sampling (no
class or weapon weights).

**Candidates compared on validation** (mean of AUC on all players and on manually reviewed
matches):

| Model | Validation score |
|---|---|
| LSTM, 5 channels | 0.908 |
| LSTM, 39 channels | 0.929 |
| Patch Transformer | 0.931 |
| Hierarchical CNN + Transformer | 0.932 |
| LightGBM (leaves 15) | 0.938 |
| Temporal CNN (seed 42 / **seed 123**) | 0.940 / **0.944** |
| **CNN 0.75 + LightGBM 0.25** | **0.946** |

The differences between the top models are small. On the test set, the blend's AUC
advantage over the best tree is within uncertainty (difference 0.0049, paired 95% interval
−0.0015 to 0.0124; `METHODS.md`). The blend was selected on validation by the rule written
in advance in `protocol.json`, not by test results.

## 4. Calibration, thresholds and evaluation

- **Splits** (60/15/10/15%, by match-connected group): training, validation, calibration,
  test. No match appears in two splits; duplicates are grouped before splitting.
- **Calibration:** a Platt sigmoid with positive slope, fitted on the separate calibration
  split. Thresholds are also chosen there, never on the test set.
- **Operating points:** `accuracy` (0.257), `f1` (0.211), `default_0_5`, strict `fpr_1pct`
  (0.853) and `fpr_0_1pct` (0.889).
- **Test result** (1,130 player-matches, 214 positive labels): ROC-AUC **0.973** (95%
  interval 0.956–0.985, bootstrap by match). At the strict threshold: 67 true positives and
  4 false positives (recall 31%, FPR 0.44%).

**What ROC-AUC means.** Pick one player labelled as a cheater and one labelled as legitimate
at random. ROC-AUC is the chance that the model gives the cheater the higher suspicion score.
A value of 0.5 means guessing; 1.0 means every cheater is ranked above every legitimate
player. Our 0.973 means that in about 97 of 100 such pairs the cheater is ranked higher.
It does not say which score counts as "flagged": a model can have a high ROC-AUC and still
flag many innocent players at a given threshold. This is why precision, recall and
false-positive rate are reported next to it.

## 5. What has been implemented

| Area | Status | Where |
|---|---|---|
| CS2CD extraction, features, training, calibration, inference | Done, frozen | `research/anticheat/`, `scripts/` |
| Artifact integrity checks | Done | `scripts/verify_artifacts.py` |
| Demo (`.dem`) extractor for private matches | Done, tested on real demos | `anticheat/dem_extractor.py`, `scripts/score_private_match.py` |
| Server telemetry plugin (CounterStrikeSharp) | Done; recording at 64 Hz | `research/realtime/plugin/` |
| Capture checks (leakage, velocity, tick alignment, plugin vs GOTV) | Done; plugin matches the demo tick for tick | `realtime/check_capture.py`, `demo_parity.py` |
| Window builder from live captures | Done; causality and determinism tests | `realtime/capture_windows.py` |
| Live scoring on a private server | Working; about 0.4 s per hit | `realtime/live_scoring.py` |
| Capacity guide | Done: up to 10 players supported | `realtime/server/CAPACITY.md` |
| OpenCS2 data addition (legitimate pro players) | Done, **staged, not trained**: 2,356 encounters in the training addition, 25,632 in the reserve | `research/opencs2/`, `scripts/add_opencs2_awp.py` |
| Friend-session evaluation protocol | Written and reviewed, not yet run | `realtime/server/FRIEND_SESSIONS.md` |
| Tests | 248 pass; 2 failures in `test_private_scoring.py` (`KeyError: 'steamid'` in `scripts/score_private_match.py`), not caused by the real-time work and not yet fixed | `research/tests/` |

## 6. Findings that qualify the design

- **Sniper weapons.** Legitimate players who mostly use snipers are flagged more often. In
  the CS2CD test split all 4 strict-threshold false positives were scout-heavy players
  (4 of 25). Legitimate pro AWPers are not flagged at the strict threshold (0 of 100) but are
  at the lenient ones (13–21%).
- **The scout is the imbalance, not the AWP.** CS2CD training has 5,933 cheating against
  2,925 not-cheating scout encounters; the AWP has more not-cheating than cheating.
- **Possible shortcut.** The model's strict-threshold detections are concentrated in
  scout-heavy players, so part of the headline result may come from weapon style. A
  source-only baseline already reaches AUC 0.888 overall, though 0.500 within reviewed matches.
- **Source differences.** GOTV demos lack networked velocity, flash and footstep events;
  the live server plugin has them. Models trained on one source may not transfer to another.
- **Identity.** CS2CD remaps player identities per match, so separation of real people
  across splits cannot be verified.

## 7. Further considerations

**Planned, needing approval (nothing here has been started):**

1. **Scout data.** About 2,700 more not-cheating scout encounters are needed to balance
   training. Pro demos can supply about 840 (best 100 demos, 46 GB) up to about 1,950
   (all, 228 GB); friend sessions at most about 1,400. See `research/opencs2/SCOUT_AWP_BALANCE.md`.
2. **Balancing.** Reweighting the not-cheating scout players, without deleting data, then
   comparing weapon-group false-positive rates on validation. Expect lower strict recall.
3. **Friend sessions.** At least 12 different people, each playing normal, scout-only and
   AWP-only matches, scored offline and blind; an evaluation group, not training data.
4. **Retraining and recalibration.** Needed after any data change: new thresholds, a new
   independent test group, and a CUDA GPU for the neural model (LightGBM runs on a CPU).
5. **Further model ideas.** User-command features (mouse and view-angle fields in CS2CD,
   never tested as inputs) for silent aim; line-of-sight features for wallhacks; masked
   pretraining on gameplay sequences.

**Limits to state in the thesis:**

- Labels are per player-match, and negative labels may include unreviewed cheaters.
- The test set has been used; a new independent cohort is needed for any new claim.
- Live results so far use one legitimate player and bots. They show the pipeline works, not
  that it is accurate. No claim of validated live anti-cheat performance can be made.
- Precision depends on how common cheating actually is: at 1% prevalence the strict
  policy's measured rates imply about 42% precision.
- Cheating data beyond CS2CD, and cheat-type labels (wallhack, silent aim), are not available.
