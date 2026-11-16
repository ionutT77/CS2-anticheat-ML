# Kaggle spec: CS2CD follow-ups K1 to K4

Script: `cs2cd_followups.py` (usage in its header). Smoke-tested locally on stand-in data
(`--smoke`), then **run on Kaggle on 2026-11-16** (CPU, about 30 minutes, revision `e3c1e46`); results in
`../records/kaggle_followups/` and in section 5.1 of `research/notes/thesis-analysis-report.md`.

| | |
|---|---|
| Data | CS2CD pinned revision in `data/cs2cd/metadata/raw_repo.json` (`44e5129…`), 795 matches. Raw files are downloaded, extracted with the unchanged `anticheat/cs2_data.py::extract_match` and deleted a few matches at a time, so peak disk is about 2 GB of raw data plus the outputs (about 6 GB extracted windows, 6 GB cache). |
| Integrity gates | Every extracted window must match its SHA-256 in the frozen `events.parquet`; the rebuilt cache must reproduce the saved `tcn39_seed123` validation scores within 5e-3. The script stops if either fails. |
| Split | Frozen `cs2cd_v1` match split. **Test rows are never scored.** K2 uses only the original train + validation matches. |
| Seed | 20261116 (K2 fold draws); LightGBM `random_state=42`, same config as `lgbm39_leaves15`. |
| Models | Frozen `tcn39_seed123.pt` and `lgbm39_leaves15.joblib` (K1, K3); LightGBM refits (K2, K4). No neural training. |
| Outputs | `/kaggle/working/followups/results.json`, `results.txt`; cache in `/kaggle/working/cs2cd_cache`. |
| Runtime | Measured on a Kaggle CPU session: extraction 16 min, K1 + K3 8 min, K2 4 min, K4 2 min. |

Metrics per experiment, on validation and (second look) calibration: AUC on all players, on reviewed
matches, outside scout lobbies, weapon-balanced; in-sample recall at 1% FPR outside lobbies.

| | Question | What it does | Confirms | Refutes |
|---|---|---|---|---|
| K1 | Does removing the weapon prior from encounter scores help? | Subtracts each weapon group's median legitimate training logit before averaging per player. No retraining. | Weapon-balanced and outside-lobby AUC up by more than about 0.01 with all-player AUC not down by more than 0.01 | No gain: the weapon effect is inside the encoder, so only reweighted retraining can address it |
| K2 | Is the random match split optimistic because neighbouring matches share people? | Five random 80/20 match splits vs five contiguous-block splits (5-number buffer), same LightGBM | Blocked AUC lower than random by more than the fold spread: that gap bounds what adjacency contributes | Equal AUCs: leakage through adjacency is small for this model |
| K3 | Does the footstep channel, missing in GOTV pro demos, change scores? | Zeroes `victim_footstep`, re-scores validation + calibration | Scores and AUC nearly unchanged: the pro false-positive result is not an artefact of the missing channel | Legitimate scores drop: pro FPRs are optimistic and all cross-source comparisons need channel masking |
| K4 | Does weapon-neutral weighting plus removing suspect lobbies give a better tabular model? | Three LightGBM fits: baseline; unreviewed lobbies removed; removed + weights `P(y)/P(y \| main weapon)` clipped to [0.1, 10] | Weapon-balanced and outside-lobby AUC up on validation and calibration | No gain: go to neural retraining with weights, or accept the limitation |

Decision rule agreed with the author: reporting changes go ahead regardless; neural retraining is
decided after reading these results. Any retrained model needs a new test cohort (the frozen test
split has been used repeatedly).
