# Methods and interpretation

## Three evaluations

Ionuț’s journal, our Kaggle experiments and the CS2 experiment have different evaluation units. Comparisons below preserve those units. File hashes identify the code and artifacts distributed with this repository; the saved weights, splits and numerical predictions are unchanged from the recorded experiments.

| Evaluation | Input and decision | Separation | Test size |
| :--- | :--- | :--- | ---: |
| Original journal | One 192-tick engagement, 5 channels | Publisher records randomly split | 54,000 engagements |
| Audited Kaggle | One engagement or a 30-engagement record | Groups of duplicate and overlapping trajectories | 2,038 records |
| CS2CD | Eligible 256-tick encounters for one player in one match | Match groups | 1,130 player-matches |

ROC-AUC measures how often a randomly chosen positive-label example ranks above a negative-label example; tied scores count as half. A constant score has AUC 0.5. Accuracy measures correct decisions at one threshold. Precision is the positive-label share of flags. Recall is the share of positive labels found. False-positive rate is the share of negative labels flagged.

## Review of the original LSTM

Reviewed commit: `06427ceabd9990a8bc278b21758e1b04aa9d46f7`.

The architecture is appropriate for a first sequence model. The implementation normalizes using training data and keeps each publisher record's engagements together. The journal makes the training choices visible. No trained checkpoint or executed final prediction arrays were published, so the journal's scores cannot be replayed exactly.

The journal's best accuracy run reports 59.1% accuracy, 24.9% precision, 72.3% recall and 0.715 AUC. Always predicting non-cheater reaches 83.3% accuracy on that class mixture. The LSTM's ranking contains a useful signal; its default decisions create many false flags. The implied false-positive rate is approximately 43.5%, calculated from rounded journal metrics.

The latest recipe combines balanced sampling with a positive loss weight of 2, then applies a 0.5 threshold. This shifts decisions toward positive predictions. Calibration on data with the natural class mixture is needed. Repeated use of test results during development also weakens their independence.

Reconstructing the published split finds 699 of 1,800 test records connected to training through overlapping trajectories, including 25 exact full-record duplicates. This does not establish how much the reported AUC was inflated. It does establish that splitting publisher records alone is insufficient. Counts are in `reports/reference_comparison.json`.

## Kaggle audit and LSTM repair

The release has 12,000 records, each shaped `[30, 192, 5]`. The 192 ticks cover five seconds before a kill and one second after it at 32 Hz. This is a retrospective archive. Stable real-player and match identifiers are absent.

The audit groups exact records and shared active 64-tick trajectories, including shifted matches. Thirty records with contradictory labels are excluded. Groups remain within one split: 7,616 training, 1,189 validation, 1,127 calibration and 2,038 test records. These checks cannot prove separation of real people.

The repair retains the original 120,897-parameter architecture: LSTM 128, dropout, LSTM 64, final hidden state, dense 32 and a scalar output. It uses natural sampling, unweighted binary cross-entropy, asinh compression and normalization fitted on training data. A mean of engagement logits is selected on validation data; a separate calibration split fits a sigmoid and decision thresholds.

| Same audited split | Engagement AUC | Record AUC |
| :--- | ---: | ---: |
| Retrained original recipe | 0.7417 | 0.8351 |
| Revised training, same LSTM | 0.7499 | 0.8432 |
| Selected CNN pair | Not the primary evaluation unit | 0.8586 |

The LSTM's record AUC gain is 0.0081 with a paired 95% interval of -0.0044 to 0.0208. The CNN's advantage over the revised LSTM is also uncertain. Threshold calibration has a much larger effect on accuracy: for the reference LSTM, engagement accuracy changes from 53.2% at 0.5 to 86.0% at the separately calibrated accuracy threshold, without changing the ranking.

The revised LSTM has 88.5% record accuracy and the CNN pair 89.3%, compared with the 84.5% constant baseline. The CNN makes 218 errors versus 315 baseline errors. This supplement uses a previously examined test set; treat it as a controlled diagnostic that needs fresh confirmation.

## CS2 extraction

The pinned CS2CD release contains 795 matches and 52.6 GB of raw JSON and Parquet files. All 1,590 files were downloaded and checked against their Git/LFS digests. Extraction processed 734,975,247 player-tick rows into 141,633 encounters.

Each window contains 256 consecutive ticks at 64 Hz, ending strictly before the first bullet damage in an attacker-victim burst. A gap longer than 128 ticks starts a new burst. Both players must be alive; warmup, team damage, round crossings, missing ticks and nonfinite input are excluded. Nonfatal damage encounters are included. Misses before the first hit can appear in the history. Encounters with no damage are not scored.

The victim is selected retrospectively using the damage event. A score can therefore be produced after that encounter. Features exclude the damage tick and all future ticks. Target direction uses an approximate head height, not an exact hitbox. Noise events do not prove audibility, and flash duration approximates visual impairment. Exact line of sight, smoke geometry and visibility are unavailable in this pipeline.

| Feature group | Examples |
| :--- | :--- |
| Aim and target tracking | Yaw/pitch changes, target error, acceleration, target angular speed |
| Geometry and motion | Distance, relative height, movement speed, closing and lateral speed |
| Player and weapon state | Scope, crouch, airborne state, health, armor, weapon category |
| Shooting and recoil | Shots in a burst, ammunition, punch angles, recoil index, time since shot |
| Encounter context | Victim fire and footsteps, time since those events, victim stance and health |

The exact 39-channel order is in `examples/input_schema.json`. Inference accepts raw features, then applies the saved transformation and normalization. Player IDs, source folder, map, rank, absolute coordinates, absolute server tick and future outcomes are excluded from predictive features. Metadata appears in output tables for auditing only. User-command mouse fields exist in the release but were not tested as model inputs.

## CS2 selection and test

Matches are split into 477 training, 119 validation, 79 calibration and 120 test matches. One validation match has no eligible encounter. Exact match/event signatures and 32-event subsequences define groups before fitting; exact extracted windows are also checked. No exact duplicate window crosses splits.

Seven neural and three tree candidates are trained. The neural models use player-match loss, sampled bags of up to 16 encounters, an auxiliary engagement loss, AdamW and early stopping. Evaluation combines all eligible encounters. Normalization is fitted on training ticks only.

Selection averages validation AUC on all players and on players from manually reviewed matches. The winner combines 75% temporal CNN (`tcn39_seed123`) and 25% LightGBM (`lgbm39_leaves15`). Positive-slope Platt calibration and thresholds are fitted on a separate 743-player calibration set. Test evaluation follows selection and calibration. Confidence intervals resample complete match groups 2,000 times.

The reserved test has 214 positive and 916 negative player-match labels. Selected AUC is 0.9727 (95% interval 0.9560 to 0.9854), average precision 0.8829 and accuracy 93.0%. A source-only diagnostic reaches 0.888 AUC overall but 0.500 within reviewed matches. The selected model reaches 0.971 within that reviewed subset, supporting a signal beyond source membership. Collection artifacts may still remain.

At the accuracy threshold there are 184 true positives, 49 false positives and 30 false negatives. At the threshold targeting 1% calibration FPR there are 67 true positives, 4 false positives and 147 false negatives. Labels remain unchanged in the supplied review queue. The model's AUC advantage over the best tree is 0.0049, with a paired interval of -0.0015 to 0.0124.

Using at most the first 1, 5 or 10 encounters gives AUC 0.834, 0.946 or 0.962 respectively; the full-match result is 0.973. This measures eligible encounters, not elapsed detection time. Models and full-match thresholds were reused without retuning.

## What remains uncertain

CS2CD remaps identities separately in each match. Separation of real people across splits cannot be verified. Labels describe the player-match, with no cheat activation intervals or reliable cheat-family labels. The negative-only collection is unreviewed; the reviewed collection is not guaranteed error-free.

The pipeline covers 7,508 of 7,949 player-matches (94.5%), including 98.9% of positive labels and 93.6% of negative labels. No claim applies to players without an eligible damage encounter. The current result does not establish performance on new recording sources, future game versions or live play.

AUC 0.973 does not mean 97.3% accuracy or 99% reliable flags. Under an illustrative 1% cheating prevalence, the strict policy's measured recall and FPR would imply about 42% precision, assuming those rates transfer unchanged. A deployment study needs verified labels, a specified observation period and enough independent legitimate examples to measure rare false positives.

Masked pretraining on gameplay sequences and input-command features are reasonable future ablations. The patch Transformer trained here reaches AUC 0.946. Text LLM fine-tuning has not been tested; an LLM could help write evidence summaries for reviewers, with every claim tied to telemetry.
