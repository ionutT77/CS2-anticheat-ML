# Pre-declared analyses (written 2026-11-16, before the test event file or scout_v1 were analysed)

Code revision `fb5f39b` + this folder. Frozen model, Platt calibrator and thresholds from
`experiments/cs2cd_v1/calibration.json`; nothing is refitted or selected. Every analysis below is run
on validation and calibration first; the CS2CD test split (already used for the headline result and
the weapon breakdowns) is used once per item as a descriptive replication. No threshold, feature,
weight or model choice is made from test or from the pro sets.

Bootstrap: 2,000 resamples of whole match groups, seed 20261116. Proportions: exact 95% intervals.
Weapon groups: `scripts/analyze_weapon_groups.py` `GROUP`; "main group" = group with most encounters
(the definition used in `OTHER_WEAPONS.md`). No new definition is introduced.

## CS2CD test split (player-level scores)
- T1 Source: among label-0 players, AUC of the score for the source folder; FPR per folder at each frozen threshold.
- T2 Within-match ranking: pooled concordance of cheater-labelled vs non-cheater players inside the same match.
- T3 Weapon post-stratification: AUC, strict recall and strict FPR after weighting players so the
  main-weapon-group distribution is the same for both labels; AUC within each main group.
- T6 Description of the strict false positives and their matches.
- T7 Threshold uncertainty: match-cluster intervals for FPR and recall at each frozen threshold; spread of
  test FPR/recall when the strict threshold is re-derived on bootstrap resamples of the calibration split
  (the frozen threshold is not changed).
- T8 Score distribution of positive labels; score vs encounter count; score vs cheaters per match.

## CS2CD test split (per-encounter TCN scores, `tcn39_seed123_test_events.npz`, test only)
- T4 Event-drop attribution, TCN component only (75% of the blend): TCN player score = mean event logit,
  recomputed without scout encounters and without any sniper encounters, on the fixed set of players who
  keep at least one encounter. Reported as AUC only (blend thresholds are not applied). This is
  inference-time attribution, not a model trained without the shortcut.
- T5 Encounter-level AUC of the TCN within each weapon group (clustered by match).

## OpenCS2 professional demos (all label 0 by assumption)
- P1 Parity: re-score `awp_v1` on this machine and reproduce `local_runs/opencs2_pro_fpr_scores.csv`.
- P2 `scout_v1` first and only evaluation use: frozen model and thresholds, flag rates for the addition
  and the reserve, intervals by person (bootstrap over hashed player identity) and by player-map.
- P3 Person effects (both sets pooled): share of score variance between people (ICC), and whether a
  player can be re-identified across maps from window features (mechanism check for same-person
  leakage; says nothing about how often the same person recurs in CS2CD).

Identities are hashed (salted SHA-256, 8 hex chars) before anything is written.
