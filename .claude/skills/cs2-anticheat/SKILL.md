---
name: cs2-anticheat
description: Use when changing CS2 cheat-detection features, encounter windows, models, calibration, thresholds, or evaluation logic.
---

When changing anticheat logic:

1. Preserve encounter-window semantics:
   - Inputs cover four seconds before the first bullet hit.
   - The victim is identified retrospectively from that hit.
   - Features must not require future events.
2. Respect label semantics:
   - Labels are player-match level.
   - A positive label does not mean every encounter is cheating.
   - Negative labels may include unreviewed players.
3. Preserve evaluation integrity:
   - Hold out complete matches in CS2CD.
   - Keep validation, calibration, and test separate.
   - Do not tune thresholds on test data.
4. For threshold decisions, report precision, recall, false-positive rate, flagged positives/negatives, and prevalence sensitivity.
5. Compare a changed model with a suitable previous/baseline model.
6. Before finishing, identify leakage risks, separation risks, calibration risks, and false-positive consequences.
