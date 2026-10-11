---
name: anticheat-evaluation
description: Evaluate CS2 anticheat models with match-level separation, calibration discipline, false-positive analysis, and clear limitations.
---

For evaluation:

1. State dataset/version, split protocol, match-level split status, class counts, and validation/calibration/test separation.
2. Report ROC-AUC, PR-AUC when available, precision, recall, F1, false-positive rate, and flagged positive/negative player-matches.
3. Treat accuracy as secondary because the labels are imbalanced.
4. For each operating threshold, state how it was selected and verify it was not selected using test data.
5. Show recall versus false-positive tradeoffs and note that precision depends on real cheating prevalence.
6. Compare against appropriate baselines, including always-negative where relevant.
7. State limitations: label noise, identity-remapping/separation uncertainty, encounter eligibility coverage, and lack of live validation.
