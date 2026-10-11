# CS2 Anti-Cheat ML — Project Instructions

## Read first
Before nontrivial work, read only the relevant parts of:
- `README.md`
- `research/METHODS.md`
- `research/REPRODUCING.md`
- `research/DATA_SOURCES.md`

## Repository map
- `src/`, `notebooks/`: original LSTM prototype and exploration.
- `research/anticheat/`: extraction, features, training, calibration, and inference.
- `research/scripts/`: command-line experiment and validation scripts.
- `research/experiments/`: protocols, comparisons, predictions, and experiment outputs.
- `research/tests/`: geometry, split-separation, and model-parity tests.
- `research/data/`, `research/models/`, `research/results/`: local data, model, and result artifacts.

## Anticheat ML invariants
- Inputs are four-second encounter windows before the first bullet hit.
- Every feature must be available before the prediction time.
- The victim may be identified retrospectively, but later events must not leak into pre-hit features.
- Labels are player-match labels, not engagement- or tick-level cheat labels.
- Preserve complete-match separation in CS2CD splits.
- Keep training, validation, calibration, and test partitions separate.
- Fit preprocessors using training data only.
- Never tune thresholds on the test set.
- Treat negative labels as potentially noisy.
- Report precision, recall, false-positive rate, and threshold selection; do not rely on accuracy alone.
- Do not claim validated live anti-cheat performance without independently collected deployment data.

## Commands
- Install: `python -m pip install -r research/requirements.txt`
- Verify artifacts: `cd research && python scripts/verify_artifacts.py`
- Example inference: `cd research && python scripts/predict_cs2cd.py examples/example_validation_player.npz --output outputs/example_scores.csv`
- Tests: `cd research && python -m pytest -q tests`
