# Kaggle evaluation protocol

The prediction unit is one publisher-defined record containing 30 engagements, each with 192 ticks and five input channels. Class 1 is the publisher's cheating label. Engagements from a record remain together. The benchmark measures retrospective classification of recorded gameplay.

## Data audit and split

Kaggle version 1 is pinned by its archive SHA-256. The audit checks array shape, finite values, class balance, constant windows and duplicate records. Records sharing informative engagements or identical 64-tick attacker movement and firing paths are joined, including matches at different time offsets or with different target fields. Motion and distinct-angle requirements exclude trivial matches. Groups with contradictory labels are excluded.

The initial reserve uses ten deterministic stratified group folds, with seed 20260906: folds 0–1 for test, 2 for calibration, 3 for validation and 4–9 for training. A deeper overlap audit was completed after training and validation pilot runs, before calibration or test evaluation. The pilot models were superseded and rerun on the final split.

To preserve the unseen reserve, each enlarged overlap group takes the earliest role in the order training, validation, calibration, test. A group touching training cannot enter a holdout, and every final test record was in the initial test reserve. Final counts appear in `data_audit.json`. An intermediate redraw contributed no inspected model results or selections. Learned feature transformations use training records only.

## Models and features

The comparison covers constant prediction, regularized logistic regression, LightGBM, XGBoost, CatBoost, ExtraTrees, an engagement classifier with record aggregation, an independent two-layer LSTM and a temporal convolution model with attention over engagements. Neural candidates include seeds 42, 123 and 2026, averaged weights, a five-channel CNN ablation and a hybrid using record statistics.

The independent LSTM in this comparison is separate from the revision of Ionuț’s published architecture. The latter is documented in `../METHODS.md` and `../experiments/lstm_revision/`. A near-chance BF16 LSTM candidate is recorded in the numeric search results; the usable baseline uses FP32 computation and a lower learning rate. Neural inputs are cached in FP16, while computation precision is configured separately.

Derived features include acceleration, jerk, angular speed, aim-error magnitude, firing-conditioned summaries and statistics at multiple time scales. These transformations use the existing observations. Distribution summaries pool engagements within each record.

## Selection, calibration and reporting

Model and ensemble selection uses validation ROC-AUC, with average precision, training-validation differences and false-positive behavior as supporting diagnostics. Selection is fixed before calibration or test predictions are inspected. Score calibration and the F1 and low-false-positive thresholds use the separate calibration records.

The final test reports ROC-AUC, average precision, accuracy, balanced accuracy, precision, recall, F1, Brier score and confusion counts. False-positive rates are measured at thresholds targeting 1% and 0.1% on calibration data. A target does not guarantee the same rate on test or deployed data; the small clean calibration sample cannot establish reliable 0.1% performance. Confidence intervals resample complete overlap groups.

Record-level scores and Ionuț’s engagement-level journal results have different units and splits. Their comparison is explained in `../METHODS.md`. The separate LSTM revision is a supplementary evaluation on a previously examined test set. A new final holdout is required for confirmation.

References: [dataset publisher](https://www.kaggle.com/datasets/emstatsl/csgo-cheating-dataset), [grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html), [attention multiple-instance learning](https://arxiv.org/abs/1802.04712), [InceptionTime](https://arxiv.org/abs/1909.04939). The temporal architecture is a compact implementation informed by these methods; the experiment uses its own benchmark and evaluation design.
