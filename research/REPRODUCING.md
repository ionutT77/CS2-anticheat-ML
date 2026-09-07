# Run and reproduce

Commands below run from `research/` unless stated otherwise. The supplied results are frozen. Training scripts refuse to alter a selected CS2 experiment or a completed LSTM repair.

## CPU inference and verification

Install `requirements.txt` in a virtual environment. Python 3.13 was used for the supplied artifacts. `requirements-training.txt` adds the optional broader Kaggle search dependencies. Neural training and full CS2 evaluation require CUDA; example inference and tests run on CPU.

```bash
python scripts/verify_artifacts.py
python scripts/predict_cs2cd.py examples/example_validation_player.npz --output outputs/example_scores.csv
python -m pytest -q tests
```

Compare the output with `examples/example_expected_scores.csv`; floating-point differences around 1e-6 are acceptable across supported CPU/GPU implementations. The input is a small validation example, so running it does not inspect the reserved test set.

For another extracted CS2 match:

```bash
python scripts/predict_cs2cd.py /path/to/match.npz --output outputs/match_scores.csv
```

The NPZ must contain `x` with shape `[encounters, 256, 39]`, `player` with one match-local identifier per encounter and `tick` for chronological ordering. Put one match in each file. Use the raw feature order in `examples/input_schema.json`; do not normalize it beforehand. The output contains calibrated scores and flags for named policies, together with the encounter count.

For the revised original LSTM:

```bash
python scripts/predict_reference_lstm.py /path/to/engagements_or_records.npy --output outputs/lstm_scores.csv
```

Use raw `[N, 192, 5]` engagements or `[N, 30, 192, 5]` records, with binary firing in the last channel. For the selected Kaggle CNN pair, use records:

```bash
python scripts/predict.py /path/to/records.npy --output outputs/cnn_scores.csv
```

Kaggle and CS2 inputs have different sampling rates, windows and channel definitions. Each saved model needs its own input format.

## Fresh CS2 benchmark run

Raw downloads occupy 52.6 GB; allow at least 80 GB for raw data, extracted windows, caches and outputs. The current neural trainer requires CUDA and loads the sequence cache into device memory. Reducing memory use would require a loader change. Saved-model inference also supports CPU execution.

Create an isolated directory beside the frozen research directory. The helper copies source, pinned download manifests and the original match split/protocol. It does not copy trained CS2 models or test results.

```bash
python scripts/new_experiment.py ../local_runs/cs2cd_repeat --benchmark cs2cd
cd ../local_runs/cs2cd_repeat
python scripts/download_cs2cd.py --workers 4
python scripts/prepare_cs2cd.py --workers 3
python scripts/audit_cs2cd.py --finalize
python scripts/cache_cs2cd.py
python scripts/train_cs2cd_neural.py
python scripts/train_cs2cd_tabular.py
python scripts/select_cs2cd.py
python scripts/evaluate_cs2cd.py
python scripts/evaluate_cs2cd.py --test
python scripts/analyze_cs2cd.py
```

The first evaluation command fits calibration on its separate split. `--test` performs the final test evaluation. The helper's copied split is already frozen, so the split-creation form of `audit_cs2cd.py` is not needed here.

To reuse an existing raw download, add `--raw-data /absolute/path/to/data/cs2cd/raw` to the helper command and skip downloading. This creates a directory symlink. On systems without symlink permission, omit the option and copy the raw folders into the new run's `data/cs2cd/raw/` directory. Extraction writes into the new run, leaving source files intact.

This repeats the saved benchmark. Its test set is now known. New feature or model development needs a new protocol and an independent final cohort before making new performance claims. A changed extraction implementation must be audited and paired with a new protocol, rather than bypassing the hash checks.

## Repeat the LSTM repair

The helper retains the frozen Kaggle CNN reference, split and audit files that the LSTM experiment checks for integrity. It leaves the LSTM experiment directory empty.

```bash
python scripts/new_experiment.py ../local_runs/lstm_repeat --benchmark reference-lstm
cd ../local_runs/lstm_repeat
python scripts/download_data.py
python scripts/improve_reference_lstm.py
```

An existing Kaggle extraction can be reused with `--raw-data /absolute/path/to/data/raw/extracted`. That folder must contain `legit/` and `cheaters/`. The four configured candidates include the original training recipe and three revisions. Validation chooses the revised candidate and record pooling; calibration chooses thresholds.

## Inspect the saved experiments

`python scripts/audit_reference_split.py` reconstructs the original seed-42 split against the frozen record and overlap index. It reproduces the 699 shared test records and 25 exact duplicate hashes without downloading raw data. The original audit also checked those full-record duplicates against the raw arrays.

`experiments/cs2cd_v1/test_results.json` contains the model comparison, both evaluation cohorts, threshold counts, confidence intervals and observation horizons. `test_predictions.csv` supports independent metric calculations. `selection.json` records the validation choice and hashes; `calibration.json` records the fitted score mapping and thresholds.

`experiments/lstm_revision/` contains four candidates, numerical learning curves, reference and improved predictions, and paired confidence intervals. Its protocol preserves the inference-precision amendment so the evaluation can be traced.

`results/test_metrics.json` is the broader Kaggle benchmark. Only its selected CNN pair is packaged from that search; the LSTM repair and all ten CS2 candidate weights are included. Regenerating omitted Kaggle candidates requires the corresponding training scripts and optional dependencies.

Saved weights, source files and the evaluation chain can be checked with `verify_artifacts.py`. The hash checks detect modified artifacts; they do not replace evaluation on independent data.
