"""Reconstruct the original seed-42 split using the frozen overlap index.

The split recipe is from upstream commit 06427ce. This checks the published
record grouping without executing the original notebook or loading raw samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reference_split_audit.json")
    args = parser.parse_args()
    frame = pd.read_csv(ROOT / "data/processed/splits.csv").set_index("record_id")
    # Upstream loads positive records first; the audit index numbers negatives first.
    record_ids = np.r_[np.arange(10000, 12000), np.arange(10000)]
    labels = frame.loc[record_ids, "label"].to_numpy()
    np.testing.assert_array_equal(labels, np.r_[np.ones(2000), np.zeros(10000)])
    trainval, test = train_test_split(np.arange(12000), test_size=.15, stratify=labels, random_state=42)
    train, validation = train_test_split(trainval, test_size=.15/.85, stratify=labels[trainval], random_state=42)
    training = frame.loc[record_ids[train]]
    result = {}
    for name, indices in [("train", train), ("validation", validation), ("test", test)]:
        rows = frame.loc[record_ids[indices]]
        result[name] = {
            "records": len(rows),
            "positives": int(rows.label.sum()),
            "always_noncheater_accuracy": float(1-rows.label.mean()),
        }
        if name != "train":
            result[name].update({
                "records_in_groups_also_in_training": int(rows.group.isin(training.group).sum()),
                "exact_record_hashes_also_in_training": int(rows.sha256.isin(training.sha256).sum()),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
