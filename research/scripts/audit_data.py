"""Audit original records; connect shared exact engagements before splitting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from anticheat.data import ROOT, CHANNELS, load_arrays


def main():
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "overlap_duplicate_edges.npy").exists():
        raise RuntimeError("Run scripts/audit_overlaps.py first; overlap grouping is required for the final benchmark")
    (ROOT / "reports").mkdir(exist_ok=True)
    x, y = load_arrays()
    parent = np.arange(len(y))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen = {}
    record_hashes = []
    within_duplicates = 0
    cross_duplicates = 0
    empty = 0
    nonfinite = 0
    for i, bag in enumerate(x):
        nonfinite += int((~np.isfinite(bag)).sum())
        record_hashes.append(hashlib.sha256(bag.tobytes()).hexdigest())
        local = set()
        for engagement in bag:
            # Constant/empty windows are not evidence of a shared identity.
            if np.ptp(engagement[:, :4], axis=0).max() < 1e-6:
                empty += 1
                continue
            key = hashlib.blake2b(engagement.tobytes(), digest_size=16).digest()
            if key in local:
                within_duplicates += 1
            local.add(key)
            if key in seen and seen[key] != i:
                cross_duplicates += 1
                union(i, seen[key])
            else:
                seen[key] = i
        if i % 2000 == 0:
            print("Audited", i, "records", flush=True)
    if nonfinite:
        raise ValueError(f"Nonfinite input values: {nonfinite}; investigate before training")
    initial_groups = np.array([find(i) for i in range(len(y))])
    initial_conflicts = [g for g in np.unique(initial_groups) if len(np.unique(y[initial_groups == g])) > 1]
    initial_ids = np.where(~np.isin(initial_groups, initial_conflicts))[0]
    initial_split = np.full(len(y), "excluded_conflict", dtype="U20")
    fold = np.full(len(y), -1)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=20260906)
    for k, (_, te) in enumerate(cv.split(initial_ids, y[initial_ids], initial_groups[initial_ids])):
        fold[initial_ids[te]] = k
        initial_split[initial_ids[te]] = "test" if k < 2 else "calibration" if k == 2 else "validation" if k == 3 else "train"
    overlap_edges = 0
    for filename in ["extra_movement_duplicate_edges.npy", "overlap_duplicate_edges.npy"]:
        path = out / filename
        if path.exists():
            edges = np.load(path, allow_pickle=False)
            for a, b in edges:
                union(int(a), int(b))
            overlap_edges += len(edges)
    groups = np.array([find(i) for i in range(len(y))])
    conflicted = [g for g in np.unique(groups) if len(np.unique(y[groups == g])) > 1]
    eligible = ~np.isin(groups, conflicted)
    split = np.full(len(y), "excluded_conflict", dtype="U20")
    # Preserve the original unseen test reserve after discovering extra links.
    # A group touching development data is promoted to development, never test.
    priority = ["train", "validation", "calibration", "test"]
    for g in np.unique(groups[eligible]):
        members = groups == g
        roles = set(initial_split[members])
        split[members] = next(role for role in priority if role in roles)
    assert np.all(initial_split[split == "test"] == "test")
    assert not np.isin(initial_split[split == "calibration"], ["train", "validation"]).any()
    df = pd.DataFrame({"record_id": np.arange(len(y)), "label": y, "group": groups,
                       "source_row": np.r_[np.arange((y==0).sum()),np.arange((y==1).sum())],
                       "source": np.where(y, "cheaters", "legit"), "initial_fold": fold,
                       "initial_split": initial_split, "split": split, "sha256": record_hashes})
    assert df.groupby("group").split.nunique().max() == 1
    assert not df.sha256.duplicated().any() or df.groupby("sha256").split.nunique().max() == 1
    df.to_csv(out / "splits.csv", index=False)
    sizes = Counter(groups)
    stats = {}
    for label, name in [(0, "legit"), (1, "cheater")]:
        a = x[y==label]
        sample = a[::20].reshape(-1, 5)
        stats[name] = {"records": len(a), "zero_tick_fraction": float(np.all(a == 0, axis=-1).mean()),
                       "firing_values": np.unique(a[...,4]).tolist(),
                       "sample_channel_quantiles": {c: np.quantile(sample[:,j], [0,.01,.5,.99,1]).tolist() for j,c in enumerate(CHANNELS)}}
    audit = {"shape": list(x.shape), "dtype": str(x.dtype), "nonfinite_values": nonfinite,
             "class_counts": Counter(y.tolist()), "majority_accuracy": float((y==0).mean()),
             "constant_engagements": empty, "duplicate_full_records": len(y)-len(set(record_hashes)),
             "duplicate_engagements_within_records": within_duplicates,
             "shared_exact_engagement_occurrences_across_records": cross_duplicates,
             "additional_trajectory_overlap_edges": overlap_edges,
             "independent_duplicate_groups": len(sizes), "largest_duplicate_group": max(sizes.values()),
             "conflicting_label_groups": len(conflicted), "excluded_conflicting_records": int((~eligible).sum()),
             "splits": {s: {"records": len(z), "cheaters": int(z.label.sum()), "legit": int((z.label==0).sum()), "groups": z.group.nunique()} for s,z in df.groupby("split")},
             "class_audit": stats,
             "limitations": ["Player is a source array row according to the publisher; no Steam IDs or match IDs are supplied.",
                             "Duplicate-connected rows stay together, but true identity, match, temporal and provenance separation cannot be verified.",
                             "Labels and sampling procedures cannot be independently verified from these arrays.",
                             "Each engagement includes 5 seconds before and 1 second after; this benchmark is retrospective.",
                             "Kaggle lists the dataset license as Unknown; archive is retained locally, not redistributed."]}
    (ROOT / "reports/data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
