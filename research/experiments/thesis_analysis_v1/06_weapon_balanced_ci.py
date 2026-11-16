"""Match-cluster intervals for the weapon-balanced AUC (weights recomputed inside every resample).

Run from this folder: python 06_weapon_balanced_ci.py   (reads records/scored_players.csv from 01)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import OUT

rng = np.random.default_rng(20261116)
p = pd.read_csv(OUT / 'scored_players.csv')
p['match'] = p.player_match.str.rsplit('/', n=1).str[0]


def balanced_auc(d):
    tab = pd.crosstab(d.main, d.label)
    dd = d[d.main.isin(tab.index[(tab[0] > 0) & (tab[1] > 0)])]
    within = dd.groupby(['main', 'label']).label.transform('size') / dd.groupby('label').label.transform('size')
    w = dd.main.map(dd.main.value_counts(normalize=True)) / within
    return roc_auc_score(dd.label, dd.cal, sample_weight=w)


out = {}
for name, d in [('validation', p[p.split == 'validation']), ('calibration', p[p.split == 'calibration']),
                ('val+cal', p[p.split.isin(['validation', 'calibration'])]), ('test', p[p.split == 'test'])]:
    parts = [g for _, g in d.groupby('match')]
    vals = [balanced_auc(pd.concat([parts[i] for i in rng.integers(len(parts), size=len(parts))], ignore_index=True)) for _ in range(500)]
    out[name] = {'weapon_balanced_auc': round(balanced_auc(d), 4), 'ci95': np.round(np.quantile(vals, [.025, .975]), 3).tolist(),
                 'raw_auc': round(roc_auc_score(d.label, d.cal), 4)}
    print(name, out[name], flush=True)
(OUT / '06_weapon_balanced_ci.json').write_text(json.dumps(out, indent=1))
