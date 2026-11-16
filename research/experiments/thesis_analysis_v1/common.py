"""Shared loaders for the thesis analysis. Reads frozen cs2cd_v1 artifacts only; writes nothing there."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH))
from anticheat.reference_lstm import apply_calibration  # noqa: E402

V1 = RESEARCH / 'experiments' / 'cs2cd_v1'
OUT = Path(__file__).resolve().parent / 'records'
OUT.mkdir(exist_ok=True)
CAL = json.loads((V1 / 'calibration.json').read_text())['models']['selected']
THRESHOLDS = CAL['thresholds']
STRICT = THRESHOLDS['fpr_1pct']


def players():
    """Eligible player-matches for all splits with raw blend score `selected` and calibrated `cal`.

    Train rows have no saved score (NaN)."""
    p = pd.read_csv(V1 / 'player_index.csv')
    p['selected'] = np.nan
    p['tcn'] = np.nan
    p['lgbm'] = np.nan
    va =np.load(V1 / 'tcn39_seed123_validation.npz')
    lg = np.load(V1 / 'lgbm39_leaves15_validation.npz')
    assert np.array_equal(va['player_index'], lg['player_index'])
    p.loc[va['player_index'], 'selected'] = .75 * va['scores'] + .25 * lg['scores']
    p.loc[va['player_index'], 'tcn'] = va['scores']
    p.loc[va['player_index'], 'lgbm'] = lg['scores']
    c = np.load(V1 / 'calibration_predictions.npz')
    p.loc[c['player_index'], 'selected'] = c['selected']
    p.loc[c['player_index'], 'tcn'] = c['tcn39_seed123']
    p.loc[c['player_index'], 'lgbm'] = c['lgbm39_leaves15']
    t = pd.read_csv(V1 / 'test_predictions.csv').set_index('player_match')
    test = p.split == 'test'
    for col, src in [('selected', 'selected'), ('tcn', 'tcn39_seed123'), ('lgbm', 'lgbm39_leaves15')]:
        p.loc[test, col] = t.loc[p.player_match[test], src].to_numpy()
    p['cal'] = apply_calibration(p.selected.fillna(.5), CAL['platt'])
    p.loc[p.selected.isna(), 'cal'] = np.nan
    p['reviewed'] = p.source == 'with_cheater_present'
    return p


def events():
    return pd.read_parquet(V1 / 'events.parquet')


def cluster_boot(df, stat, group='group', n=2000, seed=20261116):
    """Percentile 95% interval of stat(df) resampling whole match groups."""
    rng = np.random.default_rng(seed)
    parts = [g for _, g in df.groupby(group, sort=False)]
    vals = []
    for _ in range(n):
        pick = rng.integers(len(parts), size=len(parts))
        try:
            vals.append(stat(pd.concat([parts[i] for i in pick])))
        except ValueError:  # resample without both classes
            continue
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
