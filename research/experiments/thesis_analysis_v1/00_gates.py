"""Gate checks: saved numbers reproduce before anything is built on them. Run from research/."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import V1, STRICT, THRESHOLDS, players, events, RESEARCH

p = players()
sel = json.loads((V1 / 'selection.json').read_text())
res = json.loads((V1 / 'test_results.json').read_text())

# 1. headline
t = p[p.split == 'test']
auc = roc_auc_score(t.label, t.selected)
tp = int(((t.cal >= STRICT) & (t.label == 1)).sum()); fp = int(((t.cal >= STRICT) & (t.label == 0)).sum())
print(f'test n={len(t)} pos={t.label.sum()} AUC={auc:.4f} strict TP={tp} FP={fp} thr={STRICT:.4f}')
assert abs(auc - res['cohorts']['all']['models']['selected']['roc_auc']) < 1e-6 and (tp, fp) == (67, 4)
for name, thr in THRESHOLDS.items():
    f = t.cal >= thr
    print(f'  {name:12s} thr={thr:.4f} TP={(f & (t.label == 1)).sum()} FP={(f & (t.label == 0)).sum()}')

# validation blend matches selection.json
v = p[p.split == 'validation']
va = roc_auc_score(v.label, v.selected); vr = roc_auc_score(v.label[v.reviewed], v.selected[v.reviewed])
print(f'validation all={va:.4f} reviewed={vr:.4f} winner={sel["winner"]["all_auc"]:.4f}/{sel["winner"]["reviewed_auc"]:.4f}')
assert abs(va - sel['winner']['all_auc']) < 1e-6 and abs(vr - sel['winner']['reviewed_auc']) < 1e-6
c = p[p.split == 'calibration']
print('calibration strict:', ((c.cal >= STRICT) & (c.label == 1)).sum(), 'TP',
      ((c.cal >= STRICT) & (c.label == 0)).sum(), 'FP of', (c.label == 0).sum())

# 2. per-event TCN file: logits or probabilities, and does it rebuild the player column?
e = events()
z = np.load(V1 / 'tcn39_seed123_test_events.npz')
s = z['scores'].astype(np.float64)
print('event scores range', s.min(), s.max(), 'n', len(s))
ev = e.iloc[z['event_index']].assign(s=s)
assert (ev.split == 'test').all()
from scipy.special import expit, logit
as_logit = expit(ev.groupby('player_match').s.mean())
as_prob = expit(ev.assign(l=logit(np.clip(s, 1e-7, 1 - 1e-7))).groupby('player_match').l.mean())
ref = t.set_index('player_match').tcn
print('max |diff| if file holds logits:', float((as_logit - ref).abs().max()),
      '| if probabilities:', float((as_prob - ref).abs().max()))
