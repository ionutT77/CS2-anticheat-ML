"""Score the staged OpenCS2 pro windows with the frozen cs2cd_v1 blend on CPU (P1, P2 inputs).

Run from this folder: python 02_score_pro.py
Inference only. Writes records/pro_players.csv, records/pro_channel_shift.csv and cache/ (events,
player features). Player identities are hashed before anything is written.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from common import OUT, RESEARCH, V1, CAL, apply_calibration
from anticheat.cs2_data import FEATURES, transform
from anticheat.cs2_features import summarize_events, summarize_player
from anticheat.cs2_inference import predict
from anticheat.cs2_models import MatchDetector
from anticheat.cs2_training import predict_players
from scripts.analyze_weapon_groups import to_group

CACHE = Path(__file__).resolve().parent / 'cache'
CACHE.mkdir(exist_ok=True)
SALT = 'thesis_analysis_v1'
pid = lambda steamid: hashlib.sha256((SALT + str(steamid)).encode()).hexdigest()[:8]

norm = json.loads((V1 / 'normalization.json').read_text())
mean, std = torch.tensor(norm['mean']), torch.tensor(norm['std'])
ck = torch.load(V1 / 'tcn39_seed123.pt', map_location='cpu', weights_only=True)
tcn = MatchDetector(ck['config']['channels'], ck['config']['architecture'], ck['config'].get('hierarchical', False))
tcn.load_state_dict(ck['state_dict'])
lg = joblib.load(V1 / 'lgbm39_leaves15.joblib')
SCOPED, DYAW, SNIPER = FEATURES.index('scoped'), FEATURES.index('delta_yaw'), FEATURES.index('weapon_sniper')

rows, ev_rows, feats, shift = [], [], [], []
checked = False
for name in ['awp_v1', 'scout_v1']:
    root = RESEARCH / 'data' / 'opencs2' / name
    ev = pd.concat([pd.read_parquet(root / 'events.parquet').assign(cohort='addition'),
                    pd.read_parquet(root / 'reserve_events.parquet').assign(cohort='reserve')])
    ev['wg'] = to_group(ev.weapon)
    for npz, g in ev.groupby('npz', sort=True):
        z = np.load(RESEARCH / npz)
        raw, ids, ticks = z['x'], z['player'].astype(str), z['tick']
        meta = json.loads((RESEARCH / npz).with_suffix('.json').read_text())
        steam = {q['local_id']: pid(q['steamid']) for q in meta['players']}
        unique = sorted(set(ids))
        bags = [np.flatnonzero(ids == u)[np.argsort(ticks[ids == u], kind='stable')] for u in unique]
        tr = transform(raw)
        event_features = summarize_events(tr)
        tab = np.stack([summarize_player(event_features[b]) for b in bags])
        logits, ev_logits, order = predict_players(tcn, torch.from_numpy(tr.astype(np.float16)), mean, std, bags, np.arange(len(bags)))
        p_tcn = expit(logits)
        p_lgb = lg['model'].predict_proba(tab[:, lg['feature_indices']])[:, 1]
        blend = .75 * p_tcn + .25 * p_lgb
        if not checked:  # self-check against the packaged inference path
            ref = {r['player']: r['raw_score'] for r in predict(raw, ids, ticks, V1)}
            assert max(abs(ref[u] - b) for u, b in zip(unique, blend)) < 1e-6
            checked = True
        g = g.set_index('local_index')
        event_logit = np.empty(len(raw)); event_logit[order] = ev_logits
        zn = ((torch.from_numpy(tr) - mean) / std).numpy()
        for sniper in (0, 1):
            m = raw[:, -1, SNIPER] == sniper
            if m.any():
                shift.append({'set': name, 'sniper': sniper, 'n': int(m.sum()), **{f'm_{f}': zn[m, :, i].mean() for i, f in enumerate(FEATURES)},
                              **{f's_{f}': (zn[m, :, i] ** 2).mean() for i, f in enumerate(FEATURES)},
                              **{f'z_{f}': (raw[m, :, i] == 0).mean() for i, f in enumerate(FEATURES)}})
        for u, b, a, c, d in zip(unique, bags, p_tcn, p_lgb, blend):
            w = g.loc[b]
            dy = np.abs(raw[b][..., DYAW][raw[b][..., SCOPED] == 0]); dy = dy[dy > 1e-6]
            rows.append({'set': name, 'cohort': w.cohort.iloc[0], 'player_match': w.player_match.iloc[0], 'match_group': w.group.iloc[0],
                         'pid': steam[u], 'encounters': len(b), 'Scout': (w.wg == 'Scout').mean(), 'AWP': (w.wg == 'AWP').mean(),
                         'main': w.wg.value_counts().idxmax(), 'tcn': a, 'lgbm': c, 'selected': d,
                         'yaw_q_min': dy.min() if len(dy) else np.nan, 'yaw_q_p1': np.quantile(dy, .01) if len(dy) else np.nan,
                         'yaw_nonzero': len(dy)})
            ev_rows.append(pd.DataFrame({'pid': steam[u], 'player_match': w.player_match.iloc[0], 'set': name, 'wg': w.wg.to_numpy(),
                                         'tcn_event': expit(event_logit[b])}))
        feats.append(tab)
        print(name, npz.split('/')[-1], len(raw), flush=True)

out = pd.DataFrame(rows)
out['cal'] = apply_calibration(out.selected.to_numpy(), CAL['platt'])
out.to_csv(OUT / 'pro_players.csv', index=False)
pd.DataFrame(shift).to_csv(CACHE / 'pro_channel_shift_by_match.csv', index=False)
pd.concat(ev_rows).to_parquet(CACHE / 'pro_events.parquet')
np.save(CACHE / 'pro_player_features.npy', np.concatenate(feats))

# P1 parity with the saved pro scores
old = pd.read_csv(RESEARCH.parent / 'local_runs' / 'opencs2_pro_fpr_scores.csv').set_index('player')
new = out[out.set == 'awp_v1'].set_index('player_match')
diff = (new.cal.reindex(old.index) - old.calibrated_score).abs()
print('P1 parity: rows', len(old), 'matched', int(diff.notna().sum()), 'max |diff|', float(diff.max()))
