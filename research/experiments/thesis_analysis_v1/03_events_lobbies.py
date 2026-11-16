"""Scout-lobby stratification (A1, A2), label-noise bounds (A3) and per-encounter TCN analyses (T4, T5).

Run from this folder: python 03_events_lobbies.py   (see PREREGISTRATION.md)
"""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from common import OUT, STRICT, THRESHOLDS, V1, events, players
from anticheat.reference_lstm import choose_thresholds
from scripts.analyze_weapon_groups import ORDER, to_group

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
REC = {}
EVAL = ['validation', 'calibration', 'test']
rng = np.random.default_rng(20261116)


def boot_auc(y, s, groups, n=2000):
    """Match-cluster percentile interval for AUC via resampling weights."""
    u, inv = np.unique(groups, return_inverse=True)
    out = []
    for _ in range(n):
        w = np.bincount(rng.integers(len(u), size=len(u)), minlength=len(u))[inv]
        if (w * y).sum() and (w * (1 - y)).sum():
            out.append(roc_auc_score(y, s, sample_weight=w))
    return np.round(np.quantile(out, [.025, .975]), 3).tolist()


p = players()
e = events()
e['wg'] = to_group(e.weapon)
shares = pd.crosstab(e.player_match, e.wg, normalize='index').reindex(columns=ORDER, fill_value=0)
p['main'] = shares.idxmax(axis=1).reindex(p.player_match).to_numpy()
p['Scout'] = shares.Scout.reindex(p.player_match).to_numpy()
p['scout_main_in_match'] = p.match_id.map((p.main == 'Scout').groupby(p.match_id).sum())
p['lobby'] = p.scout_main_in_match >= 3
p['match_cheaters'] = p.match_id.map(p.groupby('match_id').label.sum())

print('## A1 Scout lobbies (>=3 scout-main players in the match), all splits, labels and weapons only')
m = p.groupby('match_id').agg(source=('source', 'first'), split=('split', 'first'), lobby=('lobby', 'first'),
                              players=('label', 'size'), cheaters=('label', 'sum'))
t = m.groupby(['source', 'lobby']).agg(matches=('players', 'size'), players=('players', 'sum'), cheater_labels=('cheaters', 'sum'),
                                       mean_cheaters=('cheaters', 'mean'))
print(t.round(2).to_string())
print('by split:\n', m.groupby(['split', 'source']).lobby.agg(['sum', 'size']).to_string())
pos = p[p.label == 1]
print(f'Share of all cheater-labelled player-matches that are in scout lobbies: {pos.lobby.mean():.3f}; '
      f'in matches with >=4 cheater labels: {(pos.match_cheaters >= 4).mean():.3f}; scout-main: {(pos.main == "Scout").mean():.3f}')
print('Scout-main label-0 player-matches inside lobbies, by folder:\n',
      p[(p.label == 0) & (p.main == 'Scout')].groupby('source').lobby.agg(['sum', 'size']).to_string())
REC['A1_lobby_counts'] = t.reset_index().to_dict('records')
REC['A1_positive_shares'] = {'in_lobby': float(pos.lobby.mean()), 'match_ge4_cheaters': float((pos.match_cheaters >= 4).mean()),
                             'scout_main': float((pos.main == 'Scout').mean())}

rows = []
for s in ['val+cal', 'test']:
    d = p[p.split.isin(['validation', 'calibration'])] if s == 'val+cal' else p[p.split == s]
    for name, x in [('all', d), ('outside lobbies', d[~d.lobby]), ('inside lobbies', d[d.lobby]),
                    ('outside lobbies, match has <=3 cheater labels', d[~d.lobby & (d.match_cheaters <= 3)]),
                    ('outside lobbies, reviewed only', d[~d.lobby & d.reviewed])]:
        neg, po = x[x.label == 0], x[x.label == 1]
        row = {'split': s, 'cohort': name, 'matches': x.match_id.nunique(), 'neg': len(neg), 'pos': len(po),
               'AUC': round(roc_auc_score(x.label, x.cal), 4), 'AUC ci': boot_auc(x.label.to_numpy(), x.cal.to_numpy(), x.group.to_numpy())}
        for k in ['accuracy', 'default_0_5', 'fpr_1pct']:
            row[f'{k} recall'] = f'{int((po.cal >= THRESHOLDS[k]).sum())}/{len(po)} = {(po.cal >= THRESHOLDS[k]).mean():.3f}'
            row[f'{k} FP'] = f'{int((neg.cal >= THRESHOLDS[k]).sum())}/{len(neg)} = {100 * (neg.cal >= THRESHOLDS[k]).mean():.2f}%'
        rows.append(row)
print('\nPerformance inside and outside scout lobbies (frozen thresholds):')
print(pd.DataFrame(rows).T.to_string())
REC['A1_performance'] = rows

print('\n## A2 Strict-threshold sensitivity (calibration -> validation; test not used)')
c, v = p[p.split == 'calibration'], p[p.split == 'validation']
rows = []
for name, cc in [('frozen: all calibration negatives', c), ('calibration outside scout lobbies', c[~c.lobby]),
                 ('calibration reviewed matches only', c[c.reviewed])]:
    thr = choose_thresholds(cc.label.to_numpy(), cc.cal.to_numpy())['fpr_1pct']
    for vname, vv in [('validation all', v), ('validation outside lobbies', v[~v.lobby])]:
        neg, po = vv[vv.label == 0], vv[vv.label == 1]
        rows.append({'threshold from': name, 'cal negatives': int((cc.label == 0).sum()), 'threshold': round(thr, 4), 'applied to': vname,
                     'recall': f'{int((po.cal >= thr).sum())}/{len(po)} = {(po.cal >= thr).mean():.3f}',
                     'FP': f'{int((neg.cal >= thr).sum())}/{len(neg)} = {100 * (neg.cal >= thr).mean():.2f}%'})
print(pd.DataFrame(rows).to_string(index=False))
REC['A2_threshold_sensitivity'] = rows

print('\n## A3 Label-noise bounds (publisher: non-cheater label precision 97.2% in unreviewed matches)')
rows = []
t_ = p[p.split == 'test']
unrev_neg = int(((t_.label == 0) & ~t_.reviewed).sum())
for eps in [0.0, 0.014, 0.028, 0.056]:
    hidden = eps * unrev_neg
    for k in ['fpr_1pct', 'accuracy']:
        thr = THRESHOLDS[k]
        tp = int(((t_.cal >= thr) & (t_.label == 1)).sum()); fp = int(((t_.cal >= thr) & (t_.label == 0)).sum())
        fp_unrev = int(((t_.cal >= thr) & (t_.label == 0) & ~t_.reviewed).sum())
        npos, nneg = int(t_.label.sum()), int((t_.label == 0).sum())
        caught = min(fp_unrev, hidden)   # best case: flagged unreviewed negatives are the hidden cheaters
        rows.append({'hidden-cheater rate': eps, 'policy': k, 'hidden cheaters (expected)': round(hidden, 1),
                     'FPR measured %': round(100 * fp / nneg, 2),
                     'FPR if flags are hidden cheaters %': round(100 * (fp - caught) / (nneg - hidden), 2),
                     'FPR if none are %': round(100 * fp / (nneg - hidden), 2),
                     'recall measured': round(tp / npos, 3),
                     'recall best case': round((tp + caught) / (npos + hidden), 3),
                     'recall worst case': round(tp / (npos + hidden), 3),
                     'precision measured': round(tp / (tp + fp), 3), 'precision best case': round((tp + caught) / (tp + fp), 3)})
print(pd.DataFrame(rows).to_string(index=False))
REC['A3_label_noise_bounds_test'] = rows

# ---------------- per-encounter TCN scores: test split only ----------------
z = np.load(V1 / 'tcn39_seed123_test_events.npz')
ev = e.iloc[z['event_index']].copy()
ev['prob'] = z['scores'].astype(np.float64)
ev['logit'] = logit(np.clip(ev.prob, 1e-7, 1 - 1e-7))
print('\n## T4 Event-drop attribution, TCN component only, test split (AUC of mean event logit)')
rows = []
for name, keep in [('all encounters', ev.wg.notna() | True), ('without scout', ev.wg != 'Scout'),
                   ('without any sniper (AWP, scout, auto)', ~ev.wg.isin(['AWP', 'Scout', 'Auto-sniper'])),
                   ('scout encounters only', ev.wg == 'Scout'), ('rifle + SMG encounters only', ev.wg.isin(['Rifle', 'SMG']))]:
    kept = ev[keep].groupby('player_match').agg(s=('logit', 'mean'), label=('label', 'first'), group=('group', 'first'))
    full = ev[ev.player_match.isin(kept.index)].groupby('player_match').agg(s=('logit', 'mean'), label=('label', 'first'), group=('group', 'first')).loc[kept.index]
    rows.append({'encounters used': name, 'players kept': len(kept), 'pos': int(kept.label.sum()),
                 'AUC, subset of encounters': round(roc_auc_score(kept.label, kept.s), 4),
                 'ci': boot_auc(kept.label.to_numpy(), kept.s.to_numpy(), kept.group.to_numpy()),
                 'AUC, all encounters, same players': round(roc_auc_score(full.label, full.s), 4),
                 'median pos score subset/all': f'{expit(kept.s[kept.label == 1].median()):.2f}/{expit(full.s[full.label == 1].median()):.2f}',
                 'median neg score subset/all': f'{expit(kept.s[kept.label == 0].median()):.3f}/{expit(full.s[full.label == 0].median()):.3f}'})
print(pd.DataFrame(rows).T.to_string())
REC['T4_event_drop'] = rows

print('\n## T5 Encounter-level TCN score by weapon group, test split (labels are player-match labels)')
rows = []
for g in ORDER:
    x = ev[ev.wg == g]
    if x.label.nunique() < 2 or x.label.sum() < 20:
        continue
    rows.append({'group': g, 'neg': int((x.label == 0).sum()), 'pos': int(x.label.sum()),
                 'event AUC': round(roc_auc_score(x.label, x.prob), 3), 'ci': boot_auc(x.label.to_numpy(), x.prob.to_numpy(), x.group.to_numpy(), 500),
                 'neg >=0.5 %': round(100 * (x.prob[x.label == 0] >= .5).mean(), 1), 'pos >=0.5 %': round(100 * (x.prob[x.label == 1] >= .5).mean(), 1),
                 'neg median': round(x.prob[x.label == 0].median(), 3), 'pos median': round(x.prob[x.label == 1].median(), 3)})
print(pd.DataFrame(rows).to_string(index=False))
REC['T5_event_by_weapon'] = rows
# cheater-labelled players' own encounters: is the evidence spread or concentrated?
pm = ev[ev.label == 1].groupby('player_match').prob.agg(n='size', share_hi=lambda s: (s >= .5).mean())
print('cheater-labelled test players: median share of their encounters scoring >=0.5:', round(pm.share_hi.median(), 2),
      '| players with no encounter >=0.5:', int((pm.share_hi == 0).sum()), 'of', len(pm))
ev[['player_match', 'label', 'wg', 'prob']].assign(lobby=ev.match_id.map(p.groupby('match_id').lobby.first()).to_numpy()).to_csv(OUT / 'test_event_scores_by_weapon.csv', index=False)
(OUT / '03_events_lobbies.json').write_text(json.dumps(REC, indent=1, default=str))
