"""Pro-demo analyses on the scores from 02_score_pro.py: P2 flag rates, P3 person effects, channel shift.

Run from this folder: python 04_pro.py   (all players are label 0 by assumption)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import beta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import OUT, THRESHOLDS, V1
from anticheat.cs2_data import FEATURES

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
CACHE = Path(__file__).resolve().parent / 'cache'
REC = {}
rng = np.random.default_rng(20261116)
d = pd.read_csv(OUT / 'pro_players.csv')
X = np.load(CACHE / 'pro_player_features.npy')
assert len(X) == len(d)
POL = ['accuracy', 'f1', 'default_0_5', 'fpr_1pct']


def cp(k, n):
    return [100 * (float(beta.ppf(.025, k, n - k + 1)) if k else 0.0), 100 * (float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0)]


def person_ci(x, thr):
    """People flagged at least once / people, and a person-level bootstrap of the player-map flag rate."""
    g = x.assign(f=x.cal >= thr).groupby('pid').f.agg(['sum', 'size'])
    idx = rng.integers(len(g), size=(2000, len(g)))
    rates = g['sum'].to_numpy()[idx].sum(1) / g['size'].to_numpy()[idx].sum(1)
    return int((g['sum'] > 0).sum()), len(g), np.round(100 * np.quantile(rates, [.025, .975]), 2).tolist()


print('## P2 Flag rates on legitimate pro players, frozen model and thresholds')
print('people:', d.pid.nunique(), '| player-maps:', len(d), '| maps per person: median', d.groupby('pid').size().median(), 'max', d.groupby('pid').size().max())
rows = []
cohorts = [('awp_v1 addition (pro AWPers) [second use]', (d.set == 'awp_v1') & (d.cohort == 'addition')),
           ('awp_v1 reserve [second use]', (d.set == 'awp_v1') & (d.cohort == 'reserve')),
           ('scout_v1 addition (>=3 scout encounters) [first use]', (d.set == 'scout_v1') & (d.cohort == 'addition')),
           ('scout_v1 reserve [first use]', (d.set == 'scout_v1') & (d.cohort == 'reserve')),
           ('scout_v1, scout share >= 30%', (d.set == 'scout_v1') & (d.Scout >= .3)),
           ('both sets, AWP-main', d.main == 'AWP'), ('both sets, Scout-main', d.main == 'Scout'),
           ('both sets, Rifle-main', d.main == 'Rifle'), ('both sets, >=10 encounters', d.encounters >= 10),
           ('both sets, <5 encounters', d.encounters < 5), ('both sets, all', d.set.notna())]
for name, m in cohorts:
    x = d[m]
    if not len(x):
        print('empty cohort:', name); continue
    row = {'cohort': name, 'player-maps': len(x), 'people': x.pid.nunique(), 'median score': round(x.cal.median(), 3), 'max': round(x.cal.max(), 3)}
    for k in POL:
        f = int((x.cal >= THRESHOLDS[k]).sum())
        people, n, ci = person_ci(x, THRESHOLDS[k])
        row[k] = f'{f}/{len(x)} = {100 * f / len(x):.1f}% exact{np.round(cp(f, len(x)), 1).tolist()} person-boot{ci}; people {people}/{n}'
    rows.append(row)
print(pd.DataFrame(rows).T.to_string())
REC['P2'] = rows
print('\nFlag rate at `accuracy` by AWP share and by scout share (both sets):')
for col in ['AWP', 'Scout']:
    b = pd.cut(d[col], [-.01, 0, .2, .5, 1.0])
    print(col, d.groupby(b, observed=True).cal.agg(n='size', median='median', flagged=lambda s: (s >= THRESHOLDS['accuracy']).mean()).round(3).T.to_string())

print('\n## P3 Person effects')
multi = d[d.pid.map(d.groupby('pid').match_group.nunique()) >= 2].copy()
multi['z'] = logit(np.clip(multi.cal, 1e-6, 1 - 1e-6))
def icc(x, key):
    g = x.groupby(key).z
    k = g.size().mean(); msb = (g.size() * (g.mean() - x.z.mean()) ** 2).sum() / (g.ngroups - 1)
    msw = ((x.z - g.transform('mean')) ** 2).sum() / (len(x) - g.ngroups)
    return float((msb - msw) / (msb + (k - 1) * msw))
print(f'people with >=2 different series: {multi.pid.nunique()}, player-maps {len(multi)}')
print(f'ICC of logit score by person: {icc(multi, "pid"):.3f}; by player-map main weapon alone: {icc(multi, "main"):.3f}')
res = multi.assign(z=multi.z - multi.groupby('main').z.transform('mean'))
print(f'ICC by person after removing main-weapon means: {icc(res, "pid"):.3f}')
REC['P3_icc'] = {'person': icc(multi, 'pid'), 'main_weapon': icc(multi, 'main'), 'person_after_weapon': icc(res, 'pid')}

# re-identification across different series from window summary features
Z = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(np.nan_to_num(X)))
ok = (d.encounters >= 10).to_numpy()
pidv, grp = d.pid.to_numpy(), d.match_group.to_numpy()
hits = total = 0; chance = []
for i in np.flatnonzero(ok):
    cand = ok & (grp != grp[i])
    if not (cand & (pidv == pidv[i])).any():
        continue
    dist = np.linalg.norm(Z[cand] - Z[i], axis=1)
    hits += pidv[cand][dist.argmin()] == pidv[i]; total += 1
    chance.append((pidv[cand] == pidv[i]).mean())
print(f'1-NN re-identification across different series (>=10 encounters): {hits}/{total} = {hits / total:.3f}; chance {np.mean(chance):.4f}')
REC['P3_reid_features'] = {'hits': int(hits), 'total': int(total), 'chance': float(np.mean(chance))}

# yaw step (mouse sensitivity) as a quasi-identifier
q = d[(d.yaw_nonzero >= 2000)].copy()
q['q'] = q.yaw_q_min.round(4)
print('\nSmallest non-zero unscoped |delta_yaw| per player-map (degrees): distinct values', q.q.nunique(), 'of', len(q), 'player-maps;',
      'most common:', q.q.value_counts().head(6).to_dict())
per = q.groupby('pid').q.agg(['nunique', 'size'])
per = per[per['size'] >= 2]
print(f'people with >=2 player-maps: {len(per)}; same value on all their maps: {(per["nunique"] == 1).mean():.3f}')
share = q.q.map(q.groupby('q').pid.nunique())
print(f'median number of people sharing a value: {share.median():.0f}; player-maps whose value is unique to one person: {(share == 1).mean():.3f}')
REC['P3_yaw_step'] = {'distinct': int(q.q.nunique()), 'n': len(q), 'stable_within_person': float((per['nunique'] == 1).mean()),
                      'unique_to_one_person': float((share == 1).mean())}

print('\n## Channel shift: pro windows in CS2CD-training z units (mean z; 0 = same as CS2CD training mean)')
s = pd.read_csv(CACHE / 'pro_channel_shift_by_match.csv')
rows = []
for f in FEATURES:
    row = {'channel': f}
    for sn in (0, 1):
        x = s[s.sniper == sn]; w = x.n / x.n.sum()
        row[f'mean z (sniper={sn})'] = round(float((x[f'm_{f}'] * w).sum()), 2)
        row[f'rms z (sniper={sn})'] = round(float(np.sqrt((x[f's_{f}'] * w).sum())), 2)
        row[f'zero frac (sniper={sn})'] = round(float((x[f'z_{f}'] * w).sum()), 3)
    rows.append(row)
t = pd.DataFrame(rows)
print(t.reindex(t['mean z (sniper=0)'].abs().sort_values(ascending=False).index).head(14).to_string(index=False))
print('always-zero channels in pro windows:', t[(t['zero frac (sniper=0)'] > .9999)].channel.tolist())
t.to_csv(OUT / 'pro_channel_shift.csv', index=False)
(OUT / '04_pro.json').write_text(json.dumps(REC, indent=1, default=str))
