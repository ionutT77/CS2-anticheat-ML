"""Source, weapon, false-positive, threshold and metadata checks on saved cs2cd_v1 scores.

Run from this folder: python 01_cs2cd_hypotheses.py   (see PREREGISTRATION.md; nothing is refitted
except small metadata-only baselines, fitted on the training split).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import OUT, STRICT, THRESHOLDS, CAL, cluster_boot, events, players, apply_calibration
from anticheat.reference_lstm import choose_thresholds
from scripts.analyze_weapon_groups import ORDER, to_group

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
REC = {}
EVAL = ['validation', 'calibration', 'test']
POL = ['accuracy', 'f1', 'default_0_5', 'fpr_1pct', 'fpr_0_1pct']


def cp(k, n):
    return [float(beta.ppf(.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0]


def wauc(y, s, w):
    """Weighted AUC (ties count half)."""
    y, s, w = map(np.asarray, (y, s, w))
    pos, neg = y == 1, y == 0
    sp, sn, wp, wn = s[pos], s[neg], w[pos], w[neg]
    gt = (sp[:, None] > sn[None]) + .5 * (sp[:, None] == sn[None])
    return float((wp[:, None] * wn[None] * gt).sum() / (wp.sum() * wn.sum()))


p = players()
e = events()
e['wg'] = to_group(e.weapon)
shares = pd.crosstab(e.player_match, e.wg, normalize='index').reindex(columns=ORDER, fill_value=0)
p = p.join(shares, on='player_match')
p['main'] = shares.idxmax(axis=1).reindex(p.player_match).to_numpy()
cheaters_in_match = p.groupby('match_id').label.sum()
p['match_cheaters'] = p.match_id.map(cheaters_in_match)

# ---------------- structure of the positive class (labels only, all splits) -----------------
allp = pd.read_csv('../cs2cd_v1/players.csv')
per_match = allp.groupby(['source', 'match_id']).agg(n=('label', 'size'), cheaters=('label', 'sum'))
rev = per_match.loc['with_cheater_present']
print('\n## Cheater-labelled players per reviewed match (all 7,949 player-matches)')
print(rev.cheaters.value_counts().sort_index().to_string())
print('players per match:', per_match.n.value_counts().sort_index().to_dict())
REC['cheaters_per_reviewed_match'] = {int(k): int(v) for k, v in rev.cheaters.value_counts().sort_index().items()}
m = p.groupby('match_cheaters').agg(players=('label', 'size'), positives=('label', 'sum'),
                                    scout_share_pos=('Scout', lambda s: s[p.loc[s.index, 'label'] == 1].mean()))
print('\nEligible players by cheaters-in-match, mean scout share of positives:\n', m.round(3).to_string())
print('\nMaps by source (matches):\n', pd.crosstab(allp.drop_duplicates('match_id')['map'], allp.drop_duplicates('match_id')['source']).to_string())

# ---------------- T1 source ----------------
print('\n## T1 Source folder signal among label-0 players')
rows = []
for s in EVAL:
    d = p[(p.split == s) & (p.label == 0)]
    auc = roc_auc_score(d.reviewed, d.cal)
    ci = cluster_boot(d, lambda x: roc_auc_score(x.reviewed, x.cal))
    row = {'split': s, 'neg_unrev': int((~d.reviewed).sum()), 'neg_rev': int(d.reviewed.sum()),
           'AUC score->reviewed folder': round(auc, 3), 'ci': np.round(ci, 3).tolist(),
           'median unrev': round(d.cal[~d.reviewed].median(), 4), 'median rev': round(d.cal[d.reviewed].median(), 4)}
    for k in ['accuracy', 'default_0_5', 'fpr_1pct']:
        for name, g in [('unrev', d[~d.reviewed]), ('rev', d[d.reviewed])]:
            row[f'FP {k} {name}'] = f'{int((g.cal >= THRESHOLDS[k]).sum())}/{len(g)}'
    rows.append(row)
print(pd.DataFrame(rows).T.to_string())
REC['T1_source_among_negatives'] = rows
rows = []
for s in EVAL:
    d = p[p.split == s]
    r = d[d.reviewed]
    rows.append({'split': s, 'AUC all': round(roc_auc_score(d.label, d.cal), 4),
                 'AUC reviewed only': round(roc_auc_score(r.label, r.cal), 4),
                 'ci reviewed': np.round(cluster_boot(r, lambda x: roc_auc_score(x.label, x.cal)), 3).tolist(),
                 'AUC source-only': round(roc_auc_score(d.label, d.reviewed), 4)})
print(pd.DataFrame(rows).to_string(index=False))
REC['T1_auc_by_cohort'] = rows

# ---------------- T2 within-match ranking ----------------
print('\n## T2 Within-match concordance (cheater vs non-cheater in the same match)')
rows = []
for s in EVAL:
    d = p[(p.split == s) & p.reviewed]
    def within(x):
        c = n = 0.0
        for _, g in x.groupby('match_id'):
            a, b = g.cal[g.label == 1].to_numpy(), g.cal[g.label == 0].to_numpy()
            c += ((a[:, None] > b[None]) + .5 * (a[:, None] == b[None])).sum(); n += len(a) * len(b)
        return c / n
    rows.append({'split': s, 'within-match AUC': round(within(d), 4), 'ci': np.round(cluster_boot(d, within), 3).tolist(),
                 'matches': d.match_id.nunique()})
print(pd.DataFrame(rows).to_string(index=False))
REC['T2_within_match'] = rows

# ---------------- T3 weapon post-stratification ----------------
print('\n## T3 Weapon: within-group AUC and post-stratified metrics (main weapon group)')
rows = []
for s in EVAL + ['val+cal']:
    d = p[p.split.isin(['validation', 'calibration'])] if s == 'val+cal' else p[p.split == s]
    y = d.label.to_numpy()
    tab = pd.crosstab(d.main, d.label)
    both = tab.index[(tab[0] > 0) & (tab[1] > 0)]
    dd = d[d.main.isin(both)]
    yy = dd.label.to_numpy()
    # weight each player so both labels share the pooled main-group distribution
    pooled = dd.main.value_counts(normalize=True)
    within = dd.groupby(['main', 'label']).label.transform('size') / dd.groupby('label').label.transform('size')
    w = (dd.main.map(pooled) / within).to_numpy()
    flag = (dd.cal >= STRICT).to_numpy()
    row = {'split': s, 'n': len(d), 'AUC raw': round(roc_auc_score(y, d.cal), 4),
           'AUC weapon-balanced': round(wauc(yy, dd.cal, w), 4),
           'strict recall raw': round(float(((d.cal >= STRICT) & (y == 1)).sum() / y.sum()), 3),
           'strict recall balanced': round(float((w * flag * (yy == 1)).sum() / (w * (yy == 1)).sum()), 3),
           'strict FPR raw %': round(100 * float(((d.cal >= STRICT) & (y == 0)).sum() / (y == 0).sum()), 2),
           'strict FPR balanced %': round(100 * float((w * flag * (yy == 0)).sum() / (w * (yy == 0)).sum()), 2),
           'AUC weapon-shares-only(scout share)': round(roc_auc_score(y, d.Scout), 4)}
    for g in ['Scout', 'AWP', 'Rifle']:
        x = d[d.main == g]
        row[f'{g}: n neg/pos'] = f'{(x.label == 0).sum()}/{(x.label == 1).sum()}'
        row[f'{g}: AUC'] = round(roc_auc_score(x.label, x.cal), 3)
        row[f'{g}: strict recall'] = round(float((x.cal[x.label == 1] >= STRICT).mean()), 3)
        row[f'{g}: strict FP'] = f'{int((x.cal[x.label == 0] >= STRICT).sum())}/{(x.label == 0).sum()}'
    x = d[d.main != 'Scout']
    row['non-scout-main: AUC'] = round(roc_auc_score(x.label, x.cal), 4)
    row['non-scout-main: strict recall'] = round(float((x.cal[x.label == 1] >= STRICT).mean()), 3)
    row['non-scout-main: positives'] = int(x.label.sum())
    x = d[d.Scout == 0]
    row['zero-scout players: AUC'] = round(roc_auc_score(x.label, x.cal), 4)
    row['zero-scout: strict recall'] = f'{int((x.cal[x.label == 1] >= STRICT).sum())}/{int(x.label.sum())}'
    row['zero-scout: strict FP'] = f'{int((x.cal[x.label == 0] >= STRICT).sum())}/{int((x.label == 0).sum())}'
    rows.append(row)
print(pd.DataFrame(rows).T.to_string())
REC['T3_weapon'] = rows

# ---------------- T6 strict false positives ----------------
print('\n## T6 Label-0 players at or above the strict threshold, all scored splits')
fp = p[(p.label == 0) & (p.cal >= STRICT) & p.split.isin(EVAL)]
cols = ['split', 'match_id', 'map', 'encounters', 'main', 'Scout', 'cal', 'tcn', 'lgbm']
print(fp.sort_values(['split', 'match_id'])[cols].round(3).to_string(index=False))
REC['T6_strict_fp'] = fp[cols].round(4).to_dict('records')
print('\nMatches containing them, all eligible players (label, calibrated score, main, scout share):')
for mid in fp.match_id.unique():
    g = p[p.match_id == mid]
    print(f'  {mid} [{g["split"].iloc[0]}, {g["map"].iloc[0]}] ' + ' | '.join(
        f'{int(l)}:{c:.2f}:{mn}:{sc:.1f}:{n}' for l, c, mn, sc, n in zip(g.label, g.cal, g.main, g.Scout, g.encounters)))
print('\nUnreviewed matches: how many label-0 players are at or above 0.5 (calibrated) in the same match?')
u = p[~p.reviewed & p.split.isin(EVAL)]
hi = u.assign(h=u.cal >= .5).groupby(['split', 'match_id']).agg(h=('h', 'sum'), n=('h', 'size'), scout=('Scout', 'mean'))
print(hi.groupby('split').h.value_counts().unstack(fill_value=0).to_string())
REC['unreviewed_matches_by_count_over_0.5'] = {s: {int(k): int(v) for k, v in g.h.value_counts().items()} for s, g in hi.groupby('split')}
# is clustering stronger than chance? permute high-flags among unreviewed players
rng = np.random.default_rng(20261116)
obs = (hi.h >= 2).sum(); h = (u.cal >= .5).to_numpy(); mids = u.match_id.to_numpy()
perm = [(pd.Series(rng.permutation(h)).groupby(mids).sum() >= 2).sum() for _ in range(2000)]
print(f'unreviewed matches with >=2 such players: observed {obs}, permutation mean {np.mean(perm):.1f}, '
      f'p={np.mean(np.array(perm) >= obs):.4f}')
REC['clustering_test'] = {'observed': int(obs), 'perm_mean': float(np.mean(perm)), 'p': float(np.mean(np.array(perm) >= obs))}
print('scout-heavy unreviewed matches (mean scout share >= .3) by split:',
      hi[hi.scout >= .3].groupby('split').size().to_dict(), 'of', hi.groupby('split').size().to_dict())

# ---------------- T7 thresholds ----------------
print('\n## T7 Operating points with match-cluster and exact intervals')
rows = []
for s in EVAL:
    d = p[p.split == s]
    neg, pos = d[d.label == 0], d[d.label == 1]
    for k in POL:
        t = THRESHOLDS[k]
        f, r = int((neg.cal >= t).sum()), int((pos.cal >= t).sum())
        rows.append({'split': s, 'policy': k, 'FP': f'{f}/{len(neg)}', 'FPR %': round(100 * f / len(neg), 2),
                     'FPR exact95 %': np.round(100 * np.array(cp(f, len(neg))), 2).tolist(),
                     'FPR cluster95 %': np.round(100 * np.array(cluster_boot(d, lambda x: ((x.cal >= t) & (x.label == 0)).sum() / (x.label == 0).sum())), 2).tolist(),
                     'recall': round(r / len(pos), 3),
                     'recall cluster95': np.round(cluster_boot(d, lambda x: ((x.cal >= t) & (x.label == 1)).sum() / (x.label == 1).sum()), 3).tolist(),
                     'precision': round(r / max(r + f, 1), 3)})
print(pd.DataFrame(rows).to_string(index=False))
REC['T7_operating_points'] = rows
print('\nSmallest FPR distinguishable from zero (95% upper bound when 0 FP observed = 1-0.05^(1/n)):')
for name, n in [('calibration', 625), ('test', 916), ('val+cal+test', int((p[p.split.isin(EVAL)].label == 0).sum())),
                ('reviewed negatives val+cal+test', int(((p.label == 0) & p.reviewed & p.split.isin(EVAL)).sum()))]:
    print(f'  {name}: n={n}, zero-FP upper bound {100 * (1 - .05 ** (1 / n)):.2f}%; '
          f'negatives needed to bound FPR<0.1%: {int(np.ceil(np.log(.05) / np.log(1 - .001)))}')
# strict-threshold instability: re-derive it on resampled calibration matches, apply to test
c, t = p[p.split == 'calibration'], p[p.split == 'test']
parts = [g for _, g in c.groupby('group')]
out = []
for _ in range(2000):
    b = pd.concat([parts[i] for i in rng.integers(len(parts), size=len(parts))])
    thr = choose_thresholds(b.label.to_numpy(), b.cal.to_numpy())['fpr_1pct']
    out.append((thr, ((t.cal >= thr) & (t.label == 0)).sum() / (t.label == 0).sum(), ((t.cal >= thr) & (t.label == 1)).sum() / t.label.sum()))
out = np.array(out)
q = lambda a: np.round(np.quantile(a, [.025, .5, .975]), 4).tolist()
print(f'Strict threshold over calibration resamples: {q(out[:, 0])}; test FPR {q(out[:, 1])}; test recall {q(out[:, 2])}')
REC['T7_threshold_resampling'] = {'threshold_q': q(out[:, 0]), 'test_fpr_q': q(out[:, 1]), 'test_recall_q': q(out[:, 2])}
# calibration quality of the Platt map
print('\nReliability of the calibrated score (bins by score):')
for s in EVAL:
    d = p[p.split == s]
    b = pd.cut(d.cal, [0, .05, .1, .2, .4, .6, .8, 1.0])
    print(s, d.groupby(b, observed=True).agg(n=('label', 'size'), mean_score=('cal', 'mean'), observed=('label', 'mean')).round(3).T.to_string())

# ---------------- T8 distributions ----------------
print('\n## T8 Positive-label score distribution, encounters, cheaters per match')
for s in EVAL:
    d = p[(p.split == s) & (p.label == 1)]
    print(s, 'positives by calibrated-score bin:', pd.cut(d.cal, [0, .1, .25, .5, .75, .85, 1.0]).value_counts(sort=False).to_list(),
          '| by main: ', d.groupby('main').cal.agg(['size', 'median']).round(2).query('size>=8').to_dict('index'))
rows = []
for s in ['train'] + EVAL:
    d = p[p.split == s]
    row = {'split': s, 'AUC encounters-only': round(roc_auc_score(d.label, d.encounters), 3),
           'median enc pos/neg': f'{d.encounters[d.label == 1].median():.0f}/{d.encounters[d.label == 0].median():.0f}'}
    if s != 'train':
        r = d[d.reviewed & (d.label == 1)]
        row['pos median score by match cheaters 1 / 2-3 / 4+'] = ' / '.join(
            f'{r.cal[m_].median():.2f} (n={m_.sum()})' for m_ in [r.match_cheaters == 1, r.match_cheaters.between(2, 3), r.match_cheaters >= 4])
        row['strict recall by match cheaters 1 / 2-3 / 4+'] = ' / '.join(
            f'{(r.cal[m_] >= STRICT).mean():.2f}' for m_ in [r.match_cheaters == 1, r.match_cheaters.between(2, 3), r.match_cheaters >= 4])
        row['AUC by encounters <5 / 5-14 / 15+'] = ' / '.join(
            f'{roc_auc_score(d.label[m_], d.cal[m_]):.3f} (n={m_.sum()})' for m_ in [d.encounters < 5, d.encounters.between(5, 14), d.encounters >= 15])
    rows.append(row)
print(pd.DataFrame(rows).T.to_string())
REC['T8'] = rows

# ---------------- metadata-only baselines (fit on train; test not used) ----------------
print('\n## Metadata-only baselines, fitted on the training split (no window features)')
tr = p[p.split == 'train']
feats = {'weapon shares': ORDER, 'weapon shares + log encounters': ORDER + ['loge'],
         'weapon shares + log encounters + match cheaters (LEAKY: uses labels of others)': ORDER + ['loge', 'others']}
p['loge'] = np.log1p(p.encounters); p['others'] = p.match_cheaters - p.label
tr = p[p.split == 'train']
rows = []
for name, f in feats.items():
    mdl = LogisticRegression(C=10, max_iter=2000).fit(tr[f], tr.label)
    row = {'features': name}
    for s in ['validation', 'calibration']:
        d = p[p.split == s]
        pr = mdl.predict_proba(d[f])[:, 1]
        row[f'{s} AUC'] = round(roc_auc_score(d.label, pr), 4)
        row[f'{s} AUC reviewed'] = round(roc_auc_score(d.label[d.reviewed], pr[d.reviewed]), 4)
        if name == 'weapon shares + log encounters':
            p.loc[d.index, 'meta'] = pr
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
REC['metadata_baselines'] = rows
# does the detector add to metadata, and vice versa? partial check on val+cal
d = p[p.split.isin(['validation', 'calibration'])]
print('val+cal: detector AUC', round(roc_auc_score(d.label, d.cal), 4), '| metadata AUC', round(roc_auc_score(d.label, d.meta), 4),
      '| Spearman(detector, metadata)', round(d[['cal', 'meta']].corr(method='spearman').iloc[0, 1], 3))

(OUT / '01_cs2cd_hypotheses.json').write_text(json.dumps(REC, indent=1, default=str))
p[p.split.isin(EVAL)][['player_match', 'split', 'label', 'source', 'map', 'encounters', 'main', 'Scout', 'AWP', 'match_cheaters', 'selected', 'cal']].to_csv(OUT / 'scored_players.csv', index=False)
