"""Weapon-group counts, false-positive rates and pro-demo yield (read-only).

Reproduces the tables in research/opencs2/OTHER_WEAPONS.md. Run from `research/`:
    python scripts/analyze_weapon_groups.py
Uses saved scores only: the validation score files, test_predictions.csv and the test encounter
scores, with the Platt calibrator and thresholds frozen on the calibration split. No detector is
run or refitted. The only model fitted here is a small weapon-share baseline, on the training
split. An encounter is the first hit of an attacker on a victim in a burst; labels are
player-match labels.
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anticheat.reference_lstm import apply_calibration

ROOT = Path('experiments/cs2cd_v1')
PRO = Path('data/opencs2/awp_v1')
METADATA = 'D:/opencs2/metadata/*.parquet'
# CS2CD and our extractor take weapon names from player_hurt, which reports the USP-S as
# hkp2000 and the M4A1-S as m4a1. The OpenCS2 kill metadata keeps usp_silencer separate.
GROUP = {
    'awp': 'AWP', 'ssg08': 'Scout', 'scar20': 'Auto-sniper', 'g3sg1': 'Auto-sniper',
    'aug': 'Scoped rifle', 'sg556': 'Scoped rifle',
    'ak47': 'Rifle', 'm4a1': 'Rifle', 'm4a1_silencer': 'Rifle', 'galilar': 'Rifle', 'famas': 'Rifle',
    'negev': 'Machine gun', 'm249': 'Machine gun',
    'mp9': 'SMG', 'mac10': 'SMG', 'p90': 'SMG', 'mp7': 'SMG', 'mp5sd': 'SMG', 'bizon': 'SMG', 'ump45': 'SMG',
    'xm1014': 'Shotgun', 'nova': 'Shotgun', 'mag7': 'Shotgun', 'sawedoff': 'Shotgun',
    'hkp2000': 'USP-S/P2000', 'usp_silencer': 'USP-S/P2000', 'deagle': 'Deagle/R8', 'revolver': 'Deagle/R8',
    'glock': 'Other pistol', 'tec9': 'Other pistol', 'p250': 'Other pistol', 'fiveseven': 'Other pistol',
    'elite': 'Other pistol', 'cz75a': 'Other pistol'}
ORDER = ['AWP', 'Scout', 'Auto-sniper', 'Scoped rifle', 'Rifle', 'Machine gun', 'SMG', 'Shotgun',
         'USP-S/P2000', 'Deagle/R8', 'Other pistol']
PISTOLS = ['USP-S/P2000', 'Deagle/R8', 'Other pistol']
POLICIES = ['accuracy', 'f1', 'fpr_1pct']
SPLITS = ['train', 'validation', 'calibration', 'test']
FAIL_RATE = 10 / 127  # demos that failed extraction so far


def to_group(weapons):
    return weapons.astype(str).str.removeprefix('weapon_').str.lower().map(GROUP)


def cp(k, n):
    """Exact (Clopper-Pearson) 95% interval for k of n."""
    lo = beta.ppf(.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def rate(k, n):
    if not n:
        return '—'
    lo, hi = cp(k, n)
    return f'{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.1f}–{100 * hi:.1f}]'


def encounter_table(events):
    print('\n## 1. CS2CD encounters by weapon group (cheating / not cheating)')
    t = events.groupby(['group', 'split', 'label']).size().unstack(['split', 'label'], fill_value=0)
    pistol = t.loc[PISTOLS].sum().rename('Pistol (all)')
    t = pd.concat([t.reindex(ORDER), pistol.to_frame().T]).fillna(0).astype(int)
    rows = []
    for g, r in t.iterrows():
        cells = [f"{r.get((s, 1), 0):,} / {r.get((s, 0), 0):,}" for s in SPLITS]
        rows.append([g] + cells)
    print(pd.DataFrame(rows, columns=['group'] + SPLITS).to_string(index=False))
    train = t[[c for c in t.columns if c[0] == 'train']].droplevel(0, axis=1)
    return train


def player_table(players, shares):
    print('\n## 1b. Player-matches by main weapon group (cheating / not cheating)')
    d = players.assign(main=shares.idxmax(axis=1).to_numpy())
    t = d.groupby(['main', 'split', 'label']).size().unstack(['split', 'label'], fill_value=0).reindex(ORDER).fillna(0).astype(int)
    rows = [[g] + [f"{r.get((s, 1), 0)} / {r.get((s, 0), 0)}" for s in SPLITS] for g, r in t.iterrows()]
    print(pd.DataFrame(rows, columns=['main group'] + SPLITS).to_string(index=False))


def load_scores(players, calibration, selection):
    """Raw blend and calibrated score for every validation and test player-match."""
    weights = selection['components']
    val = {name: np.load(ROOT / f'{name}_validation.npz') for name in weights}
    index = next(iter(val.values()))['player_index']
    assert all(np.array_equal(v['player_index'], index) for v in val.values())
    raw_val = sum(w * val[name]['scores'] for name, w in weights.items())
    src_val = np.load(ROOT / 'diagnostic_source_validation.npz')
    assert np.array_equal(src_val['player_index'], index)
    test = pd.read_csv(ROOT / 'test_predictions.csv')
    test_index = players.reset_index().set_index('player_match').loc[test.player_match, 'index'].to_numpy()
    out = pd.concat([
        players.iloc[index].assign(raw=raw_val, source_only=src_val['scores']),
        players.iloc[test_index].assign(raw=test.selected.to_numpy(), source_only=test.diagnostic_source.to_numpy())])
    out['score'] = apply_calibration(out.raw.to_numpy(), calibration['platt'])
    return out


def check_frozen_result(scored, thresholds):
    """The strict test result in METHODS.md must come out of these scores unchanged."""
    t = scored[scored.split == 'test']
    flagged = t.score >= thresholds['fpr_1pct']
    tp, fp = int((flagged & (t.label == 1)).sum()), int((flagged & (t.label == 0)).sum())
    print(f'\nCheck: test strict policy TP {tp}, FP {fp} (published: 67, 4)')
    assert (tp, fp) == (67, 4)


def fpr_by_main_group(scored, thresholds):
    print('\n## 2. False-positive rate and recall by main weapon group (validation + test)')
    print('Thresholds frozen on the calibration split:', {k: round(thresholds[k], 3) for k in POLICIES})
    rows = []
    for g in ORDER + ['Pistol (all)']:
        d = scored[scored.main.isin(PISTOLS)] if g == 'Pistol (all)' else scored[scored.main == g]
        legit, cheat = d[d.label == 0], d[d.label == 1]
        if not len(d):
            continue
        row = {'main group': g, 'legit (val/test)': f'{(legit.split == "validation").sum()}/{(legit.split == "test").sum()}'}
        for k in POLICIES:
            row[f'FPR {k}'] = rate(int((legit.score >= thresholds[k]).sum()), len(legit))
        held_out = legit[legit.split == 'test']   # validation also chose epochs and blend weights
        for k in ['accuracy', 'fpr_1pct']:
            row[f'test only {k}'] = rate(int((held_out.score >= thresholds[k]).sum()), len(held_out))
        row['cheaters'] = len(cheat)
        row['recall f1 / strict'] = (f'{(cheat.score >= thresholds["f1"]).mean():.2f} / '
                                     f'{(cheat.score >= thresholds["fpr_1pct"]).mean():.2f}') if len(cheat) else '—'
        precision = []
        for k in ['f1', 'fpr_1pct']:
            tp, fp = int((cheat.score >= thresholds[k]).sum()), int((legit.score >= thresholds[k]).sum())
            precision.append(f'{tp}/{tp + fp}')
        row['precision (TP/flagged) f1 / strict'] = ' / '.join(precision)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    for s in ['validation', 'test']:
        d = scored[scored.split == s]
        k = 'fpr_1pct'
        legit = d[d.label == 0]
        print(f'{s}: all legitimate FPR at {k} {rate(int((legit.score >= thresholds[k]).sum()), len(legit))}')


def share_analysis(scored, thresholds):
    print('\n## 3a. Legitimate players: weapon share vs calibrated score (validation + test)')
    legit = scored[scored.label == 0]
    rows = []
    for g in ['Scout', 'AWP', 'USP-S/P2000', 'Deagle/R8', 'Other pistol', 'Rifle', 'SMG']:
        rho, p = spearmanr(legit[g], legit.score)
        rows.append({'share of': g, 'spearman rho': round(rho, 3), 'p': f'{p:.1e}'})
    print(pd.DataFrame(rows).to_string(index=False))
    print('\n## 3b. Legitimate players binned by weapon share (bin edges fixed before the first run)')
    bins = [-.001, 0, .1, .2, .35, 1]
    labels = ['0', '0–10%', '10–20%', '20–35%', '>35%']
    for g in ['USP-S/P2000', 'Deagle/R8', 'Other pistol', 'Scout']:
        b = pd.cut(legit[g], bins, labels=labels)
        rows = []
        for lab in labels:
            d = legit[b == lab]
            rows.append({'share': lab, 'players': len(d),
                         'median score': round(float(d.score.median()), 4) if len(d) else None,
                         'FPR accuracy': rate(int((d.score >= thresholds['accuracy']).sum()), len(d)),
                         'FPR strict': rate(int((d.score >= thresholds['fpr_1pct']).sum()), len(d)),
                         'test only, accuracy': rate(int((d[d.split == 'test'].score >= thresholds['accuracy']).sum()),
                                                     int((d.split == 'test').sum()))})
        print(f'\n{g}\n' + pd.DataFrame(rows).to_string(index=False))


def event_scores(events):
    print('\n## 3c. Test encounters: event-head score of the CNN by weapon group '
          '(player-match labels copied to encounters; encounters are clustered by player)')
    z = np.load(ROOT / 'tcn39_seed123_test_events.npz')
    e = events.iloc[z['event_index']].assign(event_score=z['scores'])
    rows = []
    for g in ORDER:
        d = e[e.group == g]
        legit, cheat = d[d.label == 0], d[d.label == 1]
        if len(legit) < 20:
            continue
        rows.append({'group': g, 'legit enc.': len(legit), 'legit median': round(float(legit.event_score.median()), 3),
                     'legit >0.5': f'{(legit.event_score > .5).mean():.1%}', 'cheat enc.': len(cheat),
                     'cheat median': round(float(cheat.event_score.median()), 3) if len(cheat) else None,
                     'AUC in group': round(roc_auc_score(d.label, d.event_score), 3) if 0 < len(cheat) < len(d) else None})
    print(pd.DataFrame(rows).to_string(index=False))


def baselines(players, shares, scored):
    print('\n## 3d. Weapon-only and source-only baselines (AUC, player-matches)')
    x = shares.to_numpy()
    train = (players.split == 'train').to_numpy()
    model = LogisticRegression(max_iter=2000).fit(x[train], players.label[train])
    weapon_only = pd.Series(model.predict_proba(x)[:, 1], index=players.player_match)
    rows = []
    for s in ['validation', 'test']:
        d = scored[scored.split == s]
        w = weapon_only.loc[d.player_match].to_numpy()
        row = {'split': s, 'detector': roc_auc_score(d.label, d.raw), 'weapon-only': roc_auc_score(d.label, w),
               'source-only': roc_auc_score(d.label, d.source_only)}
        pistol = (d['Pistol (all)'] >= .2).to_numpy()
        if 0 < d.label[pistol].sum() < pistol.sum():
            row['detector, pistol share >=20%'] = roc_auc_score(d.label[pistol], d.raw[pistol])
            row['weapon-only, pistol share >=20%'] = roc_auc_score(d.label[pistol], w[pistol])
            row['n pistol>=20% (cheat)'] = f'{pistol.sum()} ({int(d.label[pistol].sum())})'
        rows.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print(pd.DataFrame(rows).to_string(index=False))
    coef = pd.Series(model.coef_[0], index=shares.columns).round(2)
    print('weapon-only coefficients (positive = more cheating):', coef.to_dict())
    imp = pd.read_csv(ROOT / 'tree_channel_importance.csv').set_index('channel').gain_fraction
    keep = ['shots_in_burst', 'weapon_pistol', 'weapon_sniper', 'scoped', 'recoil_index', 'punch_pitch',
            'punch_yaw', 'ammo']
    print('LightGBM gain fraction (of 1.0):', {k: round(float(imp.get(k, np.nan)), 4) for k in keep},
          f'rank of shots_in_burst: {list(imp.sort_values(ascending=False).index).index("shots_in_burst") + 1}')


def pro_yield(train_counts):
    print('\n## 4–5. Gap and pro-demo yield (estimates)')
    manifest = json.loads((PRO / 'manifest.json').read_text())
    processed = {d['file_name'] for d in manifest['demos']}
    usable = {d['file_name'] for d in manifest['demos'] if d.get('encounters')}
    cols = ['file_name', 'demo_bytes', 'kills']
    rows = [r for f in sorted(glob.glob(METADATA)) for r in pq.read_table(f, columns=cols).to_pylist()]
    usp_names = Counter()

    def kills(row):
        c = Counter()
        for k in row['kills'] or []:
            g = GROUP.get(str(k['weapon']))
            if g:
                c[g] += 1
            if g == 'USP-S/P2000':
                usp_names[k['weapon']] += 1
        return c
    per_demo = {r['file_name']: (kills(r), r['demo_bytes']) for r in rows}
    staged = pd.concat([pd.read_parquet(PRO / f, columns=['weapon']) for f in ['events.parquet', 'reserve_events.parquet']])
    staged_enc = to_group(staged.weapon).value_counts()
    used_kills = sum((per_demo[f][0] for f in usable if f in per_demo), Counter())
    print(f'metadata demos {len(per_demo)}, processed {len(processed)}, usable {len(usable)}; '
          f'USP-group kill names in metadata {dict(usp_names)}')
    remaining = {f: v for f, v in per_demo.items() if f not in processed}
    share = train_counts[1].sum() / train_counts.to_numpy().sum()
    print(f'training cheating share over all weapons: {share:.3f}. "1:1" = as many not-cheating as cheating '
          f'encounters; "neutral" = every group at the overall share, so the weapon alone says nothing.')

    def cover(enc, gb, need):
        if not need:
            return '0'
        if enc[-1] < need:
            return 'not enough'
        n = int(np.searchsorted(enc, need) + 1)
        return f'{n} ({gb[n - 1]:.0f} GB)'
    out = []
    for g in ORDER:
        cheat, legit = int(train_counts.loc[g].get(1, 0)), int(train_counts.loc[g].get(0, 0))
        have = int(staged_enc.get(g, 0))
        ratio = have / used_kills[g] if used_kills[g] else np.nan
        need = max(cheat - legit - have, 0)
        neutral = max(int(round(cheat * (1 - share) / share)) - legit - have, 0)
        ranked = sorted(remaining.values(), key=lambda v: -v[0][g])
        enc = np.cumsum([v[0][g] * ratio * (1 - FAIL_RATE) for v in ranked]) if ratio == ratio else np.zeros(1)
        gb = np.cumsum([v[1] / 1e9 for v in ranked])
        out.append({'group': g, 'cheat share': f'{cheat / max(cheat + legit, 1):.0%}', 'staged pro enc.': have,
                    'needed 1:1': need, 'needed neutral': neutral,
                    'enc./kill': round(ratio, 2) if ratio == ratio else None,
                    'remaining enc. (est.)': int(enc[-1]) if ratio == ratio else None,
                    'demos for 1:1': cover(enc, gb, need), 'demos for neutral': cover(enc, gb, neutral)})
    print(pd.DataFrame(out).to_string(index=False))
    ratios = {r['group']: r['enc./kill'] for r in out}
    for n in (100, 200):
        top = sorted(remaining.values(), key=lambda v: -v[0]['Scout'])[:n]
        extra = {g: int(sum(v[0][g] for v in top) * ratios[g] * (1 - FAIL_RATE))
                 for g in ['Scout', 'AWP', 'Deagle/R8', 'USP-S/P2000'] if ratios[g]}
        print(f'best {n} scout demos ({sum(v[1] for v in top) / 1e9:.0f} GB) would also bring (est.): {extra}')


def main():
    events = pd.read_parquet(ROOT / 'events.parquet', columns=['player_match', 'split', 'label', 'weapon'])
    events['group'] = to_group(events.weapon)
    unmapped = events.group.isna().sum()
    print(f'CS2CD encounters {len(events):,}; without a weapon group: {unmapped}')
    assert unmapped == 0, 'add the new weapon names to GROUP; idxmax would misplace these players'
    players = pd.read_csv(ROOT / 'player_index.csv')
    shares = (pd.crosstab(events.player_match, events.group, normalize='index')
              .reindex(index=players.player_match, columns=ORDER).fillna(0))
    calibration = json.loads((ROOT / 'calibration.json').read_text())['models']['selected']
    selection = json.loads((ROOT / 'selection.json').read_text())
    thresholds = calibration['thresholds']

    train_counts = encounter_table(events)
    player_table(players, shares)
    scored = load_scores(players, calibration, selection)
    sh = shares.loc[scored.player_match].reset_index(drop=True)
    sh['Pistol (all)'] = sh[PISTOLS].sum(axis=1)
    scored = pd.concat([scored.reset_index(drop=True), sh], axis=1)
    scored['main'] = sh[ORDER].idxmax(axis=1)
    check_frozen_result(scored, thresholds)
    fpr_by_main_group(scored, thresholds)
    share_analysis(scored, thresholds)
    event_scores(events)
    baselines(players, shares, scored)
    pro_yield(train_counts)


if __name__ == '__main__':
    main()
