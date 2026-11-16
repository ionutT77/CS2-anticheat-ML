# =============================================================================
# CS2CD follow-up experiments K1-K4 (see KAGGLE_SPEC.md next to this file)
# =============================================================================
# HOW TO USE ON KAGGLE
#   1. New Notebook, Internet ON, Accelerator: GPU T4 (CPU also works, slower for K1/K3).
#   2. In one cell:
#        !git clone --branch thesis-analysis <your repo url> /kaggle/working/repo
#        %cd /kaggle/working/repo/research
#        !pip -q install lightgbm==4.6.0
#        !python experiments/thesis_analysis_v1/kaggle/cs2cd_followups.py --build-cache --run all
#   3. Download /kaggle/working/followups/ (results.json, results.txt) and bring it back.
#      The window cache is written to /kaggle/working/cs2cd_cache (about 6 GB); save the notebook
#      version and turn that output into a Kaggle dataset to skip --build-cache next time
#      (then pass --cache /kaggle/input/<name>/cs2cd_cache).
#
# Rules this script keeps: the CS2CD test split is never scored or read for metrics; every choice is
# made on training or validation data; the frozen cs2cd_v1 files are read, never written.
# Local smoke test (no CS2CD needed): python cs2cd_followups.py --smoke --run all   (from research/)
# =============================================================================
import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RESEARCH))
V1 = RESEARCH / 'experiments' / 'cs2cd_v1'
SEED = 20261116
LGBM = dict(n_estimators=2000, num_leaves=15, max_depth=-1, learning_rate=.025, min_child_samples=35,
            colsample_bytree=.8, subsample=.8, subsample_freq=1, reg_lambda=10, reg_alpha=.3, n_jobs=4,
            random_state=42, verbosity=-1, metric='None')   # = lgbm39_leaves15 in train_cs2cd_tabular.py


# ----------------------------------------------------------------------------- stage 0: cache
def fetch(info, f, dest):
    """One file of the pinned release, size and digest checked (same checks as download_cs2cd.py)."""
    import requests
    from urllib.parse import quote
    dst = dest / f['path']
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/{info["id"]}/resolve/{info["sha"]}/{quote(f["path"])}'
    for attempt in range(6):
        try:
            sha = hashlib.sha256(); git = hashlib.sha1(f'blob {f["size"]}\0'.encode())
            with requests.get(url, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                with dst.open('wb') as out:
                    for chunk in r.iter_content(4 * 1024 * 1024):
                        out.write(chunk); sha.update(chunk); git.update(chunk)
            assert dst.stat().st_size == f['size']
            assert (sha.hexdigest() == f['lfs']['oid']) if 'lfs' in f else (git.hexdigest() == f['oid'])
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 30))


def build_cache(cache, workers, chunk=12):
    """Download, extract and delete raw matches a few at a time, then write the transformed cache.

    Must run with research/ as the working directory (extract_match writes relative paths)."""
    from anticheat.cs2_data import FEATURES, extract_match, transform
    from anticheat.cs2_features import summarize_events, summarize_player
    meta = RESEARCH / 'data' / 'cs2cd' / 'metadata'
    info = json.loads((meta / 'raw_repo.json').read_text())
    files = [f for f in json.loads((meta / 'raw_files.json').read_text()) if f['path'].endswith(('.json', '.parquet'))]
    by_match = {}
    for f in files:
        by_match.setdefault(f['path'].rsplit('.', 1)[0], []).append(f)
    raw = Path('data/cs2cd/raw'); done_dir = Path('data/cs2cd/processed/matches')
    todo = [m for m in sorted(by_match) if not (done_dir / (m.replace('/', '_') + '.npz')).exists()]
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        with cf.ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda f: fetch(info, f, raw), [f for m in part for f in by_match[m]]))
        with cf.ProcessPoolExecutor(workers) as pool:
            list(pool.map(extract_match, [str(raw / (m + '.parquet')) for m in part]))
        for m in part:                       # raw files are not needed again
            for f in by_match[m]:
                (raw / f['path']).unlink()
        print(f'extracted {i + len(part)}/{len(todo)} matches; {time.time() - t0:.0f}s', flush=True)
    events = pd.read_parquet(V1 / 'events.parquet')
    players = pd.read_csv(V1 / 'player_index.csv')
    cache.mkdir(parents=True, exist_ok=True)
    n, c = len(events), len(FEATURES)
    xout = np.lib.format.open_memmap(cache / 'x.npy', mode='w+', dtype=np.float16, shape=(n, 256, c))
    fout = np.lib.format.open_memmap(cache / 'event_features.npy', mode='w+', dtype=np.float32, shape=(n, 14 * c))
    bad = 0
    for path, g in events.groupby('npz', sort=False):
        with np.load(path) as d:
            rawx = d['x'][g.local_index.to_numpy()]
        bad += sum(hashlib.sha256(w.tobytes()).hexdigest() != h for w, h in zip(rawx, g.sha256))
        x = transform(rawx)
        xout[g.index.to_numpy()] = x.astype(np.float16); fout[g.index.to_numpy()] = summarize_events(x)
    assert bad == 0, f'{bad} extracted windows differ from the frozen events.parquet hashes'
    groups = {k: g.sort_values('tick').index.to_numpy() for k, g in events.groupby('player_match')}
    bags = [groups[p] for p in players.player_match]
    np.savez(cache / 'bags.npz', starts=np.cumsum([0] + [len(b) for b in bags]), indices=np.concatenate(bags))
    np.save(cache / 'player_features.npy', np.stack([summarize_player(fout[b]) for b in bags]))
    xout.flush(); fout.flush()
    print('cache written; all window hashes match the frozen experiment', flush=True)


# ----------------------------------------------------------------------------- context
def load_context(cache, smoke):
    from scripts.analyze_weapon_groups import ORDER, to_group
    if smoke:
        events, players, x, pf = smoke_data()
    else:
        events = pd.read_parquet(V1 / 'events.parquet'); players = pd.read_csv(V1 / 'player_index.csv')
        x = np.load(cache / 'x.npy', mmap_mode='r'); pf = np.load(cache / 'player_features.npy')
    events['wg'] = to_group(events.weapon).fillna('Other')
    groups = {k: g.sort_values('tick').index.to_numpy() for k, g in events.groupby('player_match')}
    bags = [groups[p] for p in players.player_match]
    shares = pd.crosstab(events.player_match, events.wg, normalize='index')
    players['main'] = shares.idxmax(axis=1).reindex(players.player_match).to_numpy()
    players['reviewed'] = players.source == 'with_cheater_present'
    players['lobby'] = players.match_id.map((players.main == 'Scout').groupby(players.match_id).sum()) >= 3
    players['idx'] = players.match_id.str.split('/').str[1].astype(int)
    return dict(events=events, players=players, x=x, pf=pf, bags=bags, smoke=smoke)


def smoke_data():
    """Tiny stand-in built from local OpenCS2 windows with random labels: exercises code paths only."""
    from anticheat.cs2_data import transform
    from anticheat.cs2_features import summarize_events, summarize_player
    rng = np.random.default_rng(0)
    ev = pd.read_parquet(RESEARCH / 'data/opencs2/awp_v1/reserve_events.parquet')
    keep = sorted(ev.npz.unique())[:30]
    ev = ev[ev.npz.isin(keep)].reset_index(drop=True)
    mid = {m: (f'with_cheater_present/{i}' if i % 2 else f'no_cheater_present/{i}') for i, m in enumerate(sorted(ev.match_id.unique()))}
    split = {m: ['train', 'train', 'train', 'validation', 'calibration'][i % 5] for i, m in enumerate(sorted(ev.match_id.unique()))}
    ev['split'] = ev.match_id.map(split); ev['match_id'] = ev.match_id.map(mid)
    ev['source'] = ev.match_id.str.split('/').str[0]; ev['group'] = ev.match_id
    ev['player_match'] = ev.match_id + '/' + ev.player
    lab = {p: int(p.startswith('with') and rng.random() < .4) for p in ev.player_match.unique()}
    ev['label'] = ev.player_match.map(lab)
    x = np.concatenate([np.load(RESEARCH / p)['x'][g.local_index.to_numpy()] for p, g in ev.groupby('npz', sort=False)])
    order = np.concatenate([g.index.to_numpy() for _, g in ev.groupby('npz', sort=False)])
    xt = np.empty_like(x); xt[order] = transform(x)
    players = ev.groupby('player_match', sort=False).agg(label=('label', 'first'), source=('source', 'first'), split=('split', 'first'),
                                                         match_id=('match_id', 'first'), group=('group', 'first'), encounters=('label', 'size')).reset_index()
    ef = summarize_events(xt)
    pf = np.stack([summarize_player(ef[ev.index[ev.player_match == p]]) for p in players.player_match])
    return ev, players, xt.astype(np.float16), pf


# ----------------------------------------------------------------------------- metrics
def wauc(y, s, w):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s, sample_weight=w))


def weapon_weights(d):
    """Weights that give both labels the pooled main-weapon distribution (groups seen with both labels)."""
    tab = pd.crosstab(d.main, d.label)
    ok = d.main.isin(tab.index[(tab.get(0, 0) > 0) & (tab.get(1, 0) > 0)]).to_numpy()
    dd = d[ok]
    within = dd.groupby(['main', 'label']).label.transform('size') / dd.groupby('label').label.transform('size')
    w = np.zeros(len(d)); w[ok] = (dd.main.map(dd.main.value_counts(normalize=True)) / within).to_numpy()
    return w


def report(players, score, name):
    """Threshold-free metrics on validation and calibration. Test rows are never passed in."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for split in ['validation', 'calibration']:
        m = (players.split == split).to_numpy()
        d, s = players[m], np.asarray(score)[m]
        y = d.label.to_numpy()
        out, o = {'experiment': name, 'split': split, 'n': int(m.sum()), 'pos': int(y.sum())}, (~d.lobby).to_numpy()
        def auc(mask, w=None):
            return round(float(roc_auc_score(y[mask], s[mask], sample_weight=None if w is None else w[mask])), 4) if len(set(y[mask])) == 2 else None
        out['auc_all'] = auc(np.ones(len(d), bool)); out['auc_reviewed'] = auc(d.reviewed.to_numpy())
        out['auc_outside_lobbies'] = auc(o)
        w = weapon_weights(d); out['auc_weapon_balanced'] = auc(w > 0, w)
        if (y[o] == 0).sum() >= 100 and y[o].sum():      # in-sample operating point, outside lobbies
            thr = np.sort(s[o][y[o] == 0])[::-1][int(.01 * (y[o] == 0).sum())]
            out['recall_at_1pct_fpr_outside_lobbies_in_sample'] = round(float((s[o][y[o] == 1] > thr).mean()), 3)
        rows.append(out)
    return rows


# ----------------------------------------------------------------------------- models
def frozen_models(device):
    import joblib, torch
    from anticheat.cs2_models import MatchDetector
    norm = json.loads((V1 / 'normalization.json').read_text())
    ck = torch.load(V1 / 'tcn39_seed123.pt', map_location='cpu', weights_only=True)
    tcn = MatchDetector(ck['config']['channels'], ck['config']['architecture'], ck['config'].get('hierarchical', False)).to(device)
    tcn.load_state_dict(ck['state_dict'])
    return tcn, torch.tensor(norm['mean'], device=device), torch.tensor(norm['std'], device=device), joblib.load(V1 / 'lgbm39_leaves15.joblib')


def tcn_event_logits(C, rows, device, x=None):
    """Per-encounter logits for the given player rows (never test). Returns {player row: logits in bag order}."""
    import torch
    from anticheat.cs2_training import predict_players
    tcn, mean, std, _ = C['frozen']
    xs = C['x'] if x is None else x
    out = {}
    for start in range(0, len(rows), 400):          # chunks keep host and GPU memory bounded
        part = rows[start:start + 400]
        idx = np.concatenate([C['bags'][i] for i in part])
        local = torch.from_numpy(np.ascontiguousarray(xs[np.sort(idx)])).to(device)
        pos = {g: j for j, g in enumerate(np.sort(idx))}
        bags = [np.array([pos[g] for g in C['bags'][i]]) for i in part]
        _, ev, order = predict_players(tcn, local, mean, std, bags, np.arange(len(bags)))
        cuts = np.cumsum([0] + [len(b) for b in bags])
        for j, i in enumerate(part):
            out[i] = ev[cuts[j]:cuts[j + 1]]
    return out


def not_test(C):
    return np.flatnonzero((C['players'].split != 'test').to_numpy())


# ----------------------------------------------------------------------------- K1
def k1_weapon_normalised(C, device):
    """Subtract each weapon group's typical legitimate encounter logit (fitted on train) before averaging."""
    from scipy.special import expit
    P, E = C['players'], C['events']
    rows = not_test(C)
    logits = tcn_event_logits(C, rows, device)
    C['tcn_logits'] = logits
    tr = [i for i in rows if P.split[i] == 'train' and P.label[i] == 0 and not P.lobby[i]]
    ev = pd.DataFrame({'wg': np.concatenate([E.wg.to_numpy()[C['bags'][i]] for i in tr]), 'l': np.concatenate([logits[i] for i in tr])})
    base = ev.l.median()
    offset = {g: float(v.median() - base) for g, v in ev.groupby('wg').l if len(v) >= (5 if C['smoke'] else 200)}
    plain = np.full(len(P), np.nan); adj = np.full(len(P), np.nan)
    for i in rows:
        off = np.array([offset.get(g, 0.) for g in E.wg.to_numpy()[C['bags'][i]]])
        plain[i] = logits[i].mean(); adj[i] = (logits[i] - off).mean()
    _, _, _, lg = C['frozen']
    lgb = np.full(len(P), np.nan); lgb[rows] = lg['model'].predict_proba(C['pf'][rows][:, lg['feature_indices']])[:, 1]
    C['plain_tcn'], C['lgbm'] = plain, lgb
    res = report(P, plain, 'K1 baseline: frozen TCN') + report(P, adj, 'K1 weapon-normalised TCN')
    res += report(P, .75 * expit(plain) + .25 * lgb, 'K1 baseline: frozen blend') + report(P, .75 * expit(adj) + .25 * lgb, 'K1 weapon-normalised blend')
    if not C['smoke']:   # parity gate with the saved validation scores of the frozen experiment
        va = np.load(V1 / 'tcn39_seed123_validation.npz')
        diff = float(np.abs(expit(plain[va['player_index']]) - va['scores']).max())
        print('parity with saved TCN validation scores, max |diff|:', diff)
        assert diff < 5e-3, 'rebuilt cache does not reproduce the frozen validation scores'
    return {'offsets_logit': offset, 'metrics': res}


# ----------------------------------------------------------------------------- K2
def k2_blocked_split(C):
    """Does a split by contiguous match-number blocks score lower than a random match split?

    Pool = original train + validation matches only. Five rotations of each scheme, same model config."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    P = C['players']
    pool = P[P.split.isin(['train', 'validation'])]
    matches = pool.drop_duplicates('match_id')[['match_id', 'source', 'idx']]
    rng = np.random.default_rng(SEED)
    feats = np.flatnonzero(np.arange(C['pf'].shape[1]) % 39 < 39)
    params = dict(LGBM, n_estimators=60) if C['smoke'] else LGBM
    out = []
    for scheme in ['random', 'blocked']:
        for k in range(5):
            val = set()
            for _, g in matches.groupby('source'):
                g = g.sort_values('idx'); n = len(g); size = max(1, n // 5)
                if scheme == 'random':
                    val |= set(rng.choice(g.match_id.to_numpy(), size, replace=False))
                else:
                    val |= set(g.match_id.to_numpy()[k * size:(k + 1) * size])
            # buffer: in the blocked scheme drop training matches within 5 numbers of a validation match
            vi = matches[matches.match_id.isin(val)]
            near = set()
            if scheme == 'blocked':
                for src, g in matches.groupby('source'):
                    v = vi.idx[vi.source == src].to_numpy()
                    if len(v):
                        near |= set(g.match_id[(np.abs(g.idx.to_numpy()[:, None] - v[None]).min(1) <= 5)])
            va = pool.index[pool.match_id.isin(val)].to_numpy()
            tr = pool.index[~pool.match_id.isin(val | near)].to_numpy()
            y = P.label.to_numpy()
            if len(set(y[va])) < 2:
                continue
            m = lgb.LGBMClassifier(**params)
            m.fit(C['pf'][tr][:, feats], y[tr], eval_set=[(C['pf'][va][:, feats], y[va])],
                  eval_metric=lambda yt, pr: ('auc', roc_auc_score(yt, pr), True),
                  callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
            s = m.predict_proba(C['pf'][va][:, feats])[:, 1]
            d = P.loc[va]; o = (~d.lobby).to_numpy(); r = d.reviewed.to_numpy()
            a = lambda mask: round(float(roc_auc_score(y[va][mask], s[mask])), 4) if len(set(y[va][mask])) == 2 else None
            out.append({'scheme': scheme, 'fold': k, 'train_players': len(tr), 'val_players': len(va), 'val_pos': int(y[va].sum()),
                        'auc_all': a(np.ones(len(va), bool)), 'auc_reviewed': a(r), 'auc_outside_lobbies': a(o)})
            print(out[-1], flush=True)
    t = pd.DataFrame(out)
    summary = t.groupby('scheme')[['auc_all', 'auc_reviewed', 'auc_outside_lobbies']].agg(['mean', 'std']).round(4)
    print(summary.to_string())
    return {'folds': out, 'summary': json.loads(summary.to_json())}


# ----------------------------------------------------------------------------- K3
def k3_footstep_mask(C, device):
    """Zero the victim_footstep channel (always zero in GOTV pro demos) and re-score validation + calibration."""
    from scipy.special import expit
    from anticheat.cs2_data import FEATURES
    from anticheat.cs2_features import summarize_events, summarize_player
    P = C['players']
    rows = np.flatnonzero(P.split.isin(['validation', 'calibration']).to_numpy())
    ch = FEATURES.index('victim_footstep')
    idx = np.sort(np.concatenate([C['bags'][i] for i in rows]))
    xm = np.array(C['x'][idx]); xm[..., ch] = 0
    pos = np.full(len(C['events']), -1); pos[idx] = np.arange(len(idx))
    class View:                                    # masked windows addressed by global event index
        def __getitem__(self, g):
            return xm[pos[g]]
    logits = tcn_event_logits(C, list(rows), device, x=View())
    ef = summarize_events(xm.astype(np.float32))
    _, _, _, lg = C['frozen']
    tab = np.stack([summarize_player(ef[pos[C['bags'][i]]]) for i in rows])
    masked = np.full(len(P), np.nan)
    masked[rows] = .75 * expit(np.array([logits[i].mean() for i in rows])) + .25 * lg['model'].predict_proba(tab[:, lg['feature_indices']])[:, 1]
    base = .75 * expit(C['plain_tcn']) + .25 * C['lgbm']
    neg = rows[(P.label.to_numpy()[rows] == 0)]
    shift = {'legit median unmasked': float(np.median(base[neg])), 'legit median masked': float(np.median(masked[neg])),
             'legit share >=0.5 unmasked': float((base[neg] >= .5).mean()), 'legit share >=0.5 masked': float((masked[neg] >= .5).mean()),
             'spearman unmasked vs masked': float(pd.Series(base[rows]).corr(pd.Series(masked[rows]), method='spearman'))}
    print(shift)
    return {'shift_raw_blend_score': shift, 'metrics': report(P, base, 'K3 unmasked blend') + report(P, masked, 'K3 footstep channel zeroed')}


# ----------------------------------------------------------------------------- K4
def k4_reweighted_lgbm(C):
    """LightGBM with weapon-neutral weights, suspect unreviewed lobbies removed from training negatives.

    Early stopping and the comparison both use validation; calibration is reported as a second look."""
    import lightgbm as lgb
    P = C['players']; y = P.label.to_numpy()
    feats = np.flatnonzero(np.arange(C['pf'].shape[1]) % 39 < 39)
    tr = np.flatnonzero((P.split == 'train').to_numpy()); va = np.flatnonzero((P.split == 'validation').to_numpy())
    params = dict(LGBM, n_estimators=60) if C['smoke'] else LGBM
    wv = weapon_weights(P.iloc[va])
    balanced = lambda yt, pr: ('weapon_balanced_auc', wauc(yt[wv > 0], pr[wv > 0], wv[wv > 0]), True)
    res, rows = [], not_test(C)
    for name, keep, weights in [
            ('K4 baseline: unweighted, all training rows', tr, None),
            ('K4 lobbies removed from unreviewed negatives', tr[~(P.lobby & ~P.reviewed).to_numpy()[tr]], None),
            ('K4 weapon-neutral weights + lobbies removed', tr[~(P.lobby & ~P.reviewed).to_numpy()[tr]], 'weapon')]:
        d = P.iloc[keep]
        w = None
        if weights:
            prior = d.label.mean(); cond = d.groupby('main').label.transform('mean'); n = d.groupby('main').label.transform('size')
            w = np.where(d.label == 1, prior / cond.clip(lower=1e-3), (1 - prior) / (1 - cond).clip(lower=1e-3))
            w = np.where(n >= (5 if C['smoke'] else 30), w, 1.0); w = np.clip(w, .1, 10)
        m = lgb.LGBMClassifier(**params)
        m.fit(C['pf'][keep][:, feats], y[keep], sample_weight=w, eval_set=[(C['pf'][va][:, feats], y[va])], eval_metric=balanced,
              callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
        s = np.full(len(P), np.nan); s[rows] = m.predict_proba(C['pf'][rows][:, feats])[:, 1]
        res += [dict(r, best_iteration=int(m.best_iteration_ or 0), train_rows=len(keep)) for r in report(P, s, name)]
    return {'metrics': res}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-cache', action='store_true')
    ap.add_argument('--cache', default='/kaggle/working/cs2cd_cache')
    ap.add_argument('--out', default='/kaggle/working/followups')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--run', default='all', help='all or comma list of k1,k2,k3,k4')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out if not a.smoke else 'experiments/thesis_analysis_v1/cache/kaggle_smoke')
    if a.build_cache:
        build_cache(cache, a.workers)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    C = load_context(cache, a.smoke)
    C['frozen'] = frozen_models(device)
    todo = ['k1', 'k2', 'k3', 'k4'] if a.run == 'all' else a.run.split(',')
    results = {'seed': SEED, 'device': device, 'smoke': a.smoke, 'players': len(C['players']),
               'test_rows_scored': 0, 'lgbm_config': LGBM}
    if 'k1' in todo or 'k3' in todo:
        results['K1_weapon_normalised'] = k1_weapon_normalised(C, device)
    if 'k2' in todo:
        results['K2_blocked_split'] = k2_blocked_split(C)
    if 'k3' in todo:
        results['K3_footstep_mask'] = k3_footstep_mask(C, device)
    if 'k4' in todo:
        results['K4_reweighted_lgbm'] = k4_reweighted_lgbm(C)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str))
    table = pd.DataFrame([r for k in results.values() if isinstance(k, dict) and 'metrics' in k for r in k['metrics']])
    (out / 'results.txt').write_text(table.to_string(index=False))
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
