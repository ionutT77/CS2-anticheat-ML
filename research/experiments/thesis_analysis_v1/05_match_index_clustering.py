"""Do similar matches sit next to each other in the publisher's numbering? (labels, weapons, maps only)

If neighbouring match numbers share players (e.g. collected from one account's match history), a random
match split does not separate people. Run from this folder: python 05_match_index_clustering.py
"""
import json

import numpy as np
import pandas as pd

from common import OUT, events, players
from scripts.analyze_weapon_groups import ORDER, to_group

rng = np.random.default_rng(20261116)
p, e = players(), events()
e['wg'] = to_group(e.weapon)
shares = pd.crosstab(e.player_match, e.wg, normalize='index').reindex(columns=ORDER, fill_value=0)
p['scout_main'] = (shares.idxmax(axis=1).reindex(p.player_match) == 'Scout').to_numpy()
m = p.groupby('match_id').agg(source=('source', 'first'), split=('split', 'first'), map=('map', 'first'),
                              cheaters=('label', 'sum'), scout_main=('scout_main', 'sum'))
m['idx'] = m.index.str.split('/').str[1].astype(int)
m['lobby'] = m.scout_main >= 3
REC = {}
for src, g in m.groupby('source'):
    g = g.sort_values('idx')
    print(f'\n## {src}: {len(g)} matches, index range {g.idx.min()}-{g.idx.max()}')
    print('scout-lobby match numbers:', g.idx[g.lobby].tolist())
    out = {}
    for col in ['lobby', 'cheaters', 'scout_main', 'map']:
        v = g[col].to_numpy()
        same = lambda a: float(np.mean(a[1:] == a[:-1])) if col in ('lobby', 'map') else float(np.corrcoef(a[1:], a[:-1])[0, 1])
        obs = same(v); perm = np.array([same(rng.permutation(v)) for _ in range(5000)])
        out[col] = {'observed': round(obs, 3), 'shuffled_mean': round(float(perm.mean()), 3), 'p': float((perm >= obs).mean())}
        print(f'  neighbour similarity of {col}: observed {obs:.3f}, shuffled {perm.mean():.3f}, p={out[col]["p"]:.4f}')
    # how often does a lobby match in val/cal/test have a lobby match from TRAIN within 5 numbers?
    tr = g.idx[g.lobby & (g.split == 'train')].to_numpy()
    ev = g[g.lobby & (g.split != 'train')]
    near = [np.abs(tr - i).min() <= 5 if len(tr) else False for i in ev.idx]
    base = np.mean([np.abs(tr - i).min() <= 5 for i in g.idx[~g.lobby & (g.split != 'train')]]) if len(tr) else 0
    print(f'  held-out lobby matches with a TRAINING lobby match within 5 numbers: {int(np.sum(near))}/{len(ev)}; '
          f'same for held-out non-lobby matches: {base:.2f}')
    out['heldout_lobby_near_train_lobby'] = [int(np.sum(near)), len(ev), float(base)]
    REC[src] = out
(OUT / '05_match_index_clustering.json').write_text(json.dumps(REC, indent=1))
