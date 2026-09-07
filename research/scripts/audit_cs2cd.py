"""Freeze match groups/splits before model fitting; finalize extraction audit later."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anticheat.cs2_data import FEATURES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split():
    dest = Path('experiments/cs2cd_v1')
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'protocol.json').exists():
        raise RuntimeError('Protocol already exists; refusing to alter a frozen split.')
    paths = sorted(Path('data/cs2cd/raw').glob('*/*.json'))
    assert len(paths) == 795, len(paths)
    rows, fingerprints = [], defaultdict(list)
    # Player IDs are renamed on every match. Fingerprints deliberately omit them.
    blocks = defaultdict(list)
    for p in paths:
        d = json.loads(p.read_text())
        mid = p.parent.name + '/' + p.stem
        hits = sorted(d.get('player_hurt', []), key=lambda e: e['tick'])
        canonical = [(e['tick'], e.get('weapon'), e.get('dmg_health'), e.get('hitgroup')) for e in hits]
        if canonical:
            first = canonical[0][0]
            sig = [(t-first, *rest) for t, *rest in canonical]
            fingerprints[hashlib.sha256(json.dumps(sig).encode()).hexdigest()].append(mid)
            for start in range(0, len(canonical)-31, 16):
                chunk = canonical[start:start+32]
                block = [(t-chunk[0][0], *rest) for t, *rest in chunk]
                blocks[hashlib.sha256(json.dumps(block).encode()).hexdigest()].append(mid)
        info = d.get('CSstats_info', [{}])[0]
        rows.append({'match_id': mid, 'source': p.parent.name, 'map': info.get('map'),
                     'cheater_labels': len(d.get('cheaters', [])), 'hurt_events': len(hits)})
    parent = {r['match_id']: r['match_id'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for bucket in [*fingerprints.values(), *blocks.values()]:
        for mid in bucket[1:]:
            union(bucket[0], mid)
    frame = pd.DataFrame(rows)
    frame['group'] = [find(m) for m in frame.match_id]
    groups = frame.groupby('group').agg(positive=('cheater_labels', 'max')).reset_index()
    strat = (groups.positive > 0).astype(int)
    tr, other = train_test_split(np.arange(len(groups)), test_size=.40, random_state=9072026, stratify=strat)
    va, ct = train_test_split(other, test_size=.625, random_state=9072027, stratify=strat.iloc[other])
    ca, te = train_test_split(ct, test_size=.60, random_state=9072028, stratify=strat.iloc[ct])
    assignment = {groups.group.iloc[i]: name for name, idx in [('train', tr), ('validation', va),
                    ('calibration', ca), ('test', te)] for i in idx}
    frame['split'] = frame.group.map(assignment)
    frame.to_csv(dest / 'match_splits.csv', index=False)
    protocol = {
        'version': 1, 'dataset': 'CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection',
        'revision': json.loads(Path('data/cs2cd/metadata/raw_repo.json').read_text())['sha'],
        'decision_unit': 'player within one match, with at least one eligible encounter',
        'window': '256 consecutive ticks ending strictly before first bullet damage in a two-second attacker/victim burst',
        'labels': 'publisher player-match labels; not verified per-event cheat activation',
        'features': FEATURES, 'feature_count': len(FEATURES),
        'excluded_inputs': ['source folder', 'map', 'rank', 'server', 'player ID', 'absolute tick',
                            'kill outcome', 'future ticks', 'usercmd mouse fields', 'spotted flags'],
        'identity_limit': 'Source anonymization reassigns Player_N separately per DEM; true player-disjointness cannot be verified.',
        'split': '60/15/10/15 percent by match-connected group, stratified on presence of a positive label',
        'duplicate_grouping': 'Exact full hurt-event signatures and 32-event subsequences, relative ticks, IDs omitted; plus extracted exact-window checks before training',
        'primary_selection': 'Mean of validation player-match ROC-AUC on all eligible players and on manually-reviewed matches only; ties within 0.001 resolved by reviewed-cohort AUC then fewer parameters',
        'models_planned': ['five-channel LSTM', '39-channel LSTM', '39-channel temporal CNN',
                           '39-channel patch Transformer', 'hierarchical CNN with chronological encounter Transformer',
                           'LightGBM player summaries leaves15/31', 'five-channel LightGBM ablation',
                           '39-channel TCN with supervised training only on manually-reviewed matches'],
        'neural_training': {'epochs': 40, 'patience': 8, 'seeds': [42, 123],
                            'optimizer': 'AdamW', 'lr': .0005, 'weight_decay': .001,
                            'loss': 'unweighted player-match BCE; sampled encounter bags; optional event auxiliary BCE 0.25',
                            'training_bag_size': 16, 'evaluation_bag': 'all eligible encounters'},
        'ensemble_selection': 'Top neural and top tree blend weights 0,0.25,0.5,0.75,1 using the same validation mean AUC criterion only',
        'calibration': 'positive-slope Platt on calibration; thresholds for accuracy, 1% and 0.1% observed calibration FPR, precision99% exploratory',
        'test': 'single frozen evaluation after selection/calibration; cluster bootstrap by match group; all-cohort and manually-reviewed-cohort results',
        'horizons': 'complete match primary; first 1,5,10 encounters supplementary without retuning thresholds',
        'coverage': 'Report players excluded for zero eligible encounters; no performance claim for miss-only/no-damage players',
        'claims': 'Research benchmark only. No new-player, future-time, unseen-cheat-family or 99% deployment claim.',
        'match_split_sha256': digest(dest / 'match_splits.csv'),
        'extraction_source_sha256': digest('anticheat/cs2_data.py'),
        'exact_signature_duplicate_groups': [v for v in fingerprints.values() if len(v) > 1],
        'connected_groups': int(frame.group.nunique()),
        'counts': frame.groupby('split').agg(matches=('match_id', 'size'), positive_player_labels=('cheater_labels', 'sum')).to_dict('index'),
    }
    (dest / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps({k: protocol[k] for k in ['counts', 'connected_groups', 'exact_signature_duplicate_groups']}, indent=2))


def finalize():
    dest = Path('experiments/cs2cd_v1')
    protocol = json.loads((dest / 'protocol.json').read_text())
    assert digest('anticheat/cs2_data.py') == protocol['extraction_source_sha256']
    matches = pd.read_csv(dest / 'match_splits.csv').set_index('match_id')
    files = sorted(Path('data/cs2cd/processed/matches').glob('*.json'))
    assert len(files) == 795, len(files)
    all_events, players, rejected, nulls, extra = [], [], Counter(), Counter(), Counter()
    raw_rows = 0
    for p in files:
        d = json.loads(p.read_text())
        mid = d['match_id']
        fields = matches.loc[mid].to_dict()
        for i, e in enumerate(d['events']):
            all_events.append({**e, **fields, 'match_id': mid, 'npz': str(p.with_suffix('.npz')),
                               'local_index': i})
        for player in d['players']:
            players.append({**player, **fields, 'match_id': mid, 'player_match': mid + '/' + player['player']})
        rejected.update(d['rejected']); nulls.update(d['nulls']); extra.update(d['extra_nulls']); raw_rows += d['raw_rows']
    events = pd.DataFrame(all_events)
    dup = events.groupby('sha256').agg(splits=('split', 'nunique'), matches=('match_id', 'nunique'), n=('label', 'size'))
    cross = dup[dup.splits > 1]
    assert len(cross) == 0, f'Cross-split exact duplicates found: {len(cross)}; stop and audit before training'
    events['player_match'] = events.match_id + '/' + events.player
    events.to_parquet(dest / 'events.parquet', index=False)
    players = pd.DataFrame(players)
    players.to_csv(dest / 'players.csv', index=False)
    report = {'raw_rows': raw_rows, 'matches': len(files), 'features': FEATURES,
              'encounters': len(events), 'players': len(players),
              'eligible_players': int((players.encounters > 0).sum()),
              'rejected_encounters': dict(rejected), 'null_fraction': {k: v/raw_rows for k,v in nulls.items()},
              'unused_field_null_fraction': {k: v/raw_rows for k,v in extra.items()},
              'exact_duplicate_windows': int((dup.n-1).sum()), 'cross_split_exact_duplicates': len(cross),
              'split_counts': events.groupby('split').agg(encounters=('label', 'size'), positives=('label', 'sum'),
                                players=('player_match', 'nunique'), matches=('match_id', 'nunique')).to_dict('index'),
              'coverage_by_label': players.groupby('label').encounters.agg(['size', lambda s: int((s>0).sum())]).to_dict('index'),
              'events_sha256': digest(dest / 'events.parquet'), 'players_sha256': digest(dest / 'players.csv')}
    (dest / 'data_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['encounters', 'eligible_players', 'rejected_encounters', 'split_counts', 'coverage_by_label']}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--finalize', action='store_true'); args = ap.parse_args()
    finalize() if args.finalize else split()
