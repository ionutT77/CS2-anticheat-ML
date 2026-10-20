"""Add legitimate sniper-heavy pro player-maps from OpenCS2 (see research/opencs2/PLAN.md).

Data addition only: no training, no change to the frozen cs2cd_v1 experiment.
Resumable: re-running continues from the manifest. Run from `research/`:
    python scripts/add_opencs2_awp.py [--target 211] [--raw D:/opencs2]
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anticheat.cs2_data import SNIPER
from anticheat.dem_extractor import extract_dem

REPO = 'blanchon/opencs2_dataset_demo'
REVISION = '912a363098feec3f220e41d68634d2f71676c28b'
LICENSE = 'CC BY 4.0'
SOURCE = 'opencs2_pro'


def url(path, revision):
    return f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}'


def download(path, revision, dest, expected_bytes=None, attempts=3):
    """Download with retries: a cut-off transfer (size mismatch or network error) is retried,
    so one bad connection does not stop a multi-hour run."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        return dest
    part = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url(path, revision), timeout=120) as response, open(part, 'wb') as out:
                while chunk := response.read(1 << 22):
                    out.write(chunk)
            if expected_bytes is not None and part.stat().st_size != expected_bytes:
                raise IOError(f'size mismatch for {path}')
            part.replace(dest)
            return dest
        except (IOError, OSError) as error:
            part.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            print(f'retry {attempt}/{attempts - 1} for {path}: {error}', flush=True)
            time.sleep(10 * attempt)


def settle(future):
    """Return (result, error). A failed download is reported, not raised, so one bad
    connection costs one demo instead of the whole run."""
    try:
        return future.result(), None
    except Exception as error:  # network, size mismatch after retries, disk errors
        return None, f'{type(error).__name__}: {error}'[:300]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def metadata(raw, revision):
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}/revision/{revision}', timeout=60) as r:
        files = [s['rfilename'] for s in json.load(r)['siblings']
                 if s['rfilename'].startswith('data/metadata-') and s['rfilename'].endswith('.parquet')]
    columns = ['file_name', 'match_id', 'map_index', 'map_name', 'patch_version', 'demo_bytes', 'kills']
    rows = []
    for f in sorted(files):
        local = download(f, revision, raw / 'metadata' / Path(f).name)
        rows += pq.read_table(local, columns=columns).to_pylist()
    return rows


def candidates(row, min_kills=8, min_share=0.5):
    kills, sniper = Counter(), Counter()
    for k in row['kills'] or []:
        sid = str(k['attacker_steamid'])
        kills[sid] += 1
        sniper[sid] += k['weapon'] in SNIPER
    return sorted(s for s in kills if kills[s] >= min_kills and sniper[s] / kills[s] >= min_share)


def ordered_demos(rows, seed):
    pool = [(r, candidates(r)) for r in rows]
    pool = [(r, c) for r, c in pool if c]
    order = np.random.default_rng(seed).permutation(len(pool))
    pool = [pool[i] for i in order]
    return sorted(pool, key=lambda item: -min(len(item[1]), 2))  # stable: two-candidate demos first


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--target', type=int, default=211)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--max-maps-per-player', type=int, default=3)
    p.add_argument('--min-encounters', type=int, default=5)
    p.add_argument('--raw', default='D:/opencs2')
    p.add_argument('--out', default='data/opencs2/awp_v1')
    p.add_argument('--revision', default=REVISION)
    p.add_argument('--max-demos', type=int, default=400, help='safety stop on downloads')
    p.add_argument('--prefetch', type=int, default=3, help='demos downloaded ahead in parallel')
    args = p.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    (out / 'matches').mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'source': SOURCE, 'repo': REPO, 'revision': args.revision, 'license': LICENSE,
        'selection': {k: getattr(args, k) for k in ('target', 'seed', 'max_maps_per_player', 'min_encounters')}
                     | {'candidate_rule': '>=8 kills and >=50% sniper kills (metadata)',
                        'accept_rule': f'>={args.min_encounters} eligible encounters and >50% with a sniper'},
        'demos': [], 'accepted': []}

    if manifest['selection']['target'] != args.target:   # e.g. staged 211 -> 100, recorded for provenance
        manifest.setdefault('target_history', []).append(
            {'from': manifest['selection']['target'], 'to': args.target, 'at_accepted': len(manifest['accepted'])})
        manifest['selection']['target'] = args.target

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding='utf-8')

    cs2cd_sha = set(pd.read_parquet('experiments/cs2cd_v1/events.parquet', columns=['sha256'])['sha256'])
    rows = metadata(raw, args.revision)
    pool = ordered_demos(rows, args.seed)
    print(f'metadata rows {len(rows)}, demos with AWP candidates {len(pool)}', flush=True)
    done = {d['file_name'] for d in manifest['demos']}
    maps_per_player = Counter(a['steamid'] for a in manifest['accepted'])
    todo = [(r, c) for r, c in pool if r['file_name'] not in done]
    dest = lambda r: raw / 'demos' / r['file_name'].split('demos/', 1)[1]
    fetch = lambda r: download(r['file_name'], args.revision, dest(r), expected_bytes=r['demo_bytes'])
    executor, futures = ThreadPoolExecutor(max_workers=args.prefetch), {}

    def prefetch(start):
        # download the next demos while the current one is extracted; a prefetched demo can
        # still be skipped later if its candidates reach the per-player cap meanwhile
        for r, c in todo[start:start + args.prefetch]:
            if r['file_name'] not in futures and any(maps_per_player[x] < args.max_maps_per_player for x in c):
                futures[r['file_name']] = executor.submit(fetch, r)

    for position, (row, cands) in enumerate(todo):
        if len(manifest['accepted']) >= args.target or len(manifest['demos']) >= args.max_demos:
            break
        wanted = [c for c in cands if maps_per_player[c] < args.max_maps_per_player]
        if not wanted:
            continue
        started = time.perf_counter()
        prefetch(position)
        dem, download_error = settle(futures.pop(row['file_name']))
        prefetch(position + 1)
        if download_error:
            # not recorded in 'demos', so a later run retries this file
            manifest.setdefault('download_failures', []).append({
                'file_name': row['file_name'], 'match_id': row['match_id'], 'map_index': row['map_index'],
                'error': download_error, 'at_accepted': len(manifest['accepted'])})
            save()
            print(f"skip download {row['match_id']}_m{row['map_index']}: {download_error}", flush=True)
            continue
        stem = f"{row['match_id']}_m{row['map_index']}"
        npz = out / 'matches' / f'{stem}.npz'
        record = {'file_name': row['file_name'], 'match_id': row['match_id'], 'map_index': row['map_index'],
                  'map': row['map_name'], 'patch_version': row['patch_version'], 'bytes': row['demo_bytes'],
                  'sha256': sha256(dem), 'candidates': len(wanted), 'accepted': 0}
        try:
            # all players are extracted (parsing dominates the cost); only accepted candidates
            # enter the training addition, everyone else goes to the reserve table
            audit = extract_dem(dem, output_path=npz, consent_players=None)
        except Exception as error:  # timing/format rejections are recorded, not hidden
            record['error'] = f'{type(error).__name__}: {error}'[:300]
            manifest['demos'].append(record)
            save()
            print(f"skip {stem}: {record['error']}", flush=True)
            continue
        x = np.load(npz)['x']
        alias = {pl['local_id']: str(pl['steamid']) for pl in audit['players']}
        by_player = defaultdict(list)
        for i, ev in enumerate(audit['events']):
            by_player[alias.get(ev['player'])].append((i, ev))
        record.update(encounters=audit['encounter_count'], rejected=audit['rejected'],
                      approximated=audit['approximated_features'], warnings=audit['warnings'][:5])
        for sid in wanted:
            evs = by_player.get(sid, [])
            snip = sum(ev['weapon'] in SNIPER for _, ev in evs)
            if len(evs) < args.min_encounters or snip * 2 <= len(evs):
                continue
            if len(manifest['accepted']) >= args.target:
                break
            local = evs[0][1]['player']
            manifest['accepted'].append({
                'steamid': sid, 'match_id': row['match_id'], 'map_index': row['map_index'], 'npz': str(npz),
                'local_player': local, 'encounters': len(evs), 'sniper_encounters': snip,
                'rows': [{'local_index': i, 'tick': int(ev['tick']), 'victim': ev['victim'], 'weapon': ev['weapon'],
                          'sha256': hashlib.sha256(x[i].tobytes()).hexdigest()} for i, ev in evs]})
            maps_per_player[sid] += 1
            record['accepted'] += 1
        record['seconds'] = round(time.perf_counter() - started, 1)
        manifest['demos'].append(record)
        save()
        print(f"{stem}: candidates {len(wanted)}, accepted {record['accepted']}, total "
              f"{len(manifest['accepted'])}/{args.target}, {record['seconds']} s", flush=True)
    executor.shutdown(wait=False, cancel_futures=True)

    def row(demo, local_player, ev_index, ev, sha, split, encounters):
        match = f"opencs2/{demo['match_id']}_m{demo['map_index']}"
        return {'match_id': match, 'player': local_player, 'victim': ev['victim'], 'tick': int(ev['tick']),
                'label': 0, 'weapon': ev['weapon'], 'sha256': sha, 'source': SOURCE, 'map': demo['map'],
                'cheater_labels': 0, 'hurt_events': encounters, 'group': f"opencs2/{demo['match_id']}",
                'split': split, 'npz': str(out / 'matches' / f"{demo['match_id']}_m{demo['map_index']}.npz").replace('\\', '/'),
                'local_index': ev_index, 'player_match': f"{match}/{local_player}"}

    demos = {(d['match_id'], d['map_index']): d for d in manifest['demos'] if 'error' not in d}
    train, reserve = [], []
    accepted_keys = {(a['match_id'], a['map_index'], a['local_player']) for a in manifest['accepted']}
    for (match_id, map_index), demo in demos.items():
        npz = out / 'matches' / f'{match_id}_m{map_index}.npz'
        audit = json.loads(npz.with_suffix('.json').read_text(encoding='utf-8'))
        x = np.load(npz)['x']
        for i, ev in enumerate(audit['events']):
            sha = hashlib.sha256(x[i].tobytes()).hexdigest()
            is_train = (match_id, map_index, ev['player']) in accepted_keys
            (train if is_train else reserve).append(   # hurt_events: per match, as in CS2CD
                row(demo, ev['player'], i, ev, sha, 'train' if is_train else 'reserve', len(audit['events'])))
    table, reserve_table = pd.DataFrame(train), pd.DataFrame(reserve)
    if len(table):
        table.to_parquet(out / 'events.parquet', index=False)
    if len(reserve_table):
        # NOT independent of the training addition (same matches): extra training data later,
        # never an independent evaluation cohort for a model trained on events.parquet
        reserve_table.to_parquet(out / 'reserve_events.parquet', index=False)
    manifest['totals'] = {
        'accepted_player_maps': len(manifest['accepted']),
        'distinct_players': len({a['steamid'] for a in manifest['accepted']}),
        'demos_processed': len(manifest['demos']),
        'demos_failed': sum('error' in d for d in manifest['demos']),
        'downloaded_gb': round(sum(d['bytes'] for d in manifest['demos']) / 1e9, 1),
        'encounters': len(table), 'sniper_encounters': int(table['weapon'].isin(SNIPER).sum()) if len(table) else 0,
        'awp_encounters': int((table['weapon'] == 'awp').sum()) if len(table) else 0,
        'duplicate_windows_vs_cs2cd': int(table['sha256'].isin(cs2cd_sha).sum()) if len(table) else 0,
        'reserve_encounters': len(reserve_table),
        'reserve_player_maps': int(reserve_table['player_match'].nunique()) if len(reserve_table) else 0,
        'reserve_sniper_encounters': int(reserve_table['weapon'].isin(SNIPER).sum()) if len(reserve_table) else 0,
        'reserve_duplicate_windows_vs_cs2cd': int(reserve_table['sha256'].isin(cs2cd_sha).sum()) if len(reserve_table) else 0}
    save()
    print(json.dumps(manifest['totals'], indent=1))


if __name__ == '__main__':
    main()
