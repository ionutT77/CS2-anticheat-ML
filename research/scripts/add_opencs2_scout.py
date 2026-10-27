"""Add legitimate scout (SSG 08) encounters from OpenCS2 pro demos (see research/opencs2/OTHER_WEAPONS.md).

Data addition only: no training, no change to the frozen cs2cd_v1 experiment or to awp_v1.
Demos are taken in order of scout kills (from the metadata), skipping the demos already handled by
add_opencs2_awp.py. Every player in a demo is extracted; a player-map with enough scout encounters
goes to the training addition (`events.parquet`), everyone else to `reserve_events.parquet`.
Resumable: re-running continues from the manifest. Run from `research/`:
    python scripts/add_opencs2_scout.py [--max-demos 100] [--raw D:/opencs2]
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anticheat.dem_extractor import extract_dem
from scripts.add_opencs2_awp import REVISION, SOURCE, LICENSE, REPO, download, metadata, settle, sha256

SCOUT = {'ssg08', 'SSG 08'}


def scout_kills(row):
    return sum(k['weapon'] in SCOUT for k in row['kills'] or [])


def ranked_demos(rows, exclude):
    """Demos with scout kills, most first; ties keep the metadata order (deterministic)."""
    pool = [(scout_kills(r), i, r) for i, r in enumerate(rows) if r['file_name'] not in exclude]
    return [r for n, _, r in sorted(pool, key=lambda t: (-t[0], t[1])) if n]


def accepted_player(events, min_scout):
    """A player-map is accepted when it has at least `min_scout` scout encounters."""
    return sum(ev['weapon'] in SCOUT for ev in events) >= min_scout


def fetch_locked(path, revision, dest, expected_bytes, attempts=6, wait=30):
    """download(), but a Windows sharing violation on the .part file (WinError 32, typically a
    scanner holding the fresh file) is waited out instead of failing the demo. A transfer that
    finished and only waits for the rename is renamed, not downloaded again."""
    part = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(1, attempts + 1):
        try:
            return download(path, revision, dest, expected_bytes=expected_bytes)
        except PermissionError as error:
            if attempt == attempts:
                raise
            print(f'file locked, waiting {wait} s ({attempt}/{attempts - 1}): {Path(path).name}: {error.winerror}',
                  flush=True)
            time.sleep(wait)
            try:
                if part.exists() and part.stat().st_size == expected_bytes:
                    part.replace(dest)
                    return dest
            except PermissionError:
                pass


def yield_exhausted(demos, window, threshold):
    """True when the last `window` usable demos averaged at most `threshold` scout encounters.
    Demos are taken best-first, so a low recent mean means the rest will give no more."""
    recent = [d['scout_encounters'] for d in demos if 'error' not in d][-window:]
    return len(recent) == window and sum(recent) / window <= threshold


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--max-demos', type=int, default=100, help='stop after this many demos in the manifest')
    p.add_argument('--min-free-gb', type=float, default=40, help='stop when the raw drive has less free space')
    p.add_argument('--stop-mean', type=float, default=2.0,
                   help='stop when the last --stop-window usable demos average at most this many scout encounters')
    p.add_argument('--stop-window', type=int, default=5)
    p.add_argument('--min-scout', type=int, default=3, help='scout encounters for a player-map to be accepted')
    p.add_argument('--raw', default='D:/opencs2')
    p.add_argument('--out', default='data/opencs2/scout_v1')
    p.add_argument('--awp-manifest', default='data/opencs2/awp_v1/manifest.json')
    p.add_argument('--revision', default=REVISION)
    p.add_argument('--prefetch', type=int, default=3, help='demos downloaded ahead in parallel')
    args = p.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    (out / 'matches').mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'source': SOURCE, 'repo': REPO, 'revision': args.revision, 'license': LICENSE,
        'selection': {'order': 'metadata scout kills, most first', 'min_scout': args.min_scout,
                      'excluded': 'demos already in awp_v1 manifest'},
        'demos': []}
    save = lambda: manifest_path.write_text(json.dumps(manifest, indent=1), encoding='utf-8')

    handled = {d['file_name'] for d in json.loads(Path(args.awp_manifest).read_text())['demos']}
    cs2cd_sha = set(pd.read_parquet('experiments/cs2cd_v1/events.parquet', columns=['sha256'])['sha256'])
    todo = ranked_demos(metadata(raw, args.revision), handled)
    done = {d['file_name'] for d in manifest['demos']}
    todo = [r for r in todo if r['file_name'] not in done]
    print(f'demos with scout kills not yet handled: {len(todo)}; already in manifest: {len(done)}', flush=True)
    dest = lambda r: raw / 'demos' / r['file_name'].split('demos/', 1)[1]
    fetch = lambda r: fetch_locked(r['file_name'], args.revision, dest(r), r['demo_bytes'])
    executor, futures = ThreadPoolExecutor(max_workers=args.prefetch), {}

    def prefetch(start):
        for r in todo[start:start + args.prefetch]:
            if r['file_name'] not in futures:   # setdefault would submit a duplicate download every call
                futures[r['file_name']] = executor.submit(fetch, r)

    for position, row in enumerate(todo):
        if len(manifest['demos']) >= args.max_demos:
            break
        if yield_exhausted(manifest['demos'], args.stop_window, args.stop_mean):
            print(f'stopping: the last {args.stop_window} usable demos averaged at most {args.stop_mean} '
                  f'scout encounters', flush=True)
            break
        free_gb = shutil.disk_usage(raw).free / 1e9
        if free_gb < args.min_free_gb:
            print(f'stopping: {free_gb:.0f} GB free on {raw}, below --min-free-gb {args.min_free_gb}', flush=True)
            break
        started = time.perf_counter()
        prefetch(position)
        dem, download_error = settle(futures.pop(row['file_name']))
        prefetch(position + 1)
        if download_error:
            manifest.setdefault('download_failures', []).append({
                'file_name': row['file_name'], 'match_id': row['match_id'], 'map_index': row['map_index'],
                'error': download_error})
            save()
            print(f"skip download {row['match_id']}_m{row['map_index']}: {download_error}", flush=True)
            continue
        stem = f"{row['match_id']}_m{row['map_index']}"
        npz = out / 'matches' / f'{stem}.npz'
        record = {'file_name': row['file_name'], 'match_id': row['match_id'], 'map_index': row['map_index'],
                  'map': row['map_name'], 'patch_version': row['patch_version'], 'bytes': row['demo_bytes'],
                  'sha256': sha256(dem), 'scout_kills': scout_kills(row), 'accepted': 0}
        try:
            audit = extract_dem(dem, output_path=npz, consent_players=None)
        except Exception as error:  # timing/format rejections are recorded, not hidden
            record['error'] = f'{type(error).__name__}: {error}'[:300]
            manifest['demos'].append(record)
            save()
            print(f"skip {stem}: {record['error']}", flush=True)
            continue
        alias = {pl['local_id']: str(pl['steamid']) for pl in audit['players']}
        by_player = defaultdict(list)
        for ev in audit['events']:
            by_player[ev['player']].append(ev)
        accepted = []
        for local, evs in by_player.items():
            if accepted_player(evs, args.min_scout):
                accepted.append({'steamid': alias.get(local), 'local_player': local, 'encounters': len(evs),
                                 'scout_encounters': sum(ev['weapon'] in SCOUT for ev in evs)})
        record.update(encounters=audit['encounter_count'], rejected=audit['rejected'],
                      approximated=audit['approximated_features'], warnings=audit['warnings'][:5],
                      scout_encounters=sum(ev['weapon'] in SCOUT for ev in audit['events']),
                      accepted=len(accepted), accepted_players=accepted,
                      seconds=round(time.perf_counter() - started, 1))
        manifest['demos'].append(record)
        save()
        print(f"{stem}: scout kills {record['scout_kills']}, scout encounters {record['scout_encounters']}, "
              f"accepted {record['accepted']}, demos {len(manifest['demos'])}/{args.max_demos}, "
              f"{record['seconds']} s", flush=True)
    executor.shutdown(wait=False, cancel_futures=True)

    def table_row(demo, ev_index, ev, sha, split, encounters):
        match = f"opencs2/{demo['match_id']}_m{demo['map_index']}"
        return {'match_id': match, 'player': ev['player'], 'victim': ev['victim'], 'tick': int(ev['tick']),
                'label': 0, 'weapon': ev['weapon'], 'sha256': sha, 'source': SOURCE, 'map': demo['map'],
                'cheater_labels': 0, 'hurt_events': encounters, 'group': f"opencs2/{demo['match_id']}",
                'split': split,
                'npz': str(out / 'matches' / f"{demo['match_id']}_m{demo['map_index']}.npz").replace('\\', '/'),
                'local_index': ev_index, 'player_match': f"{match}/{ev['player']}"}

    train, reserve = [], []
    for demo in (d for d in manifest['demos'] if 'error' not in d):
        npz = out / 'matches' / f"{demo['match_id']}_m{demo['map_index']}.npz"
        audit = json.loads(npz.with_suffix('.json').read_text(encoding='utf-8'))
        x = np.load(npz)['x']
        keep = {a['local_player'] for a in demo['accepted_players']}
        for i, ev in enumerate(audit['events']):
            sha = hashlib.sha256(x[i].tobytes()).hexdigest()
            is_train = ev['player'] in keep
            (train if is_train else reserve).append(
                table_row(demo, i, ev, sha, 'train' if is_train else 'reserve', len(audit['events'])))
    table, reserve_table = pd.DataFrame(train), pd.DataFrame(reserve)
    if len(table):
        table.to_parquet(out / 'events.parquet', index=False)
    if len(reserve_table):
        reserve_table.to_parquet(out / 'reserve_events.parquet', index=False)
    scout = lambda t: int(t['weapon'].isin(SCOUT).sum()) if len(t) else 0
    manifest['totals'] = {
        'demos_processed': len(manifest['demos']),
        'demos_failed': sum('error' in d for d in manifest['demos']),
        'downloaded_gb': round(sum(d['bytes'] for d in manifest['demos']) / 1e9, 1),
        'accepted_player_maps': int(table['player_match'].nunique()) if len(table) else 0,
        'encounters': len(table), 'scout_encounters': scout(table),
        'reserve_encounters': len(reserve_table), 'reserve_scout_encounters': scout(reserve_table),
        'duplicate_windows_vs_cs2cd': int(table['sha256'].isin(cs2cd_sha).sum()) if len(table) else 0,
        'reserve_duplicate_windows_vs_cs2cd': int(reserve_table['sha256'].isin(cs2cd_sha).sum()) if len(reserve_table) else 0}
    save()
    print(json.dumps(manifest['totals'], indent=1))


if __name__ == '__main__':
    main()
