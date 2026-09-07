"""Download the pinned public CS2CD release, checking every Git/LFS digest."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--kind', choices=['raw', 'windows'], default='raw')
    args = ap.parse_args()
    root = Path('data/cs2cd')
    info = json.loads((root / 'metadata' / f'{args.kind}_repo.json').read_text())
    files = json.loads((root / 'metadata' / f'{args.kind}_files.json').read_text())
    files = [f for f in files if f['type'] == 'file' and f['path'].endswith(('.json', '.parquet', '.tar'))]
    # Small representative matches first; then all labels before the larger ticks.
    priority = {'no_cheater_present/0.json', 'no_cheater_present/0.parquet',
                'with_cheater_present/0.json', 'with_cheater_present/0.parquet'}
    files.sort(key=lambda f: (f['path'] not in priority, not f['path'].endswith('.json'), f['size']))
    destroot = root / args.kind
    destroot.mkdir(parents=True, exist_ok=True)
    manifest = root / 'metadata' / f'{args.kind}_verified.jsonl'
    done = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            r = json.loads(line)
            done[r['path']] = r

    def fetch(f):
        dst = destroot / f['path']
        if f['path'] in done and dst.exists() and dst.stat().st_size == f['size']:
            return {**done[f['path']], 'cached': True}
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/{info["id"]}/resolve/{info["sha"]}/{quote(f["path"])}'
        for attempt in range(6):
            try:
                sha = hashlib.sha256()
                git = hashlib.sha1(f'blob {f["size"]}\0'.encode())
                tmp = dst.with_suffix(dst.suffix + '.part')
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with tmp.open('wb') as out:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            out.write(chunk)
                            sha.update(chunk)
                            git.update(chunk)
                assert tmp.stat().st_size == f['size'], f'Unexpected size: {f["path"]}'
                if 'lfs' in f:
                    assert sha.hexdigest() == f['lfs']['oid'], f'LFS mismatch: {f["path"]}'
                else:
                    assert git.hexdigest() == f['oid'], f'Git mismatch: {f["path"]}'
                tmp.replace(dst)
                return {'path': f['path'], 'bytes': f['size'], 'sha256': sha.hexdigest(), 'revision': info['sha']}
            except (requests.RequestException, OSError, AssertionError) as exc:
                if attempt == 5:
                    raise RuntimeError(f'{f["path"]}: {type(exc).__name__}') from exc
                time.sleep(min(2 ** attempt, 30))

    t0 = time.time()
    total = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool, manifest.open('a') as out:
        futures = [pool.submit(fetch, f) for f in files]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            result = fut.result()
            total += result['bytes']
            if not result.get('cached'):
                out.write(json.dumps(result) + '\n')
                out.flush()
            if i <= 5 or i % 20 == 0 or i == len(files):
                print(f'{i}/{len(files)} verified; {total / 1e9:.2f} GB; {time.time()-t0:.0f}s; {result["path"]}', flush=True)


if __name__ == '__main__':
    main()
