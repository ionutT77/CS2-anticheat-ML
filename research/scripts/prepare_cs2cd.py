"""Process completed downloads incrementally, using bounded memory per match."""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anticheat.cs2_data import extract_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--wait-for-download', action='store_true')
    args = ap.parse_args()
    completed = set()
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        while True:
            available = list(Path('data/cs2cd/raw').glob('*/*.parquet'))
            for p in sorted(available):
                if p in completed or p in futures.values():
                    continue
                futures[pool.submit(extract_match, str(p))] = p
            if not futures:
                if not args.wait_for_download or len(completed) == 795:
                    break
                time.sleep(5)
                continue
            ready, _ = cf.wait(futures, timeout=10, return_when=cf.FIRST_COMPLETED)
            for fut in ready:
                p = futures.pop(fut)
                result = fut.result()
                completed.add(p)
                if len(completed) <= 3 or len(completed) % 20 == 0:
                    print(f'{len(completed)}/795 matches; {time.time()-t0:.0f}s; {json.dumps(result)}', flush=True)
    print(f'Completed {len(completed)} matches', flush=True)


if __name__ == '__main__':
    main()
