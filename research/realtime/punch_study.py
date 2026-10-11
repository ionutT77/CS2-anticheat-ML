"""C4 study: is the live aim-punch source the same quantity as the training feature?

Compares the raw punch_pitch/punch_yaw channels of CS2CD reference windows with the
`punch0/punch1` (m_predictableBaseAngle) values in a plugin capture. Behavioural statistics
only; no model is run. Run from `research/`:
    python realtime/punch_study.py capture.ndjson [--reference examples/example_validation_player.npz]
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anticheat.cs2_data import FEATURES

ZERO = 1e-3      # degrees; below this the punch counts as "at rest"
SETTLED = 0.05   # degrees; return-to-rest threshold after the last shot


def series_stats(pitch, yaw, shot):
    """Statistics for one contiguous series of per-tick values and a 0/1 shot flag."""
    pitch, yaw, shot = map(np.asarray, (pitch, yaw, shot))
    change = np.abs(np.diff(pitch)) > ZERO
    quiet = shot[1:] == 0
    settle = []
    shots = np.flatnonzero(shot > 0)
    for i in shots:
        nxt = shots[shots > i]
        end = nxt[0] if len(nxt) else len(pitch)
        if end - i < 64:      # need >= 1 s without another shot to measure settling
            continue
        rest = np.flatnonzero(np.abs(pitch[i + 1:end]) < SETTLED)
        settle.append(int(rest[0]) + 1 if len(rest) else None)
    measured = [s for s in settle if s is not None]
    return dict(n=len(pitch), shot_ticks=int(shot.sum()),
                zero_fraction=float(np.mean(np.abs(pitch) < ZERO)),
                pitch_min=float(pitch.min()), pitch_max=float(pitch.max()),
                yaw_min=float(yaw.min()), yaw_max=float(yaw.max()),
                changes_while_not_shooting=float(change[quiet].mean()) if quiet.any() else None,
                settle_ticks_median=float(np.median(measured)) if measured else None,
                never_settled=sum(s is None for s in settle), settle_samples=len(settle))


def reference(path):
    x = np.load(path)['x']
    i = {n: FEATURES.index(n) for n in ('punch_pitch', 'punch_yaw', 'shot')}
    per = [series_stats(w[:, i['punch_pitch']], w[:, i['punch_yaw']], w[:, i['shot']]) for w in x]
    return combine(per)


def capture(path):
    rows, fires = defaultdict(list), defaultdict(set)
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('type') == 'state' and r.get('alive') and r.get('punch0') is not None:
                rows[r['slot']].append(r)
            elif r.get('type') == 'event' and r.get('name') == 'weapon_fire':
                fires[r['user']].add(r['tick'])
    per, base_tick_changes = [], []
    for slot, items in rows.items():
        items.sort(key=lambda r: r['tick'])
        ticks = np.array([r['tick'] for r in items])
        # split into contiguous alive runs so deaths/round resets do not join series
        for run in np.split(np.arange(len(items)), np.flatnonzero(np.diff(ticks) != 1) + 1):
            if len(run) < 128:
                continue
            seg = [items[k] for k in run]
            shot = [1 if r['tick'] in fires[slot] else 0 for r in seg]
            per.append(series_stats([r['punch0'] for r in seg], [r['punch1'] for r in seg], shot))
            pb = np.array([r['pb_tick'] for r in seg])
            base_tick_changes.append(float(np.mean(np.diff(pb) != 0)))
    out = combine(per)
    out['base_tick_changes_per_tick'] = float(np.mean(base_tick_changes)) if base_tick_changes else None
    return out


def combine(per):
    shots = [p for p in per if p['shot_ticks']]
    settle = [p['settle_ticks_median'] for p in per if p['settle_ticks_median'] is not None]
    nc = [p['changes_while_not_shooting'] for p in per if p['changes_while_not_shooting'] is not None]
    return dict(series=len(per), with_shots=len(shots),
                ticks=int(sum(p['n'] for p in per)),
                zero_fraction=float(np.average([p['zero_fraction'] for p in per], weights=[p['n'] for p in per])),
                pitch_range=[min(p['pitch_min'] for p in per), max(p['pitch_max'] for p in per)],
                yaw_range=[min(p['yaw_min'] for p in per), max(p['yaw_max'] for p in per)],
                changes_while_not_shooting=float(np.mean(nc)) if nc else None,
                settle_ticks_median=float(np.median(settle)) if settle else None,
                never_settled=int(sum(p['never_settled'] for p in per)),
                settle_samples=int(sum(p['settle_samples'] for p in per)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('capture')
    parser.add_argument('--reference', default='examples/example_validation_player.npz')
    args = parser.parse_args()
    print(json.dumps({'reference_cs2cd': reference(args.reference),
                      'live_predictableBaseAngle': capture(args.capture)}, indent=2))


if __name__ == '__main__':
    main()
