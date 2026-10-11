"""Phase 1 parity checks for a plugin capture (see PHASE0_PARITY.md, C1-C3).

Run from `research/`:
    python realtime/check_capture.py capture.ndjson [--reference examples/example_validation_player.npz]
Exit status is 1 if the schema is invalid or the C1 leakage test fails.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from realtime.capture_schema import FL_ONGROUND, TICKRATE, read_capture, validate

BURST_GAP = 128  # ticks; same rule as anticheat/cs2_data.py
GUNS_FALLBACK = None


def _guns():
    global GUNS_FALLBACK
    if GUNS_FALLBACK is None:
        from anticheat.cs2_data import GUNS
        GUNS_FALLBACK = GUNS
    return GUNS_FALLBACK


def by_slot(states):
    """slot -> {field: array} sorted by tick. Slots, not Steam IDs: bots share ID 0."""
    rows = defaultdict(list)
    for row in states:
        rows[row['slot']].append(row)
    out = {}
    for slot, items in rows.items():
        items.sort(key=lambda r: r['tick'])
        out[slot] = {k: np.array([r[k] for r in items]) for k in items[0] if k != 'weapon'}
    return out


def tick_gaps(series):
    """Share of ticks missing between each slot's first and last sample.
    Missing ticks cause the training-time extractor to reject a window."""
    total = missing = 0
    for s in series.values():
        span = int(s['tick'][-1] - s['tick'][0]) + 1
        total += span
        missing += span - len(np.unique(s['tick']))
    return {'missing_fraction': missing / total if total else 0.0, 'ticks': total}


def anchors(events):
    """First hit of each attacker->victim burst, gun damage only, no self damage."""
    last, out = {}, []
    hurts = sorted((e for e in events if e['name'] == 'player_hurt'), key=lambda e: e['tick'])
    for e in hurts:
        weapon = str(e['weapon']).removeprefix('weapon_')
        if weapon not in _guns() or e['attacker'] == e['victim'] or e['attacker'] < 0:
            continue
        key = (e['attacker'], e['victim'])
        if e['tick'] - last.get(key, -10**9) > BURST_GAP:
            out.append(e)
        last[key] = e['tick']
    return out


def leakage_check(series, events):
    """C1. For every window anchor T the victim's health must be unchanged over the ticks
    just before T, so no sample inside [T-256, T) reflects the hit. The test does not rely on
    `health + dmg_health` (damage is not capped at remaining health, and other damage
    sources break the equality). Anchors with other damage to the same victim nearby are
    reported as `confounded` and not judged. `first_drop_offset` is the tick, relative to the
    event tick, at which the victim's health first falls; 0 means the sample labeled T is the
    first to show the hit, which is what the window definition needs."""
    lookback = 8
    passed = failed = skipped = team_damage = confounded = not_alive = 0
    offsets = []
    hurts = [e for e in events if e['name'] == 'player_hurt']
    for e in anchors(events):
        s = series.get(e['victim'])
        a = series.get(e['attacker'])
        t = e['tick']
        if s is None or a is None or not np.any(s['tick'] == t - 1) or not np.any(a['tick'] == t - 1):
            skipped += 1
            continue
        if s['team'][s['tick'] == t - 1][0] == a['team'][a['tick'] == t - 1][0]:
            team_damage += 1  # training rejects team damage
            continue
        window = lambda x: (x['tick'] >= t - 256) & (x['tick'] < t)
        if not (s['alive'][window(s)].all() and a['alive'][window(a)].all()):
            not_alive += 1  # training rejects windows not alive throughout
            continue
        if any(h is not e and h['victim'] == e['victim'] and abs(h['tick'] - t) <= lookback
               for h in hurts):
            confounded += 1
            continue
        before = s['health'][(s['tick'] >= t - lookback) & (s['tick'] < t)]
        if len(before) and np.all(before == before[0]):
            passed += 1
        else:
            failed += 1
        near = (s['tick'] >= t - lookback) & (s['tick'] <= t + lookback)
        dropped = near & (s['health'] < before[0]) if len(before) else near & False
        if dropped.any():
            offsets.append(int(s['tick'][dropped].min() - t))
    values, counts = np.unique(offsets, return_counts=True)
    return {'passed': passed, 'failed': failed, 'confounded': confounded,
            'not_alive_throughout': not_alive, 'skipped': skipped, 'team_damage': team_damage,
            'first_drop_offset': {int(v): int(c) for v, c in zip(values, counts)},
            'ok': failed == 0 and passed > 0}


def velocity_check(series):
    """C2. Compare logged AbsVelocity with position differences (dt = 1 tick)."""
    diffs, speed, vertical = [], [], []
    for s in series.values():
        alive = s['alive'] > 0
        step = np.diff(s['tick']) == 1
        ok = step & alive[1:] & alive[:-1]
        derived = np.stack([np.diff(s[k]) for k in ('x', 'y', 'z')], axis=1) * TICKRATE
        logged = np.stack([s[k] for k in ('avx', 'avy', 'avz')], axis=1)[1:]
        diffs.append(np.abs(derived[ok] - logged[ok]))
        speed.append(np.hypot(logged[ok, 0], logged[ok, 1]))
        vertical.append(logged[ok, 2])
    if not diffs or not sum(len(d) for d in diffs):
        return {'samples': 0}
    d, sp, vz = np.concatenate(diffs), np.concatenate(speed), np.concatenate(vertical)
    q = [50, 90, 99]
    return {'samples': int(len(d)),
            'abs_diff_logged_vs_derived_p50_p90_p99': np.percentile(d.max(axis=1), q).round(2).tolist(),
            'logged_speed_p50_p90_p99_max': np.percentile(sp, q + [100]).round(1).tolist(),
            'logged_vertical_abs_p99_max': np.percentile(np.abs(vz), [99, 100]).round(1).tolist()}


def airborne_check(series):
    """C3 (summary only). Per-tick comparison with a GOTV demo needs that demo."""
    flags = np.concatenate([s['flags'][s['alive'] > 0] for s in series.values()])
    return {'airborne_fraction': float(np.mean((flags.astype(np.int64) & FL_ONGROUND) == 0))
            if len(flags) else None}


def reference_summary(path):
    """Raw CS2CD ranges for the channels C2/C3 care about."""
    x = np.load(path)['x'].reshape(-1, 39)
    from anticheat.cs2_data import FEATURES
    pick = {n: FEATURES.index(n) for n in ('attacker_speed', 'attacker_vertical_speed', 'airborne')}
    return {'attacker_speed_p50_p90_p99_max': np.percentile(x[:, pick['attacker_speed']], [50, 90, 99, 100]).round(1).tolist(),
            'attacker_vertical_abs_p99_max': np.percentile(np.abs(x[:, pick['attacker_vertical_speed']]), [99, 100]).round(1).tolist(),
            'airborne_fraction': float(x[:, pick['airborne']].mean())}


def run(path, reference=None):
    header, states, events = read_capture(path)
    problems = validate(header, states, events)
    series = by_slot(states) if states else {}
    report = {'capture': str(path), 'schema_problems': problems,
              'players': len(series), 'state_rows': len(states), 'events': len(events)}
    if series:
        report['tick_gaps'] = tick_gaps(series)
        report['C1_leakage'] = leakage_check(series, events)
        report['C2_velocity'] = velocity_check(series)
        report['C3_airborne'] = airborne_check(series)
    if reference:
        report['reference'] = reference_summary(reference)
    report['ok'] = not problems and bool(report.get('C1_leakage', {}).get('ok'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('capture')
    parser.add_argument('--reference')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = run(args.capture, args.reference)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
