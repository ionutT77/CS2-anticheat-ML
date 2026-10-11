"""Tick-level parity: plugin capture vs. the GOTV demo of the same session.

For one human player (SteamID64; the demo parser ignores bots) this measures:
  * the tick offset between plugin and demo event stamps (weapon_fire, player_hurt);
  * per-field agreement of state samples at the best constant state shift.
Run from `research/`:
    python realtime/demo_parity.py capture.ndjson session.dem --steamid 7656119...
Prints aggregate statistics only (no raw telemetry).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from realtime.capture_schema import read_capture

PUNCH_PROP = 'CCSPlayerPawn.CCSPlayer_AimPunchServices.m_predictableBaseAngle'


def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def demo_series(dem, steamid):
    from demoparser2 import DemoParser
    parser = DemoParser(str(dem))
    df = parser.parse_ticks(['pitch', 'yaw', 'X', 'Y', 'Z', 'health', PUNCH_PROP])
    df = df[df['steamid'].astype(str) == str(steamid)].sort_values('tick')
    punch = [v if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2 else (np.nan, np.nan)
             for v in df[PUNCH_PROP]] if PUNCH_PROP in df else [(np.nan, np.nan)] * len(df)
    series = {'tick': df['tick'].to_numpy(int), 'yaw': df['yaw'].to_numpy(float),
              'pitch': df['pitch'].to_numpy(float), 'x': df['X'].to_numpy(float),
              'y': df['Y'].to_numpy(float), 'health': df['health'].to_numpy(float),
              'punch0': np.array([p[0] for p in punch], float),
              'punch1': np.array([p[1] for p in punch], float)}
    events = {}
    for name, who in (('weapon_fire', 'user_steamid'), ('player_hurt', 'attacker_steamid')):
        frame = parser.parse_event(name)
        if isinstance(frame, list) or frame is None or not len(frame):
            events[name] = np.array([], int)
            continue
        events[name] = np.sort(frame[frame[who].astype(str) == str(steamid)]['tick'].to_numpy(int))
    return series, events


def capture_series(path, steamid):
    _, states, events = read_capture(path)
    mine = sorted((r for r in states if r['steamid'] == steamid), key=lambda r: r['tick'])
    slots = {r['slot'] for r in mine}
    series = {k: np.array([r[k] if r[k] is not None else np.nan for r in mine], float)
              for k in ('yaw', 'pitch', 'x', 'y', 'health', 'punch0', 'punch1')}
    series['tick'] = np.array([r['tick'] for r in mine], int)
    series['alive'] = np.array([r['alive'] for r in mine], int)
    ev = {'weapon_fire': np.sort([e['tick'] for e in events if e['name'] == 'weapon_fire' and e['user'] in slots]),
          'player_hurt': np.sort([e['tick'] for e in events if e['name'] == 'player_hurt' and e['attacker'] in slots])}
    return series, ev


def event_offsets(plugin_ticks, demo_ticks, max_gap=8):
    """For each plugin event, demo_tick - plugin_tick of the nearest demo event (|d| <= max_gap)
    after removing the global clock offset (most common difference)."""
    if not len(plugin_ticks) or not len(demo_ticks):
        return None, {}
    diffs = Counter()
    for t in plugin_ticks:
        j = np.searchsorted(demo_ticks, t)
        for k in (j - 1, j):
            if 0 <= k < len(demo_ticks):
                diffs[int(demo_ticks[k] - t)] += 1
    clock = diffs.most_common(1)[0][0]
    local = Counter()
    for t in plugin_ticks:
        j = np.searchsorted(demo_ticks, t + clock)
        best = min((abs(int(demo_ticks[k] - t - clock)), int(demo_ticks[k] - t - clock))
                   for k in (j - 1, j) if 0 <= k < len(demo_ticks))
        if best[0] <= max_gap:
            local[best[1]] += 1
    return clock, dict(sorted(local.items()))


def state_agreement(p, d, clock):
    """Compare fields at plugin tick t vs demo tick t + clock + shift, for shifts -3..3."""
    out = {}
    alive = p['alive'] > 0
    for shift in range(-3, 4):
        idx = np.searchsorted(d['tick'], p['tick'] + clock + shift)
        ok = alive & (idx < len(d['tick']))
        ok[ok] &= d['tick'][idx[ok]] == p['tick'][ok] + clock + shift
        di, pi = idx[ok], np.flatnonzero(ok)
        yaw = np.abs(wrap(p['yaw'][pi] - d['yaw'][di]))
        out[shift] = {'n': int(len(pi)), 'yaw_abs_diff_median': float(np.median(yaw)) if len(yaw) else None,
                      'yaw_abs_diff_p99': float(np.percentile(yaw, 99)) if len(yaw) else None}
    best = min((s for s in out if out[s]['n']), key=lambda s: out[s]['yaw_abs_diff_median'])
    idx = np.searchsorted(d['tick'], p['tick'] + clock + best)
    ok = alive & (idx < len(d['tick']))
    ok[ok] &= d['tick'][idx[ok]] == p['tick'][ok] + clock + best
    di, pi = idx[ok], np.flatnonzero(ok)
    fields = {}
    for k in ('yaw', 'pitch', 'x', 'y', 'punch0', 'punch1', 'health'):
        a, b = p[k][pi], d[k][di]
        good = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(wrap(a[good] - b[good]) if k == 'yaw' else a[good] - b[good])
        fields[k] = {'n': int(good.sum()),
                     'abs_diff_median': float(np.median(diff)) if len(diff) else None,
                     'abs_diff_p99': float(np.percentile(diff, 99)) if len(diff) else None,
                     'exact_fraction': float(np.mean(diff < 1e-3)) if len(diff) else None}
    return best, {s: v['yaw_abs_diff_median'] for s, v in out.items()}, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('capture')
    parser.add_argument('demo')
    parser.add_argument('--steamid', type=int, required=True)
    args = parser.parse_args()
    p, pe = capture_series(args.capture, args.steamid)
    d, de = demo_series(args.demo, args.steamid)
    report = {'plugin_rows': int(len(p['tick'])), 'demo_rows': int(len(d['tick'])),
              'events': {k: {'plugin': int(len(pe[k])), 'demo': int(len(de[k]))} for k in pe}}
    clock, fire = event_offsets(pe['weapon_fire'], de['weapon_fire'])
    _, hurt = event_offsets(pe['player_hurt'], de['player_hurt'] - 0) if clock is not None else (None, {})
    report['clock_offset_demo_minus_plugin'] = clock
    report['weapon_fire_residual_offsets'] = fire
    if clock is not None and len(pe['player_hurt']) and len(de['player_hurt']):
        local = Counter()
        for t in pe['player_hurt']:
            j = np.searchsorted(de['player_hurt'], t + clock)
            cands = [int(de['player_hurt'][k] - t - clock) for k in (j - 1, j) if 0 <= k < len(de['player_hurt'])]
            best = min(cands, key=abs)
            if abs(best) <= 8:
                local[best] += 1
        report['player_hurt_residual_offsets'] = dict(sorted(local.items()))
    if clock is not None and len(p['tick']) and len(d['tick']):
        best, by_shift, fields = state_agreement(p, d, clock)
        report['state_best_shift'] = best
        report['yaw_median_diff_by_shift'] = by_shift
        report['fields_at_best_shift'] = fields
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
