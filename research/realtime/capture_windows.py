"""Phase 2: build training-format encounter windows from a plugin capture.

Converts a capture (capture_schema v1) into the per-player arrays expected by
`anticheat.dem_extractor.build_encounters`, which applies the training rejection rules
and computes the 39 features with the same code as training. Output is an NPZ readable
by `scripts/predict_cs2cd.py` (x [N, 256, 39], player, tick) plus an audit JSON.

Timing (PHASE0_PARITY.md, section 5b): plugin event stamps are usually one tick earlier
than the demo, while state samples align exactly. So:
  * hit anchor  = first tick, at or after the plugin stamp, where the victim's health drops
                  (matched the demo hit tick 88/88); hits without a visible drop are dropped;
  * shot tick   = tick where the shooter's m_iShotsFired increases (matched 207/207);
                  otherwise plugin tick + 1;
  * footstep/blind = plugin tick + 1 (majority offset; not validated, flagged in the audit).

Bots (SteamID 0) are never scored. They can be included as *victims* only with
`include_bot_victims`, keyed by slot; training never had bot victims, so windows built
that way test the pipeline, not live performance.

Run from `research/`:
    python realtime/capture_windows.py capture.ndjson --output out/encounters.npz [--include-bot-victims]
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anticheat.cs2_data import FEATURES, GUNS, time_since
from anticheat.dem_extractor import TICK_RATE, build_encounters
from realtime.capture_schema import FL_ONGROUND, read_capture, validate

ANCHOR_SEARCH = 2   # ticks after a plugin stamp searched for the matching state change
EVENT_OFFSET = 1    # plugin -> demo offset used when no state signature exists


def _player_id(row, include_bot_victims):
    if row['steamid']:
        return str(row['steamid'])
    return f"bot:{row['slot']}" if include_bot_victims else None


def _owners(states, include_bot_victims):
    """slot -> (sorted ticks, ids) so an event's slot can be resolved at its tick."""
    per_slot = defaultdict(list)
    for r in states:
        per_slot[r['slot']].append((r['tick'], _player_id(r, include_bot_victims)))
    out = {}
    for slot, rows in per_slot.items():
        rows.sort(key=lambda item: item[0])
        out[slot] = (np.array([t for t, _ in rows]), [i for _, i in rows])
    return out


def _owner_at(owners, slot, tick):
    if slot not in owners:
        return None
    ticks, ids = owners[slot]
    k = np.searchsorted(ticks, tick, side='right') - 1
    return ids[k] if k >= 0 else ids[0]


def _arrays(rows, velocity):
    rows.sort(key=lambda r: r['tick'])
    col = lambda k, dtype=np.float64: np.array(
        [np.nan if r[k] is None else r[k] for r in rows], dtype=dtype)
    p = {'tick': np.array([r['tick'] for r in rows], dtype=np.int64),
         'X': col('x'), 'Y': col('y'), 'Z': col('z'), 'pitch': col('pitch'), 'yaw': col('yaw'),
         'health': col('health'), 'armor_value': col('armor'), 'is_alive': col('alive'),
         'team_num': col('team'), 'duck_amount': col('duck'), 'is_scoped': col('scoped'),
         'is_airborne': ((np.array([r['flags'] for r in rows], dtype=np.int64) & FL_ONGROUND) == 0)
         .astype(np.float64),
         'fl_recoil_idx': col('recoil'), 'punch_0': col('punch0'), 'punch_1': col('punch1'),
         'shots_fired': col('shots'), 'active_weapon_ammo': col('clip'),
         'is_warmup_period': col('warmup'), 'total_rounds_played': col('round'),
         'weapon': np.array([r['weapon'] or '' for r in rows], dtype=object)}
    if velocity == 'logged':
        p['velocity_X'], p['velocity_Y'], p['velocity_Z'] = col('avx'), col('avy'), col('avz')
    else:  # forward difference, as dem_extractor.py does for GOTV
        for axis, pos in (('velocity_X', 'X'), ('velocity_Y', 'Y'), ('velocity_Z', 'Z')):
            vel = np.zeros_like(p[pos])
            vel[1:] = np.clip(np.diff(p[pos]) * TICK_RATE, -5000, 5000)
            p[axis] = vel
    return p


def _state_change_tick(p, start, changed):
    """First tick in [start, start + ANCHOR_SEARCH] whose sample satisfies `changed(prev, cur)`."""
    for t in range(start, start + ANCHOR_SEARCH + 1):
        k = np.searchsorted(p['tick'], t)
        if 0 < k < len(p['tick']) and p['tick'][k] == t and p['tick'][k - 1] == t - 1 \
                and changed(p, k - 1, k):
            return t
    return None


def build_from_capture(path, include_bot_victims=False, velocity='logged', attackers=None):
    """Return (x, meta, local_map, audit) for one capture."""
    header, states, events = read_capture(path)
    problems = validate(header, states, events)
    owners = _owners(states, include_bot_victims)

    rows = defaultdict(list)
    for r in states:
        pid = _player_id(r, include_bot_victims)
        if pid is not None:
            rows[pid].append(r)
    players = {pid: _arrays(items, velocity) for pid, items in rows.items()}
    humans = {pid for pid in players if not pid.startswith('bot:')}
    scored = humans if attackers is None else humans & {str(a) for a in attackers}

    stats = Counter()
    shots, steps, blinds = defaultdict(list), defaultdict(list), defaultdict(list)
    hurts = []
    for e in sorted(events, key=lambda e: e['tick']):
        name, t = e['name'], int(e['tick'])
        if name == 'weapon_fire':
            pid = _owner_at(owners, e['user'], t)
            if pid not in players or str(e['weapon']).removeprefix('weapon_') not in GUNS:
                continue
            tick = _state_change_tick(players[pid], t,
                                      lambda p, a, b: p['shots_fired'][b] > p['shots_fired'][a])
            stats['shot_anchored_by_counter' if tick is not None else 'shot_offset_fallback'] += 1
            shots[pid].append(tick if tick is not None else t + EVENT_OFFSET)
        elif name == 'player_footstep':
            pid = _owner_at(owners, e['user'], t)
            if pid in players:
                steps[pid].append(t + EVENT_OFFSET)
        elif name == 'player_blind':
            pid = _owner_at(owners, e['user'], t)
            if pid in players:
                blinds[pid].append((t + EVENT_OFFSET, float(e['blind_duration'])))
        elif name == 'player_hurt':
            attacker = _owner_at(owners, e['attacker'], t) if e['attacker'] >= 0 else None
            victim = _owner_at(owners, e['victim'], t)
            if attacker is None or victim not in players:
                stats['hurt_unresolved_player'] += 1
                continue
            tick = _state_change_tick(players[victim], t,
                                      lambda p, a, b: p['health'][b] < p['health'][a])
            if tick is None:
                stats['hurt_dropped_no_health_drop'] += 1
                continue
            stats['hurt_anchored_by_health_drop'] += 1
            hurts.append({'tick': tick, 'attacker': attacker, 'victim': victim,
                          'weapon': str(e['weapon']).removeprefix('weapon_')})

    for pid, p in players.items():
        p['shot'] = np.isin(p['tick'], shots[pid]).astype(np.float32)
        p['footstep'] = np.isin(p['tick'], steps[pid]).astype(np.float32)
        p['since_shot'] = time_since(p['tick'], shots[pid])
        p['since_noise'] = time_since(p['tick'], shots[pid] + steps[pid])
        flash = np.zeros(len(p['tick']), dtype=np.float32)
        for start, duration in blinds[pid]:
            lo = np.searchsorted(p['tick'], start)
            hi = np.searchsorted(p['tick'], start + TICK_RATE * duration, side='right')
            if lo < hi:
                flash[lo:hi] = np.maximum(flash[lo:hi], duration - (p['tick'][lo:hi] - start) / TICK_RATE)
        p['flash'] = flash

    x, meta, local_map, audit = build_encounters(players, hurts, consent_players=scored)
    audit.update({
        'capture': str(path), 'plugin': header.get('plugin'), 'schema': header.get('schema'),
        'truncated_tail': header.get('truncated_tail'), 'schema_problems': problems[:10],
        'velocity_source': velocity, 'include_bot_victims': include_bot_victims,
        'scored_attackers': len(scored), 'anchoring': dict(stats),
        'approximations': ['footstep and blind ticks use plugin tick + 1 (not validated)',
                           'bot victims are keyed by slot' if include_bot_victims else 'no bot victims'],
        'content_sha256': content_hash(x, meta),
    })
    return x, meta, local_map, audit


def content_hash(x, meta):
    """Hash of the array contents. np.savez embeds timestamps, so file bytes differ between
    identical builds; determinism is defined on this content hash instead."""
    h = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    h.update(json.dumps([[m['player'], int(m['tick'])] for m in meta]).encode())
    return h.hexdigest()


def save(x, meta, audit, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    player = np.array([m['player'] for m in meta]) if meta else np.array([], dtype='<U1')
    tick = np.array([m['tick'] for m in meta], dtype=np.int64)
    np.savez(output, x=x, player=player, tick=tick)
    audit = dict(audit, npz_path=str(output), features=FEATURES, events=meta)
    output.with_suffix('.json').write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('capture')
    parser.add_argument('--output', required=True)
    parser.add_argument('--include-bot-victims', action='store_true',
                        help='allow bots as victims (pipeline testing only; never as attackers)')
    parser.add_argument('--velocity', choices=['logged', 'derived'], default='logged')
    args = parser.parse_args()
    x, meta, _, audit = build_from_capture(args.capture, args.include_bot_victims, args.velocity)
    out = save(x, meta, audit, args.output)
    print(json.dumps({k: audit[k] for k in ('encounter_count', 'anchor_count', 'rejected', 'anchoring',
                                            'scored_attackers', 'truncated_tail', 'content_sha256')},
                     indent=2))
    print(f'saved {out}')


if __name__ == '__main__':
    main()
