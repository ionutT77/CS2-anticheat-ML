"""CS2CD pre-impact encounter extraction. No fitting or labels in features.

Player IDs in this release are match-local. A player-match is the decision unit.
Angles use Source coordinates (positive pitch looks down). Head/eye positions are
approximations from origin and duck amount, not extracted hitboxes or line of sight.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

FEATURES = [
    'delta_yaw', 'delta_pitch', 'target_error_yaw', 'target_error_pitch', 'shot',
    'yaw_acceleration', 'pitch_acceleration', 'target_yaw_rate', 'target_pitch_rate',
    'distance', 'relative_height', 'attacker_speed', 'victim_speed',
    'closing_speed', 'lateral_speed', 'attacker_vertical_speed', 'victim_vertical_speed',
    'scoped', 'crouch', 'airborne', 'flash_remaining', 'health', 'armor',
    'recoil_index', 'punch_pitch', 'punch_yaw', 'shots_in_burst', 'ammo',
    'time_since_shot', 'victim_shot', 'victim_footstep', 'time_since_victim_noise',
    'victim_crouch', 'victim_health',
    'weapon_auto', 'weapon_sniper', 'weapon_pistol', 'weapon_smg', 'weapon_shotgun',
]
BASE_COLUMNS = ['tick', 'steamid', 'pitch', 'yaw', 'X', 'Y', 'Z', 'velocity_X',
    'velocity_Y', 'velocity_Z', 'is_alive', 'is_warmup_period', 'team_num',
    'total_rounds_played', 'is_scoped', 'duck_amount', 'is_airborne', 'health',
    'armor_value', 'fl_recoil_idx', 'aim_punch_angle', 'shots_fired',
    'active_weapon_ammo', 'active_weapon_name']
AUDIT_COLUMNS = ['usercmd_mouse_dx', 'usercmd_mouse_dy', 'usercmd_viewangle_x',
    'velocity', 'flash_duration', 'approximate_spotted_by', 'i_recoil_idx']
AUTO = {'ak47', 'm4a1', 'm4a1_silencer', 'galilar', 'sg556', 'aug', 'famas', 'm249', 'negev',
        'AK-47', 'M4A1-S', 'M4A4', 'Galil AR', 'SG 553', 'AUG', 'FAMAS', 'M249', 'Negev'}
SNIPER = {'g3sg1', 'ssg08', 'awp', 'scar20', 'G3SG1', 'SSG 08', 'AWP', 'SCAR-20'}
PISTOL = {'cz75a', 'deagle', 'elite', 'fiveseven', 'glock', 'hkp2000', 'p250', 'revolver',
          'tec9', 'usp_silencer', 'CZ75-Auto', 'Desert Eagle', 'Dual Berettas', 'Five-SeveN',
          'Glock-18', 'P2000', 'P250', 'R8 Revolver', 'Tec-9', 'USP-S'}
SMG = {'mac10', 'mp5sd', 'mp7', 'mp9', 'p90', 'bizon', 'ump45',
       'MAC-10', 'MP5-SD', 'MP7', 'MP9', 'P90', 'PP-Bizon', 'UMP-45'}
SHOTGUN = {'mag7', 'nova', 'sawedoff', 'xm1014', 'MAG-7', 'Nova', 'Sawed-Off', 'XM1014'}
GUNS = AUTO | SNIPER | PISTOL | SMG | SHOTGUN


def wrap(angle):
    return (angle + 180) % 360 - 180


def difference(a, circular=False):
    delta = np.diff(a, prepend=a[:1])
    return wrap(delta) if circular else delta


def time_since(ticks, events, cap=5.0):
    if not len(events):
        return np.full(len(ticks), cap, dtype=np.float32)
    events = np.asarray(sorted(events))
    ix = np.searchsorted(events, ticks, side='right') - 1
    return np.where(ix >= 0, np.minimum((ticks - events[np.maximum(ix, 0)]) / 64, cap), cap)


def extract_match(path_string):
    path = Path(path_string)
    match_id = path.parent.name + '/' + path.stem
    out = Path('data/cs2cd/processed/matches') / (path.parent.name + '_' + path.stem + '.npz')
    if out.exists():
        return json.loads(out.with_suffix('.json').read_text())
    events = json.loads(path.with_suffix('.json').read_text())
    labels = {r['steamid'] for r in events.get('cheaters', [])}
    pf = pq.ParquetFile(path)
    table = pf.read(columns=BASE_COLUMNS, use_threads=False).combine_chunks()
    nulls = {c: table[c].null_count for c in table.column_names}
    # Footer statistics permit audit of additional, deliberately unused fields.
    extra_nulls = {}
    for j in range(pf.metadata.num_columns):
        col = pf.metadata.row_group(0).column(j)
        name = col.path_in_schema
        if name in AUDIT_COLUMNS:
            extra_nulls[name] = sum(pf.metadata.row_group(g).column(j).statistics.null_count
                for g in range(pf.metadata.num_row_groups)
                if pf.metadata.row_group(g).column(j).statistics is not None)
    ids = table['steamid'].to_numpy()
    weapons = table['active_weapon_name'].to_numpy()
    arrays = {c: table[c].to_numpy().astype(np.float32) for c in BASE_COLUMNS
              if c not in ['steamid', 'active_weapon_name', 'aim_punch_angle']}
    for j in range(2):
        arrays[f'punch_{j}'] = pc.list_element(table['aim_punch_angle'], j).to_numpy().astype(np.float32)
    shots, steps, blinds = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in events.get('weapon_fire', []):
        if e.get('weapon', '').removeprefix('weapon_') in GUNS:
            shots[e.get('user_steamid')].append(e['tick'])
    for e in events.get('player_footstep', []):
        steps[e.get('user_steamid')].append(e['tick'])
    for e in events.get('player_blind', []):
        blinds[e.get('user_steamid')].append((e['tick'], e['blind_duration']))
    players = {}
    for player in sorted(set(ids)):
        if not player:
            continue
        ix = np.flatnonzero(ids == player)
        order = np.argsort(arrays['tick'][ix], kind='stable')
        ix = ix[order]
        p = {k: a[ix] for k, a in arrays.items()}
        p['weapon'] = weapons[ix]
        p['shot'] = np.isin(p['tick'], shots[player]).astype(np.float32)
        p['footstep'] = np.isin(p['tick'], steps[player]).astype(np.float32)
        p['since_shot'] = time_since(p['tick'], shots[player])
        p['since_noise'] = time_since(p['tick'], shots[player] + steps[player])
        flash = np.zeros(len(ix), dtype=np.float32)
        for start, duration in blinds[player]:
            lo = np.searchsorted(p['tick'], start)
            hi = np.searchsorted(p['tick'], start + 64 * duration, side='right')
            flash[lo:hi] = np.maximum(flash[lo:hi], duration - (p['tick'][lo:hi] - start) / 64)
        p['flash'] = flash
        players[player] = p
    # First damage in each attacker/victim burst; repeated hits within two seconds
    # do not create copies. Team damage, utility, bots and warmup are excluded.
    anchors, last_hit = [], {}
    for e in sorted(events.get('player_hurt', []), key=lambda e: e['tick']):
        a, v = e.get('attacker_steamid'), e.get('user_steamid')
        if e.get('weapon') not in GUNS or a not in players or v not in players or a == v:
            continue
        key = (a, v)
        if e['tick'] - last_hit.get(key, -100000) > 128:
            anchors.append((e['tick'], a, v, e['weapon']))
        last_hit[key] = e['tick']
    xs, meta, rejected = [], [], Counter()
    for tick, aid, vid, event_weapon in anchors:
        # Both histories end strictly BEFORE the first hit; no future-filled gaps.
        want = np.arange(tick - 256, tick, dtype=np.float32)
        ps = []
        invalid = False
        for player in [aid, vid]:
            p = players[player]
            ix = np.searchsorted(p['tick'], want)
            if ix.max() >= len(p['tick']) or not np.array_equal(p['tick'][ix], want):
                rejected['missing_tick'] += 1
                invalid = True
                break
            ps.append({k: val[ix] for k, val in p.items()})
        if invalid:
            continue
        a, v = ps
        if np.any(a['is_warmup_period']) or np.any(v['is_warmup_period']):
            rejected['warmup'] += 1
            continue
        if a['team_num'][-1] == v['team_num'][-1]:
            rejected['team_damage'] += 1
            continue
        if not (np.all(a['is_alive']) and np.all(v['is_alive'])):
            rejected['not_alive_throughout'] += 1
            continue
        if np.ptp(a['total_rounds_played']) != 0:
            rejected['round_boundary'] += 1
            continue
        dx, dy = v['X'] - a['X'], v['Y'] - a['Y']
        # Standard standing/crouching eye height approximation, equal for both.
        dz = (v['Z'] + 64 - 18 * v['duck_amount']) - (a['Z'] + 64 - 18 * a['duck_amount'])
        horizontal = np.maximum(np.hypot(dx, dy), 1)
        bearing = np.degrees(np.arctan2(dy, dx))
        elevation = -np.degrees(np.arctan2(dz, horizontal))
        yaw, pitch = difference(a['yaw'], True), difference(a['pitch'])
        rvx = v['velocity_X'] - a['velocity_X']
        rvy = v['velocity_Y'] - a['velocity_Y']
        f = [yaw, pitch, wrap(bearing - a['yaw']), elevation - a['pitch'], a['shot'],
             difference(yaw), difference(pitch), difference(bearing, True), difference(elevation),
             np.sqrt(horizontal**2 + dz**2), dz,
             np.hypot(a['velocity_X'], a['velocity_Y']), np.hypot(v['velocity_X'], v['velocity_Y']),
             -(rvx * dx + rvy * dy) / horizontal, (rvx * dy - rvy * dx) / horizontal,
             a['velocity_Z'], v['velocity_Z'], a['is_scoped'], a['duck_amount'], a['is_airborne'],
             a['flash'], a['health'], a['armor_value'], a['fl_recoil_idx'], a['punch_0'], a['punch_1'],
             a['shots_fired'], a['active_weapon_ammo'], a['since_shot'], v['shot'], v['footstep'],
             v['since_noise'], v['duck_amount'], v['health']]
        f.extend(np.isin(a['weapon'], list(group)).astype(np.float32)
                 for group in [AUTO, SNIPER, PISTOL, SMG, SHOTGUN])
        x = np.stack(f, axis=1).astype(np.float32)
        if not np.isfinite(x).all():
            rejected['nonfinite_feature'] += 1
            continue
        xs.append(x)
        meta.append({'match_id': match_id, 'player': aid, 'victim': vid, 'tick': tick,
                     'label': int(aid in labels), 'weapon': event_weapon,
                     'sha256': hashlib.sha256(x.tobytes()).hexdigest()})
    x = np.stack(xs) if xs else np.empty((0, 256, len(FEATURES)), dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=x, label=np.array([e['label'] for e in meta], dtype=np.int8),
             player=np.array([e['player'] for e in meta]), tick=np.array([e['tick'] for e in meta]))
    audit = {'match_id': match_id, 'raw_rows': len(ids), 'features': FEATURES,
             'events': meta, 'anchor_count': len(anchors), 'rejected': dict(rejected),
             'nulls': nulls, 'extra_nulls': extra_nulls,
             'players': [{'player': p, 'label': int(p in labels),
                          'encounters': sum(e['player'] == p for e in meta)} for p in players],
             'map': events.get('CSstats_info', [{}])[0].get('map', 'unknown')}
    out.with_suffix('.json').write_text(json.dumps(audit))
    return {k: audit[k] for k in ['match_id', 'raw_rows', 'anchor_count', 'rejected']}


def transform(x):
    """Fixed physical scales; subsequent mean/std are fitted on training only."""
    x = np.asarray(x, dtype=np.float32).copy()
    scales = {'delta_yaw': 1, 'delta_pitch': 1, 'target_error_yaw': 5, 'target_error_pitch': 5,
              'yaw_acceleration': 1, 'pitch_acceleration': 1, 'target_yaw_rate': 1,
              'target_pitch_rate': 1, 'distance': 500, 'relative_height': 100,
              'attacker_speed': 100, 'victim_speed': 100, 'closing_speed': 100,
              'lateral_speed': 100, 'attacker_vertical_speed': 100, 'victim_vertical_speed': 100,
              'health': 100, 'armor': 100, 'victim_health': 100, 'recoil_index': 5,
              'punch_pitch': 1, 'punch_yaw': 1, 'shots_in_burst': 5, 'ammo': 30}
    for i, feature in enumerate(FEATURES):
        if feature in scales:
            x[..., i] = np.arcsinh(x[..., i] / scales[feature])
    return np.clip(x, -12, 12)
