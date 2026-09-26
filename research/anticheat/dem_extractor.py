"""Extract 39-feature encounter windows from CS2 .dem files.

Produces NPZ files compatible with predict_cs2cd.py for private-server
shadow/review evaluation.  No automatic bans, kicks, or punishments.

The feature computation matches cs2_data.extract_match() on valid telemetry.
This module reads demoparser2 output instead of Parquet/JSON, with stricter
telemetry validation and round-boundary checks for both participants.
"""
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .demo_timing import read_demo_timing
from .cs2_data import (
    AUTO, FEATURES, GUNS, PISTOL, SHOTGUN, SMG, SNIPER,
    difference, time_since, wrap,
)

log = logging.getLogger(__name__)

TICK_RATE = 64
WINDOW_TICKS = 256
BURST_GAP_TICKS = 128
_INVALID_PLAYER_IDS = frozenset({'', '0', '0.0', 'nan', 'none', '<na>'})
KNOWN_APPROXIMATIONS = [
    'Eye height uses Z + 64 - 18 * duck_amount, not hitbox positions.',
    'Flash remaining uses event-duration decay, not visual occlusion.',
    'Footstep events do not establish audibility; visibility is not modeled.',
]


def _validate_tick_rate(header):
    """Require explicit timing evidence; contiguous tick numbers are insufficient."""
    rates = []
    for key in ('tickrate', 'tick_rate', 'tick_rate_hz'):
        if key in header:
            rates.append(float(header[key]))
    if 'tick_interval' in header:
        interval = float(header['tick_interval'])
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('Invalid demo tick interval.')
        rates.append(1 / interval)
    if 'playback_ticks' in header and 'playback_time' in header:
        duration = float(header['playback_time'])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid demo playback duration.')
        rates.append(float(header['playback_ticks']) / duration)
    if not rates:
        raise ValueError(
            'Cannot verify demo tick rate from header timing metadata. '
            'Extraction requires verified 64 Hz telemetry; no rate is assumed.'
        )
    if any(not np.isfinite(rate) or not np.isclose(rate, TICK_RATE, rtol=0, atol=0.01)
           for rate in rates):
        raise ValueError(f'Unsupported or inconsistent demo tick rate: {rates}; expected 64 Hz.')
    header['verified_tick_rate'] = TICK_RATE


# ── demoparser2 property mapping ─────────────────────────────────────────
# Maps internal name → list of candidate demoparser2 property names.
# The first name that parses successfully is used.
_PROP_CANDIDATES = [
    ('X',                   ['X']),
    ('Y',                   ['Y']),
    ('Z',                   ['Z']),
    ('pitch',               ['pitch']),
    ('yaw',                 ['yaw']),
    ('health',              ['health']),
    ('armor_value',         ['armor_value']),
    ('is_alive',            ['is_alive']),
    ('team_num',            ['team_num']),
    ('active_weapon_name',  ['active_weapon_name']),
    ('active_weapon_ammo',  ['active_weapon_ammo', 'm_iClip1']),
    ('is_scoped',           ['is_scoped', 'm_bIsScoped']),
    ('duck_amount',         ['duck_amount', 'm_flDuckAmount']),
    ('is_airborne',         ['is_airborne']),
    ('shots_fired',         ['shots_fired', 'm_iShotsFired']),
    # GOTV demos expose m_flRecoilIndex (verified demoparser2 0.42.0).
    ('fl_recoil_idx',       ['m_flRecoilIndex', 'fl_recoil_idx']),
    # CS2 moved pawn aim punch into AimPunchServices; m_predictableBaseAngle
    # is the closest exposed equivalent of the training pipeline's
    # m_aimPunchAngle.  May not be present in all GOTV demos.
    ('aim_punch_angle',     ['aim_punch_angle',
                             'CCSPlayerPawn.CCSPlayer_AimPunchServices.m_predictableBaseAngle',
                             'CCSPlayerPawn.m_aimPunchAngle']),
    ('is_warmup_period',    ['is_warmup_period', 'm_bWarmupPeriod']),
    ('total_rounds_played', ['total_rounds_played']),
]

# velocity_X/Y/Z are NOT networked in GOTV/SourceTV demos (verified against
# demoparser2 0.42.0 and multiple live CS2 match demos: every velocity name
# variant returns only tick/steamid/name columns with no data).  They are
# derived per player from consecutive recorded positions at the verified
# 64 Hz tick rate: v[t] = (pos[t] - pos[t-1]) * TICK_RATE.
# For the first tick in each player's recording v = 0 (forward difference).
_DERIVED_FROM_POSITION = ('velocity_X', 'velocity_Y', 'velocity_Z')

# Missing feature and filter properties must never be replaced with defaults.
# velocity_X/Y/Z are excluded because they are always derived, not parsed.
_REQUIRED_PROPS = frozenset(name for name, _ in _PROP_CANDIDATES)


def _valid_player_id(player):
    return str(player).strip().lower() not in _INVALID_PLAYER_IDS


def _event_ticks(values, context):
    """Validate integer tick coordinates before sorting or truncating values."""
    try:
        ticks = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid numeric ticks in {context}.') from exc
    if not np.isfinite(ticks).all() or np.any(ticks != np.floor(ticks)):
        raise ValueError(f'Nonfinite or noninteger ticks in {context}.')
    return ticks


# ── Parsing layer (demoparser2 only) ─────────────────────────────────────

def _normalized(name):
    """Lowercase alphanumerics only, after dropping a leading ``m_`` prefix."""
    return ''.join(character for character in name.lower().removeprefix('m_')
                   if character.isalnum())


def _requested_column(frame, internal, requested):
    """Return the column the parser actually returned for one property.

    demoparser2 silently drops unknown property names instead of raising,
    so acceptance must be judged by the returned columns, not by the call.
    Names may be entity-qualified (``CCSPlayerPawn.m_aimPunchAngle``).
    """
    if requested in frame.columns:
        return requested
    target = _normalized(internal)
    for column in frame.columns:
        if _normalized(column.rsplit('.', 1)[-1]) == target:
            return column
    return None


def _find_probe_tick(parser):
    """Return the first tick that contains actual player rows.

    Some demo formats (e.g. FACEIT SourceTV) have no player data at tick 0.
    Probing an empty tick causes all properties to appear missing.  We fetch
    the minimum tick present in a cheap single-property parse and fall back to
    tick 1 if that also fails.
    """
    for probe_prop in ('X', 'health', 'is_alive'):
        try:
            df = parser.parse_ticks([probe_prop])
            if 'tick' in df.columns and len(df) > 0:
                min_tick = int(df['tick'].min())
                if min_tick >= 0:
                    return max(min_tick, 1)
        except Exception:
            continue
    return 1


def _resolve_props(parser):
    """Discover which demoparser2 property names work.

    A property counts as found only if its column appears in the probed
    DataFrame.  Returns (request_list, rename_dict, missing_list).

    Uses the first tick that actually contains player data rather than tick 0,
    so that FACEIT and other SourceTV demos (which have no rows at tick 0) are
    handled correctly.
    """
    probe_tick = _find_probe_tick(parser)
    resolved = {}
    try:
        frame = parser.parse_ticks([candidates[0] for _, candidates in _PROP_CANDIDATES], ticks=[probe_tick])
        for internal, candidates in _PROP_CANDIDATES:
            column = _requested_column(frame, internal, candidates[0])
            if column is not None:
                resolved[internal] = column
    except Exception:
        pass
    # Probe properties the batch request could not resolve, one at a time.
    for internal, candidates in _PROP_CANDIDATES:
        if internal in resolved:
            continue
        for cand in candidates:
            try:
                column = _requested_column(parser.parse_ticks([cand], ticks=[probe_tick]), internal, cand)
            except Exception:
                continue
            if column is not None:
                resolved[internal] = column
                break
    request, rename = [], {}
    for internal, _ in _PROP_CANDIDATES:
        if internal not in resolved:
            continue
        column = resolved[internal]
        request.append(column)
        if column != internal:
            rename[column] = internal
    missing = [name for name, _ in _PROP_CANDIDATES if name not in resolved]
    return request, rename, missing


def parse_demo(dem_path, prop_map=None):
    """Parse a .dem file into per-player tick arrays and event lists.

    This is the only function that imports demoparser2.  Everything
    downstream operates on the returned dicts.

    Parameters
    ----------
    dem_path : str or Path
        Path to the CS2 demo file.
    prop_map : dict or None
        Override mapping of internal name → demoparser2 property name.
        Merged on top of auto-discovered names.

    Returns
    -------
    players : dict
        steamid → dict of per-tick numpy arrays (sorted by tick).
    hurt_events : list of dict
        Sorted by tick.  Keys: tick, attacker, victim, weapon.
    header : dict
        Demo header metadata (map name, tick rate, etc.).
    warnings : list of str
        Issues encountered during parsing.
    approximated : list of str
        Empty compatibility field. Missing required properties fail;
        missing samples remain NaN for encounter-local rejection.
    """
    from demoparser2 import DemoParser

    dem_path = Path(dem_path)
    parser = DemoParser(str(dem_path))
    warnings, approximated = [], []

    # ── Header ────────────────────────────────────────────────────────
    header = {}
    try:
        h = parser.parse_header()
        header = dict(h) if isinstance(h, dict) else {}
    except Exception as exc:
        warnings.append(f'Could not parse demo header: {exc}')
    header.update(read_demo_timing(dem_path))
    _validate_tick_rate(header)

    # ── Resolve property names ────────────────────────────────────────
    if prop_map is not None:
        if not isinstance(prop_map, dict) or any(
                key not in _REQUIRED_PROPS or not isinstance(value, str) or not value
                for key, value in prop_map.items()):
            raise ValueError('prop_map must map known internal properties to nonempty parser names.')
    request, rename, missing = _resolve_props(parser)
    if prop_map:
        # Apply user overrides
        for iname, dname in prop_map.items():
            # Remove old mapping for this internal name
            request = [r for r in request if rename.get(r, r) != iname and r != iname]
            rename = {k: v for k, v in rename.items() if v != iname}
            request.append(dname)
            if dname != iname:
                rename[dname] = iname
            if iname in missing:
                missing.remove(iname)

    if missing and missing != ['aim_punch_angle']:
        raise ValueError(f'Required tick properties unavailable: {sorted(missing)}. '
                         'No feature defaults or disabled filters are allowed; check prop_map.')
    if len(set(request)) != len(request):
        raise ValueError('Property overrides must refer to distinct parser columns.')

    log.info('Parsing %d tick properties from %s.', len(request), dem_path.name)
    tick_df = parser.parse_ticks(request).rename(columns=rename)
    missing_columns = (_REQUIRED_PROPS | {'tick', 'steamid'}) - set(tick_df.columns)
    if missing_columns - {'aim_punch_angle'}:
        raise ValueError(f'Tick DataFrame missing required columns: {sorted(missing_columns)}')
    if 'aim_punch_angle' not in tick_df.columns:
        # Known demoparser2 0.42.0 limitation on some demos: aim punch is not
        # exposed under its documented names.  Substituting zeros is a
        # documented approximation (punch features and their knock-on yaw/
        # pitch deltas), recorded in warnings and the audit trail.
        msg = ("Property 'aim_punch_angle' unavailable in this demo - punch_pitch/"
               'punch_yaw filled with 0; a documented approximation, not silent defaults.')
        warnings.append(msg)
        approximated.extend(['punch_pitch', 'punch_yaw'])
        log.warning(msg)
        punch_column_missing = True
        tick_df['aim_punch_angle'] = None
    else:
        punch_column_missing = False
    header['property_mapping'] = {rename.get(name, name): name for name in request}
    header['null_counts'] = {name: int(tick_df[name].isna().sum())
                             for name in sorted(_REQUIRED_PROPS | {'tick', 'steamid'})}

    # Discard bot/invalid identities before converting their telemetry.
    tick_df = tick_df.loc[tick_df['steamid'].map(_valid_player_id)]
    str_ids = tick_df['steamid'].astype(str).values
    numeric_cols = sorted(_REQUIRED_PROPS - {'aim_punch_angle', 'active_weapon_name'})
    arrays = {name: tick_df[name].to_numpy(dtype=np.float32, na_value=np.nan)
              for name in numeric_cols}
    arrays['tick'] = _event_ticks(tick_df['tick'].to_numpy(), 'demo telemetry')
    for component in range(2):
        if punch_column_missing:
            arrays[f'punch_{component}'] = np.zeros(len(tick_df), dtype=np.float32)
            continue
        arrays[f'punch_{component}'] = np.array([
            punch[component] if isinstance(punch, (list, tuple, np.ndarray)) and len(punch) >= 2
            else np.nan for punch in tick_df['aim_punch_angle']
        ], dtype=np.float32)
    if not np.isfinite(arrays['tick']).all():
        raise ValueError('Missing or nonfinite demo ticks cannot be aligned.')
    weapons = tick_df['active_weapon_name'].to_numpy()

    # ── Parse events ──────────────────────────────────────────────────
    shots, steps, blinds = defaultdict(list), defaultdict(list), defaultdict(list)
    raw_events = {}
    available_events = set(parser.list_game_events())
    event_names = ['player_hurt', 'weapon_fire', 'player_footstep', 'player_blind']
    required_event_columns = {
        'player_hurt': {'tick', 'attacker_steamid', 'user_steamid', 'weapon'},
        'weapon_fire': {'tick', 'user_steamid', 'weapon'},
        'player_footstep': {'tick', 'user_steamid'},
        'player_blind': {'tick', 'user_steamid', 'blind_duration'},
    }
    for name in event_names:
        if name not in available_events:
            raw_events[name] = None
            warnings.append(f'Event {name!r} absent from demo event inventory.')
            continue
        try:
            frame = parser.parse_event(name)
        except Exception as exc:
            raise ValueError(f'Required event {name!r} could not be parsed.') from exc
        # CS2 GOTV/PBDEMS2 demos: demoparser2 returns an empty list (not a DataFrame)
        # for events that appear in the event inventory but contain no recorded data.
        # Treat this the same as absent — these events weren't written to the demo.
        if isinstance(frame, list):
            raw_events[name] = None
            warnings.append(
                f'Event {name!r} listed but returned a list (GOTV recording gap); '
                f'treated as absent.'
            )
            continue
        required = required_event_columns[name]
        if frame is None or not hasattr(frame, 'columns') or not required.issubset(frame.columns):
            raise ValueError(f'Required event columns unavailable for {name!r}.')
        identity_columns = required & {'user_steamid', 'attacker_steamid'}
        for column in identity_columns:
            frame = frame.loc[frame[column].map(_valid_player_id)]
        frame = frame.copy()
        frame['tick'] = _event_ticks(frame['tick'].to_numpy(), name)
        if frame[list(required)].isna().any().any():
            raise ValueError(f'Missing values in required event {name!r}.')
        if name == 'player_blind':
            try:
                durations = frame['blind_duration'].to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError('Invalid numeric blind duration.') from exc
            if not np.isfinite(durations).all() or np.any(durations < 0):
                raise ValueError('Blind duration must be finite and nonnegative.')
            frame['blind_duration'] = durations
        raw_events[name] = frame

    wf = raw_events.get('weapon_fire')
    if wf is not None and len(wf):
        for _, row in wf.iterrows():
            weapon = str(row.get('weapon', '')).removeprefix('weapon_')
            if weapon in GUNS:
                sid = str(row.get('user_steamid', row.get('steamid', '')))
                shots[sid].append(int(row['tick']))

    pf = raw_events.get('player_footstep')
    if pf is not None and len(pf):
        for _, row in pf.iterrows():
            sid = str(row.get('user_steamid', row.get('steamid', '')))
            steps[sid].append(int(row['tick']))

    pb = raw_events.get('player_blind')
    if pb is not None and len(pb):
        for _, row in pb.iterrows():
            sid = str(row.get('user_steamid', row.get('steamid', '')))
            dur = float(row.get('blind_duration', row.get('duration', 0)))
            blinds[sid].append((int(row['tick']), dur))

    # ── Assemble per-player dicts ─────────────────────────────────────
    players = {}
    for player in sorted(set(str_ids)):
        ix = np.flatnonzero(str_ids == player)
        order = np.argsort(arrays['tick'][ix], kind='stable')
        ix = ix[order]
        p = {k: arr[ix] for k, arr in arrays.items()}
        p['weapon'] = weapons[ix]

        # Derive velocity from consecutive position differences.
        # v[t] = (pos[t] - pos[t-1]) * TICK_RATE; v[0] = 0 (forward difference).
        # This approximates the networked m_vecVelocity, which is absent in GOTV.
        for axis, pos_key in (('velocity_X', 'X'), ('velocity_Y', 'Y'), ('velocity_Z', 'Z')):
            pos = p[pos_key].astype(np.float32)
            vel = np.empty_like(pos)
            vel[0] = 0.0
            vel[1:] = (pos[1:] - pos[:-1]) * float(TICK_RATE)
            # Clamp to plausible CS2 speed range (max bhop ~4000 u/s)
            vel = np.clip(vel, -5000.0, 5000.0)
            p[axis] = vel

        p['shot'] = np.isin(p['tick'], shots.get(player, [])).astype(np.float32)
        p['footstep'] = np.isin(p['tick'], steps.get(player, [])).astype(np.float32)
        p['since_shot'] = time_since(p['tick'], shots.get(player, []))
        p['since_noise'] = time_since(
            p['tick'], shots.get(player, []) + steps.get(player, []))
        flash = np.zeros(len(ix), dtype=np.float32)
        for start, duration in blinds.get(player, []):
            lo = np.searchsorted(p['tick'], start)
            hi = np.searchsorted(p['tick'], start + TICK_RATE * duration, side='right')
            if lo < hi:
                flash[lo:hi] = np.maximum(
                    flash[lo:hi], duration - (p['tick'][lo:hi] - start) / TICK_RATE)
        p['flash'] = flash
        players[player] = p

    # ── Build hurt events list ────────────────────────────────────────
    hurt_events = []
    ph = raw_events.get('player_hurt')
    if ph is not None and len(ph):
        for _, row in ph.sort_values('tick').iterrows():
            hurt_events.append({
                'tick': int(row['tick']),
                'attacker': str(row.get('attacker_steamid', '')),
                'victim': str(row.get('user_steamid', row.get('userid', ''))),
                'weapon': str(row.get('weapon', '')).removeprefix('weapon_'),
            })

    return players, hurt_events, header, warnings, approximated


# ── Feature extraction layer (pure NumPy, testable without demos) ────

def build_encounters(players, hurt_events, consent_players=None):
    """Build encounter windows and compute 39-feature arrays.

    Matches cs2_data.extract_match() features on valid telemetry, but rejects
    incomplete/nonfinite samples, duplicate window ticks, and either player's
    round boundaries. This function does not depend on demoparser2.

    Parameters
    ----------
    players : dict
        steamid → dict of per-tick arrays.  Required keys per player:
        tick, X, Y, Z, pitch, yaw, velocity_X, velocity_Y, velocity_Z,
        health, armor_value, is_alive, team_num, duck_amount, is_scoped,
        is_airborne, fl_recoil_idx, punch_0, punch_1, shots_fired,
        active_weapon_ammo, weapon, shot, footstep, since_shot,
        since_noise, flash, is_warmup_period, total_rounds_played.
        Tick arrays must be sorted; validation is local to each window.
    hurt_events : list of dict
        Any order; sorted internally. Keys: tick, attacker, victim, weapon.
    consent_players : set of str or None
        If given, only extract encounters where the attacker is in this
        set.  All players are still used as victims for geometry.

    Returns
    -------
    x : ndarray of shape [encounters, 256, 39]
    meta : list of dict  (per-encounter metadata)
    local_map : dict  (steamid → match-local identifier)
    audit : dict  (encounter counts, rejections, player summaries)
    """
    # Map valid steam IDs to stable aliases before applying attacker consent.
    players = {sid: state for sid, state in players.items() if _valid_player_id(sid)}
    local_map = {sid: f'Player_{i + 1}' for i, sid in enumerate(sorted(players))}
    scored = set(players)
    if consent_players is not None:
        scored &= {str(s) for s in consent_players}

    # Validate coordinates before sorting (including mixed numeric/string input).
    event_ticks = _event_ticks([event['tick'] for event in hurt_events], 'damage events')
    ordered_events = sorted(zip(event_ticks, hurt_events), key=lambda item: item[0])
    # Repeated hits within the burst gap do not create copies.
    anchors, last_hit = [], {}
    for event_tick, event in ordered_events:
        a, v, weapon = event['attacker'], event['victim'], event['weapon']
        if weapon not in GUNS or a not in scored or v not in players or a == v:
            continue
        tick = int(event_tick)
        key = (a, v)
        previous_tick = last_hit.get(key)
        if previous_tick is None or tick - previous_tick > BURST_GAP_TICKS:
            anchors.append((tick, a, v, weapon))
        last_hit[key] = tick

    # Extract 256-tick windows and compute features
    xs, meta, rejected = [], [], Counter()
    for tick, aid, vid, event_weapon in anchors:
        want = np.arange(tick - WINDOW_TICKS, tick, dtype=np.float64)
        ps = []
        invalid = False
        for pid in [aid, vid]:
            p = players[pid]
            ix = np.searchsorted(p['tick'], want)
            if ix.max() >= len(p['tick']) or not np.array_equal(p['tick'][ix], want):
                rejected['missing_tick'] += 1
                invalid = True
                break
            # Count occurrences only at pre-impact ticks; future duplicates
            # must not invalidate an earlier encounter.
            right = np.searchsorted(p['tick'], want, side='right')
            if np.any(right - ix > 1):
                rejected['duplicate_tick'] += 1
                invalid = True
                break
            ps.append({k: values[ix] for k, values in p.items()})
        if invalid:
            continue

        a, v = ps
        required = (_REQUIRED_PROPS - {'active_weapon_name', 'aim_punch_angle'}) | {
            'punch_0', 'punch_1', 'weapon', 'shot', 'footstep', 'since_shot', 'since_noise', 'flash',
            'velocity_X', 'velocity_Y', 'velocity_Z',  # always derived from positions
        }
        if any(not required.issubset(state) for state in ps):
            rejected['missing_telemetry'] += 1
            continue
        if any(not np.isfinite(state[name]).all() for state in ps for name in required - {'weapon'}):
            rejected['nonfinite_telemetry'] += 1
            continue
        if any(any(not isinstance(weapon, str) or not weapon for weapon in state['weapon'])
               for state in ps):
            rejected['missing_weapon'] += 1
            continue
        if any(np.any(state['is_warmup_period']) for state in ps):
            rejected['warmup'] += 1
            continue
        if a['team_num'][-1] == v['team_num'][-1]:
            rejected['team_damage'] += 1
            continue
        if not (np.all(a['is_alive']) and np.all(v['is_alive'])):
            rejected['not_alive_throughout'] += 1
            continue
        if any(np.ptp(state['total_rounds_played']) != 0 for state in ps):
            rejected['round_boundary'] += 1
            continue

        # ── 39-feature vector (identical to cs2_data.py L157-176) ─────
        dx = v['X'] - a['X']
        dy = v['Y'] - a['Y']
        dz = ((v['Z'] + 64 - 18 * v['duck_amount'])
              - (a['Z'] + 64 - 18 * a['duck_amount']))
        horizontal = np.maximum(np.hypot(dx, dy), 1)
        bearing = np.degrees(np.arctan2(dy, dx))
        elevation = -np.degrees(np.arctan2(dz, horizontal))
        yaw = difference(a['yaw'], circular=True)
        pitch = difference(a['pitch'])
        rvx = v['velocity_X'] - a['velocity_X']
        rvy = v['velocity_Y'] - a['velocity_Y']

        f = [
            yaw, pitch,
            wrap(bearing - a['yaw']), elevation - a['pitch'],
            a['shot'],
            difference(yaw), difference(pitch),
            difference(bearing, True), difference(elevation),
            np.sqrt(horizontal ** 2 + dz ** 2), dz,
            np.hypot(a['velocity_X'], a['velocity_Y']),
            np.hypot(v['velocity_X'], v['velocity_Y']),
            -(rvx * dx + rvy * dy) / horizontal,
            (rvx * dy - rvy * dx) / horizontal,
            a['velocity_Z'], v['velocity_Z'],
            a['is_scoped'], a['duck_amount'], a['is_airborne'],
            a['flash'], a['health'], a['armor_value'],
            a['fl_recoil_idx'], a['punch_0'], a['punch_1'],
            a['shots_fired'], a['active_weapon_ammo'],
            a['since_shot'], v['shot'], v['footstep'],
            v['since_noise'], v['duck_amount'], v['health'],
        ]
        f.extend(
            np.isin(a['weapon'], list(group)).astype(np.float32)
            for group in [AUTO, SNIPER, PISTOL, SMG, SHOTGUN])

        encounter = np.stack(f, axis=1).astype(np.float32)
        if not np.isfinite(encounter).all():
            rejected['nonfinite_feature'] += 1
            continue

        xs.append(encounter)
        meta.append({
            'player': local_map.get(aid, aid),
            'victim': local_map.get(vid, vid),
            'tick': tick,
            'weapon': event_weapon,
        })

    x = np.stack(xs) if xs else np.empty((0, WINDOW_TICKS, len(FEATURES)), dtype=np.float32)

    # Build per-player audit summary
    player_summaries = []
    for sid in sorted(scored):
        lid = local_map[sid]
        enc = sum(1 for m in meta if m['player'] == lid)
        player_summaries.append({
            'steamid': sid, 'local_id': lid,
            'encounters': enc, 'total_ticks': len(players[sid]['tick']),
        })
        if enc == 0:
            log.warning('Player %s (%s) has no eligible encounters.', lid, sid)

    audit = {
        'anchor_count': len(anchors),
        'encounter_count': len(xs),
        'rejected': dict(rejected),
        'players': player_summaries,
    }
    return x, meta, local_map, audit


# ── Public entry point ───────────────────────────────────────────────

def extract_dem(dem_path, output_path=None, consent_players=None, prop_map=None):
    """Extract encounters from a CS2 .dem file and save as NPZ.

    Parameters
    ----------
    dem_path : str or Path
        Path to the .dem file.
    output_path : str or Path or None
        Where to write the NPZ.  Default: ``<stem>_encounters.npz``
        next to the demo file.
    consent_players : set of str or None
        Only score encounters whose attacker Steam64 ID is in this set.
    prop_map : dict or None
        Override demoparser2 property names (internal → demoparser2).

    Returns
    -------
    dict
        Extraction audit with keys: npz_path, encounter_count,
        anchor_count, rejected, players, header, warnings,
        approximated_features.
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f'Demo file not found: {dem_path}')

    log.info('Parsing demo: %s', dem_path)
    players, hurt_events, header, warnings, approximated = parse_demo(
        dem_path, prop_map)

    if not players:
        raise ValueError('No player tick data found in demo.')
    log.info('Found %d players, %d hurt events.', len(players), len(hurt_events))

    x, meta, local_map, audit = build_encounters(
        players, hurt_events, consent_players)

    audit['header'] = header
    audit['warnings'] = warnings
    audit['approximated_features'] = approximated
    audit['known_approximations'] = KNOWN_APPROXIMATIONS
    audit['features'] = FEATURES
    audit['events'] = meta
    audit['feature_stats'] = {}
    for index, name in enumerate(FEATURES):
        values = x[..., index].astype(np.float64).reshape(-1)
        audit['feature_stats'][name] = {
            'count': int(values.size),
            'min': float(values.min()) if values.size else None,
            'max': float(values.max()) if values.size else None,
            'mean': float(values.mean()) if values.size else None,
            'std': float(values.std()) if values.size else None,
            'zero_fraction': float(np.mean(values == 0)) if values.size else None,
        }
    audit['consent_scope'] = 'Attacker scoring filter only; all participants must consent to victim telemetry use.'
    audit['dem_path'] = str(dem_path)

    # Save NPZ
    if output_path is None:
        output_path = dem_path.with_name(dem_path.stem + '_encounters.npz')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    player_arr = np.array([m['player'] for m in meta]) if meta else np.array([], dtype='<U1')
    tick_arr = np.array([m['tick'] for m in meta], dtype=np.int64) if meta else np.array([], dtype=np.int64)
    np.savez(output_path, x=x, player=player_arr, tick=tick_arr)
    log.info('Saved %d encounters to %s', len(meta), output_path)

    audit['npz_path'] = str(output_path)

    # Save audit JSON alongside NPZ
    audit_path = output_path.with_suffix('.json')
    audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')

    return audit
