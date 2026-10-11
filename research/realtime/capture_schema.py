"""Versioned NDJSON capture format written by the CounterStrikeSharp plugin.

One header line, then one `state` line per alive-or-dead player per server tick and one
line per game event. Raw values only: every model feature is derived later by the
existing Python code, so parity is checked on raw fields.
"""
import json
import math

SCHEMA_VERSION = 1
TICKRATE = 64

STATE_FIELDS = [
    'tick', 'slot', 'steamid', 'yaw', 'pitch', 'x', 'y', 'z', 'avx', 'avy', 'avz',
    'alive', 'warmup', 'team', 'round', 'scoped', 'duck', 'flags', 'health', 'armor',
    'recoil', 'punch0', 'punch1', 'shots', 'clip', 'weapon']
NUMERIC_STATE_FIELDS = [f for f in STATE_FIELDS if f != 'weapon']
# Plausible absolute bounds per field (game units/degrees). A finite but absurd value, such as
# a mis-read memory offset, must fail validation rather than flow into features.
RANGE_LIMITS = {
    'yaw': 3600, 'pitch': 3600, 'x': 1e5, 'y': 1e5, 'z': 1e5,
    'avx': 5000, 'avy': 5000, 'avz': 5000, 'duck': 1.5, 'health': 1000, 'armor': 1000,
    'recoil': 100, 'punch0': 180, 'punch1': 180, 'shots': 1000, 'clip': 1000}
EVENT_FIELDS = {
    'player_hurt': ['tick', 'attacker', 'victim', 'weapon', 'dmg_health', 'health'],
    'weapon_fire': ['tick', 'user', 'weapon'],
    'player_footstep': ['tick', 'user'],
    'player_blind': ['tick', 'user', 'blind_duration'],
}
FL_ONGROUND = 1  # Source 2 pawn flag bit 0


def read_capture(path):
    """Return (header, states, events). Raises ValueError on a malformed line, except that a
    partial final line (cut off by a server shutdown) is dropped and flagged on the header as
    `truncated_tail`."""
    header, states, events = None, [], []
    with open(path, encoding='utf-8') as handle:
        lines = [line for line in handle if line.strip()]
    truncated = False
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            # A server shutdown can cut the final write short; only the last line may be partial.
            if number == len(lines) and not line.endswith('\n'):
                truncated = True
                break
            raise ValueError(f'line {number}: invalid JSON') from error
        kind = row.get('type')
        if kind == 'header':
            header = row
        elif kind == 'state':
            states.append(row)
        elif kind == 'event':
            events.append(row)
        else:
            raise ValueError(f'line {number}: unknown type {kind!r}')
    if header is None:
        raise ValueError('missing header line')
    header['truncated_tail'] = truncated
    return header, states, events


def validate(header, states, events):
    """Return a list of problems; an empty list means the capture is schema-valid."""
    problems = []
    if header.get('schema') != SCHEMA_VERSION:
        problems.append(f"schema {header.get('schema')!r} != {SCHEMA_VERSION}")
    if header.get('tickrate') != TICKRATE:
        problems.append(f"tickrate {header.get('tickrate')!r} != {TICKRATE}")
    for i, row in enumerate(states):
        for field in STATE_FIELDS:
            if field not in row:
                problems.append(f'state {i}: missing {field}')
                break
        else:
            bad = [f for f in NUMERIC_STATE_FIELDS
                   if not isinstance(row[f], (int, float)) or not math.isfinite(row[f])]
            if bad:
                problems.append(f'state {i}: nonfinite or non-numeric {bad}')
            else:
                wild = [f for f, limit in RANGE_LIMITS.items() if abs(row[f]) > limit]
                if wild:
                    problems.append(f'state {i}: implausible magnitude {wild}')
    for i, row in enumerate(events):
        name = row.get('name')
        if name not in EVENT_FIELDS:
            problems.append(f'event {i}: unknown name {name!r}')
            continue
        missing = [f for f in EVENT_FIELDS[name] if f not in row]
        if missing:
            problems.append(f'event {i} ({name}): missing {missing}')
    return problems[:50]
