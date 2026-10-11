"""Phase 3: score encounters while a capture is being written (or replay a finished one).

Follows the plugin's NDJSON capture, keeps a rolling per-session buffer, and whenever a hit
becomes resolvable (its health drop is visible) builds windows with the Phase 2 code
(`capture_windows.build_from_rows`) and scores new encounters with the existing,
checksum-verified model functions:
  * player-match score: `anticheat.cs2_inference.predict` over all of that player's
    encounters so far (calibrated score + policy flags);
  * per-encounter evidence: `anticheat.private_scoring.predict_encounters` event score.

REVIEW SIGNALS ONLY. Thresholds were calibrated on CS2CD player-matches; this population,
the number of encounters and (with bot victims) the opponents differ. Nothing here is a
cheating verdict, and nothing kicks, bans or reports anyone.

Run from `research/`:
    python realtime/live_scoring.py <capture.ndjson | plugin folder> [--include-bot-victims]
    python realtime/live_scoring.py <finished capture> --replay
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from realtime.capture_windows import ANCHOR_SEARCH, build_from_rows

BUFFER_TICKS = 2048   # ~32 s: window (256) + 5 s time-since look-back (320) + burst history
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / 'experiments' / 'cs2cd_v1'
DISCLAIMER = ('Review signals only: scores are calibrated on CS2CD, not on this server, '
              'and are not cheating verdicts. No automatic action is taken.')


class CaptureFollower:
    """Reads complete lines appended to a growing file; keeps a partial last line pending."""

    def __init__(self, path):
        self.path, self.position, self.pending = Path(path), 0, ''

    def poll(self):
        with open(self.path, encoding='utf-8', errors='replace') as handle:
            handle.seek(self.position)
            data = handle.read()
            self.position = handle.tell()
        lines = (self.pending + data).split('\n')
        self.pending = lines.pop()
        return [json.loads(line) for line in lines if line.strip()]


class LiveWindows:
    """Turns a stream of capture rows into newly final encounter windows.

    An encounter is emitted once the newest pending hit is resolvable (state exists up to
    tick + ANCHOR_SEARCH + 1). Windows only use state before their anchor and earlier hits
    of the same pair, so an emitted encounter never changes later."""

    def __init__(self, include_bot_victims=False, velocity='logged'):
        self.include_bot_victims, self.velocity = include_bot_victims, velocity
        self.header, self.states, self.events = None, [], []
        self.latest_tick, self.pending_hurts, self.emitted = -1, [], set()
        self.read_time = {}

    def add(self, rows, now=None):
        now = time.perf_counter() if now is None else now
        for r in rows:
            kind = r.get('type')
            if kind == 'header':
                self.header = r
            elif kind == 'state':
                self.states.append(r)
                self.latest_tick = max(self.latest_tick, r['tick'])
            elif kind == 'event':
                self.events.append(r)
                if r['name'] == 'player_hurt':
                    self.pending_hurts.append(r['tick'])
                    self.read_time.setdefault(r['tick'], now)
        if self.states and self.states[0]['tick'] < self.latest_tick - 2 * BUFFER_TICKS:
            self._prune()

    def ready(self):
        return any(t + ANCHOR_SEARCH + 1 <= self.latest_tick for t in self.pending_hurts)

    def emit(self):
        """Return [(window [256,39], record)] for encounters not emitted before."""
        if self.header is None or not self.ready():
            return []
        x, meta, local_map, audit = build_from_rows(
            dict(self.header), self.states, self.events, self.include_bot_victims, self.velocity)
        alias_to_id = {alias: pid for pid, alias in local_map.items()}
        out = []
        for window, m in zip(x, meta):
            key = (alias_to_id[m['player']], m['tick'], alias_to_id.get(m['victim'], m['victim']))
            if key in self.emitted:
                continue
            self.emitted.add(key)
            out.append((window, {'attacker': key[0], 'victim': key[2], 'tick': m['tick'],
                                 'weapon': m['weapon'],
                                 'read_time': self._read_time(m['tick'])}))
        self.pending_hurts = [t for t in self.pending_hurts if t + ANCHOR_SEARCH + 1 > self.latest_tick]
        self._prune()
        return out

    def _read_time(self, anchor):
        # the hurt stamp is at anchor - ANCHOR_SEARCH .. anchor
        times = [v for t, v in self.read_time.items() if anchor - ANCHOR_SEARCH <= t <= anchor]
        return min(times) if times else None

    def _prune(self):
        floor = self.latest_tick - BUFFER_TICKS
        self.states = [r for r in self.states if r['tick'] >= floor]
        self.events = [e for e in self.events if e['tick'] >= floor]
        self.read_time = {t: v for t, v in self.read_time.items() if t >= floor}


class Scorer:
    """Accumulates encounters per attacker and scores them with the existing model code."""

    def __init__(self, model_dir=DEFAULT_MODEL):
        self.model_dir = str(model_dir)
        self.windows, self.records = defaultdict(list), defaultdict(list)

    def score(self, window, record):
        from anticheat.cs2_inference import predict
        from anticheat.private_scoring import predict_encounters
        player = record['attacker']
        self.windows[player].append(window)
        self.records[player].append(record)
        x = np.stack(self.windows[player]).astype(np.float32)
        ids = np.array([player] * len(x))
        ticks = np.array([r['tick'] for r in self.records[player]], dtype=np.int64)
        match = predict(x, ids, ticks, self.model_dir)[0]
        events = [{'player': player, 'tick': int(r['tick']), 'victim': r['victim'],
                   'weapon': r['weapon']} for r in self.records[player]]
        evidence = [row for row in predict_encounters(x, ids, ticks, events, self.model_dir)
                    if row['encounter_index'] == len(x) - 1]
        return {'encounters': match['encounters'], 'calibrated_score': match['calibrated_score'],
                'raw_score': match['raw_score'],
                'flags': sorted(k.removeprefix('flag_') for k, v in match.items()
                                if k.startswith('flag_') and v),
                'event_scores': {row['component']: row['event_score'] for row in evidence}}


def short(player):
    return player if player.startswith('bot:') else f'...{player[-4:]}'


def line(record, result, latency_ms, min_encounters):
    events = ', '.join(f'{c} {s:.3f}' for c, s in result['event_scores'].items()) or 'n/a'
    flags = ', '.join(result['flags']) or 'none'
    # Thresholds were calibrated on full matches; with few encounters the player-match score
    # is unstable (CS2CD: AUC 0.834 after 1 encounter vs 0.973 for full matches). Flags are
    # shown as provisional below `min_encounters`; the thresholds themselves are unchanged.
    if result['encounters'] < min_encounters:
        flags = f'provisional (<{min_encounters} encounters): {flags}'
    return (f"[{datetime.now():%H:%M:%S}] tick {record['tick']:>7} | {short(record['attacker'])} -> "
            f"{short(record['victim'])} ({record['weapon']}) | encounter score {events} | "
            f"player-match {result['calibrated_score']:.3f} after {result['encounters']} | "
            f"review flags: {flags} | {latency_ms:.0f} ms")


def resolve_capture(path):
    path = Path(path)
    if path.is_dir():
        captures = sorted(path.glob('capture_*.ndjson'), key=lambda p: p.stat().st_mtime)
        if not captures:
            raise SystemExit(f'no capture_*.ndjson in {path}')
        return captures[-1]
    return path


def run(capture, log_path, replay=False, include_bot_victims=False, poll=0.25, chunk_lines=2000,
        model_dir=DEFAULT_MODEL, idle_exit=None, min_encounters=10):
    follower, live, scorer = CaptureFollower(capture), LiveWindows(include_bot_victims), Scorer(model_dir)
    print(DISCLAIMER)
    print(f'capture: {capture} | mode: {"replay" if replay else "live"} | '
          f'bot victims: {"yes (pipeline test only)" if include_bot_victims else "no"}')
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    idle_since, buffered = time.perf_counter(), []
    with open(log_path, 'a', encoding='utf-8') as log:
        while True:
            rows = follower.poll() if not buffered else []
            if replay:
                buffered.extend(rows)
                rows, buffered = buffered[:chunk_lines], buffered[chunk_lines:]
            if rows:
                idle_since = time.perf_counter()
                live.add(rows)
                for window, record in live.emit():
                    result = scorer.score(window, record)
                    latency = (time.perf_counter() - record['read_time']) * 1000 if record['read_time'] else float('nan')
                    print(line(record, result, latency, min_encounters), flush=True)
                    log.write(json.dumps({**record, **result, 'processing_ms': latency,
                                          'provisional': result['encounters'] < min_encounters,
                                          'wall_time': datetime.now().isoformat(timespec='seconds')},
                                         default=str) + '\n')
                    log.flush()
            elif replay or (idle_exit and time.perf_counter() - idle_since > idle_exit):
                return scorer
            else:
                time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('capture', help='capture file, or the plugin folder (uses the newest capture)')
    parser.add_argument('--log', default='../local_runs/live_scores.jsonl')
    parser.add_argument('--replay', action='store_true', help='process a finished capture and exit')
    parser.add_argument('--include-bot-victims', action='store_true',
                        help='allow bots as victims (pipeline testing only; never as attackers)')
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL))
    parser.add_argument('--idle-exit', type=float, default=None,
                        help='live mode: exit after this many seconds without new data')
    parser.add_argument('--min-encounters', type=int, default=10,
                        help='show review flags as provisional below this many encounters')
    args = parser.parse_args()
    capture = resolve_capture(args.capture)
    try:
        run(capture, args.log, args.replay, args.include_bot_victims, model_dir=args.model_dir,
            idle_exit=args.idle_exit, min_encounters=args.min_encounters)
    except KeyboardInterrupt:
        print('stopped')


if __name__ == '__main__':
    main()
