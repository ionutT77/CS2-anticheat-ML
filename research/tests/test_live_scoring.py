import json

import numpy as np

from realtime.capture_windows import build_from_capture
from realtime.live_scoring import CaptureFollower, LiveWindows, Scorer
from test_capture_windows import HIT, capture


def stream(path, chunk):
    """Feed a capture to LiveWindows in chunks of `chunk` lines; return emitted encounters."""
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    live, out = LiveWindows(include_bot_victims=True), []
    for i in range(0, len(rows), chunk):
        live.add(rows[i:i + chunk])
        out += live.emit()
    return out


def test_live_windows_equal_batch_windows(tmp_path):
    path = capture(tmp_path)
    batch, meta, _, _ = build_from_capture(path, include_bot_victims=True)
    for chunk in (7, 101, 10_000):
        live = stream(path, chunk)
        assert len(live) == len(batch) == 1
        assert np.array_equal(live[0][0], batch[0]) and live[0][1]['tick'] == meta[0]['tick'] == HIT


def test_encounter_is_not_emitted_before_the_drop_is_visible(tmp_path):
    rows = [json.loads(l) for l in capture(tmp_path).read_text(encoding='utf-8').splitlines()]
    events = [r for r in rows if r['type'] != 'state']
    states = [r for r in rows if r['type'] == 'state']
    live = LiveWindows(include_bot_victims=True)
    live.add(events + [s for s in states if s['tick'] < HIT])   # health drop not yet captured
    assert live.emit() == []
    live.add([s for s in states if s['tick'] >= HIT])
    assert len(live.emit()) == 1


def test_each_encounter_is_emitted_once(tmp_path):
    rows = [json.loads(l) for l in capture(tmp_path).read_text(encoding='utf-8').splitlines()]
    live = LiveWindows(include_bot_victims=True)
    live.add(rows)
    assert len(live.emit()) == 1
    live.add([dict(rows[-3], tick=1500 + k) for k in range(5)])   # more state, no new hit
    assert live.emit() == []


def test_follower_holds_back_a_partial_line(tmp_path):
    path = tmp_path / 'grow.ndjson'
    path.write_text('{"type":"header","schema":1,"tickrate":64}\n{"type":"sta', encoding='utf-8')
    follower = CaptureFollower(path)
    assert [r['type'] for r in follower.poll()] == ['header']
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write('te","tick":1}\n')
    assert [r['tick'] for r in follower.poll()] == [1]


def test_scorer_runs_the_existing_model_on_a_live_window(tmp_path):
    (window, record), = stream(capture(tmp_path), 500)
    result = Scorer().score(window, record)
    assert result['encounters'] == 1 and 0.0 <= result['calibrated_score'] <= 1.0
    assert result['event_scores'] and all(0.0 <= s <= 1.0 for s in result['event_scores'].values())
