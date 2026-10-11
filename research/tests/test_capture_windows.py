import json

import numpy as np

from anticheat.cs2_data import FEATURES
from realtime.capture_schema import SCHEMA_VERSION
from realtime.capture_windows import build_from_capture, content_hash, save

HUMAN = 76561190000000001
HIT = 1300          # tick at which the victim's health drops (demo-equivalent anchor)
START, END = 1000, 1400


def state(tick, slot, steamid, team, **over):
    row = dict(type='state', tick=tick, slot=slot, steamid=steamid,
               yaw=(tick * 0.37 + slot * 11) % 360 - 180, pitch=((tick % 50) - 25) * 0.1,
               x=slot * 300.0 + tick * 0.5, y=200.0 - tick * 0.25, z=0.0,
               avx=32.0, avy=-16.0, avz=0.0, alive=1, warmup=0, team=team, round=1,
               scoped=0, duck=0.0, flags=1, health=100, armor=100, recoil=0.0,
               punch0=-(tick % 7) * 0.1, punch1=(tick % 5) * 0.01, shots=0, clip=30, weapon='ak47')
    row.update(over)
    return row


def capture(tmp_path, name='c.ndjson', hurt_stamp=HIT - 1, victim_drop=HIT, victim_steamid=0,
            victim_team=2, mutate_after=None, extra_events=()):
    rows = [dict(type='header', schema=SCHEMA_VERSION, tickrate=64, plugin='test')]
    for t in range(START, END):
        human = state(t, 3, HUMAN, 3, shots=1 if t >= 1291 else 0)
        victim = state(t, 5, victim_steamid, victim_team,
                       health=70 if victim_drop is not None and t >= victim_drop else 100)
        if mutate_after is not None and t >= mutate_after:
            for r in (human, victim):
                r.update(yaw=-170.0, pitch=80.0, x=9e3, y=-9e3, z=500.0, avx=999.0, duck=1.0,
                         scoped=1, punch0=-30.0, recoil=50.0, clip=1, weapon='awp')
        rows += [human, victim]
    rows.append(dict(type='event', name='weapon_fire', tick=1290, user=3, weapon='weapon_ak47'))
    rows.append(dict(type='event', name='player_hurt', tick=hurt_stamp, attacker=3, victim=5,
                     weapon='ak47', dmg_health=30, health=70))
    rows += list(extra_events)
    path = tmp_path / name
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    return path


def build(path, **kw):
    kw.setdefault('include_bot_victims', True)
    return build_from_capture(path, **kw)


def test_window_is_anchored_on_the_health_drop_not_the_event_stamp(tmp_path):
    x, meta, _, audit = build(capture(tmp_path))
    assert x.shape == (1, 256, len(FEATURES))
    assert meta[0]['tick'] == HIT
    assert audit['anchoring']['hurt_anchored_by_health_drop'] == 1
    # last window row is tick HIT-1: victim health still pre-hit
    assert x[0, -1, FEATURES.index('victim_health')] == 100


def test_same_tick_stamp_is_also_anchored_on_the_drop(tmp_path):
    _, meta, _, _ = build(capture(tmp_path, hurt_stamp=HIT))
    assert meta[0]['tick'] == HIT


def test_nothing_at_or_after_the_hit_changes_the_window(tmp_path):
    """Causality: rewriting every state value from the hit tick on, and adding events stamped
    at or after T-1 (which shift to >= T), must leave the window bit-for-bit identical."""
    base, _, _, _ = build(capture(tmp_path, 'a.ndjson'))
    late = [dict(type='event', name=n, tick=t, user=u, **extra)
            for t in (HIT - 1, HIT, HIT + 5) for n, u, extra in (
                ('player_footstep', 5, {}), ('player_blind', 3, {'blind_duration': 2.5}),
                ('weapon_fire', 5, {'weapon': 'weapon_ak47'}))]
    changed, _, _, _ = build(capture(tmp_path, 'b.ndjson', mutate_after=HIT, extra_events=late))
    assert np.array_equal(base, changed)


def test_causality_check_is_sensitive_to_the_last_pre_hit_tick(tmp_path):
    """Negative control for the test above: changing tick T-1, which IS in the window,
    must change the output, otherwise the causality test would pass vacuously."""
    base, _, _, _ = build(capture(tmp_path, 'a.ndjson'))
    changed, _, _, _ = build(capture(tmp_path, 'b.ndjson', mutate_after=HIT - 1))
    assert changed.shape == base.shape and not np.array_equal(base, changed)
    assert np.array_equal(base[:, :-1], changed[:, :-1])  # only the last row differs


def test_shot_is_placed_where_the_shot_counter_increases(tmp_path):
    x, _, _, audit = build(capture(tmp_path))
    shot = x[0, :, FEATURES.index('shot')]
    assert shot[1291 - (HIT - 256)] == 1 and shot.sum() == 1
    assert audit['anchoring']['shot_anchored_by_counter'] == 1


def test_hit_without_a_health_drop_is_dropped_not_guessed(tmp_path):
    x, _, _, audit = build(capture(tmp_path, victim_drop=None))
    assert x.shape[0] == 0
    assert audit['anchoring']['hurt_dropped_no_health_drop'] == 1


def test_bot_victims_only_with_explicit_flag(tmp_path):
    path = capture(tmp_path)
    assert build(path, include_bot_victims=False)[0].shape[0] == 0
    assert build(path, include_bot_victims=True)[0].shape[0] == 1


def test_bots_are_never_scored_as_attackers(tmp_path):
    # victim is a consenting human, attacker slot 3 is made a bot
    path = capture(tmp_path, victim_steamid=HUMAN + 1)
    text = path.read_text(encoding='utf-8').replace(f'"steamid": {HUMAN}', '"steamid": 0')
    path.write_text(text, encoding='utf-8')
    x, _, _, audit = build(path)
    assert x.shape[0] == 0 and audit['scored_attackers'] == 1  # only the human victim id


def test_training_rules_still_apply(tmp_path):
    x, _, _, audit = build(capture(tmp_path, victim_team=3))  # same team as attacker
    assert x.shape[0] == 0 and audit['rejected'].get('team_damage') == 1


def test_same_capture_gives_same_content(tmp_path):
    path = capture(tmp_path)
    x1, m1, _, a1 = build(path)
    x2, m2, _, a2 = build(path)
    assert a1['content_sha256'] == a2['content_sha256'] == content_hash(x2, m2)
    out1 = save(x1, m1, a1, tmp_path / 'one.npz')
    out2 = save(x2, m2, a2, tmp_path / 'two.npz')
    d1, d2 = np.load(out1), np.load(out2)
    assert all(np.array_equal(d1[k], d2[k]) for k in ('x', 'player', 'tick'))


def test_output_matches_the_predictor_input_contract(tmp_path):
    x, meta, _, audit = build(capture(tmp_path))
    data = np.load(save(x, meta, audit, tmp_path / 'enc.npz'))
    assert data['x'].dtype == np.float32 and data['x'].shape[1:] == (256, 39)
    assert len(data['player']) == len(data['tick']) == data['x'].shape[0]
