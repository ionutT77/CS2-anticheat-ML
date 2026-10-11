import json

import pytest

from realtime.capture_schema import SCHEMA_VERSION, STATE_FIELDS, read_capture, validate
from realtime.check_capture import by_slot, leakage_check, run, velocity_check

HIT = 300


def state(tick, slot, x=0.0, health=100, **extra):
    row = dict(type='state', tick=tick, slot=slot, steamid=0, yaw=10.0, pitch=0.0, x=x, y=0.0,
               z=0.0, avx=0.0, avy=0.0, avz=0.0, alive=1, warmup=0, team=2 + slot, round=1,
               scoped=0, duck=0.0, flags=1, health=health, armor=0, recoil=0.0, punch0=0.0,
               punch1=0.0, shots=0, clip=30, weapon='ak47')
    row.update(extra)
    return row


def capture(tmp_path, drop_tick=HIT, tickrate=64):
    rows = [dict(type='header', schema=SCHEMA_VERSION, tickrate=tickrate)]
    for tick in range(HIT - 10, HIT + 5):
        rows.append(state(tick, 0, x=tick * 250 / 64, avx=250.0))
        rows.append(state(tick, 1, health=100 if tick < drop_tick else 70))
    rows.append(dict(type='event', name='player_hurt', tick=HIT, attacker=0, victim=1,
                     weapon='ak47', dmg_health=30, health=70))
    path = tmp_path / 'c.ndjson'
    path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    return path


def test_aligned_capture_passes(tmp_path):
    report = run(capture(tmp_path))
    assert report['ok'], report
    assert report['C1_leakage']['first_visible_tick_offset'] == {0: 1}
    assert report['C2_velocity']['abs_diff_logged_vs_derived_p50_p90_p99'][0] < 0.01


def test_state_labeled_one_tick_early_is_leakage(tmp_path):
    report = run(capture(tmp_path, drop_tick=HIT - 1))
    assert report['C1_leakage']['failed'] == 1
    assert not report['ok']


def test_earlier_health_dip_inside_window_fails(tmp_path):
    _, states, events = read_capture(capture(tmp_path))
    for row in states:
        if row['slot'] == 1 and row['tick'] == HIT - 5:
            row['health'] = 60  # lower than the pre-hit value of 100 well before T
    assert leakage_check(by_slot(states), events)['failed'] == 1


def test_team_damage_is_not_counted(tmp_path):
    _, states, events = read_capture(capture(tmp_path))
    for row in states:
        row['team'] = 2
    result = leakage_check(by_slot(states), events)
    assert result['team_damage'] == 1 and result['passed'] == 0


def test_bots_with_same_steamid_stay_separate(tmp_path):
    _, states, _ = read_capture(capture(tmp_path))
    assert set(by_slot(states)) == {0, 1}


def test_wrong_tickrate_and_missing_field_are_reported(tmp_path):
    header, states, events = read_capture(capture(tmp_path, tickrate=32))
    del states[0]['armor']
    problems = validate(header, states, events)
    assert any('tickrate' in p for p in problems) and any('armor' in p for p in problems)


def test_nonfinite_value_is_reported(tmp_path):
    header, states, events = read_capture(capture(tmp_path))
    states[3]['yaw'] = float('nan')
    assert any('nonfinite' in p for p in validate(header, states, events))


def test_unreadable_punch_is_reported_not_zero_filled(tmp_path):
    header, states, events = read_capture(capture(tmp_path))
    states[0]['punch0'] = None
    assert any('punch0' in p for p in validate(header, states, events))


def test_velocity_mismatch_is_visible(tmp_path):
    _, states, _ = read_capture(capture(tmp_path))
    for row in states:
        row['avx'] = 0.0  # logged velocity ignores real movement
    assert velocity_check(by_slot(states))['abs_diff_logged_vs_derived_p50_p90_p99'][2] > 200  # p99: one of two players moves


def test_schema_covers_all_fields_the_checks_read():
    assert {'tick', 'slot', 'health', 'flags', 'avx', 'x'} <= set(STATE_FIELDS)
