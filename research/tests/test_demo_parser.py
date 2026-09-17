"""Mock the demoparser2 boundary; no real .dem files are exercised here."""
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from anticheat import dem_extractor


@pytest.fixture
def parser_boundary(monkeypatch):
    ticks = pd.DataFrame({name: [0., 0., 0.] for name in dem_extractor._REQUIRED_PROPS})
    ticks['tick'] = [3, 1, 2]
    ticks['steamid'] = ['76561198000000001'] * 3
    ticks['active_weapon_name'] = ['AK-47'] * 3
    ticks['aim_punch_angle'] = [[3., 30., 0.], [1., 10., 0.], [2., 20., 0.]]
    state = {
        'ticks': ticks, 'header': {'map_name': 'synthetic'},
        'timing': {'playback_ticks': 640, 'playback_time': 10., 'timing_source': 'mock'},
        'events': {}, 'unavailable': set(), 'failed_events': set(), 'requests': [],
    }

    class DemoParser:
        def __init__(self, path):
            assert str(path).endswith('.dem')

        def parse_header(self):
            return state['header'].copy()

        def parse_ticks(self, props, ticks=None):
            state['requests'].append((props, ticks))
            if set(props) & state['unavailable']:
                raise ValueError('Unsupported property')
            if ticks is not None:
                return pd.DataFrame(columns=['tick', 'steamid', *props])
            return state['ticks'].copy()

        def list_game_events(self):
            return list(state['events'])

        def parse_event(self, name):
            if name in state['failed_events']:
                raise RuntimeError('Mock parser failure')
            frame = state['events'][name]
            return frame.copy() if frame is not None else None

    module = ModuleType('demoparser2')
    module.DemoParser = DemoParser
    monkeypatch.setitem(sys.modules, 'demoparser2', module)
    monkeypatch.setattr(dem_extractor, 'read_demo_timing', lambda path: state['timing'].copy())
    return state


def _parse():
    return dem_extractor.parse_demo('synthetic.dem')


def _event(name, tick=2):
    row = {'tick': tick, 'user_steamid': '76561198000000001'}
    if name == 'player_hurt':
        row['attacker_steamid'] = '76561198000000002'
    if name in ('weapon_fire', 'player_hurt'):
        row['weapon'] = 'weapon_ak47'
    if name == 'player_blind':
        row['blind_duration'] = 1.
    return row


EVENT_NAMES = ('weapon_fire', 'player_footstep', 'player_blind', 'player_hurt')


def test_vector_punch_sorted_conversion_and_absent_events(parser_boundary):
    players, hurt, header, warnings, approximated = _parse()
    state = players['76561198000000001']
    np.testing.assert_array_equal(state['tick'], [1, 2, 3])
    np.testing.assert_array_equal(state['punch_0'], [1, 2, 3])
    np.testing.assert_array_equal(state['punch_1'], [10, 20, 30])
    assert state['punch_0'].dtype == np.float32
    assert header['verified_tick_rate'] == 64
    assert header['property_mapping']['aim_punch_angle'] == 'aim_punch_angle'
    assert hurt == approximated == []
    assert len(warnings) == 4
    assert all('absent from demo event inventory' in warning for warning in warnings)
    np.testing.assert_array_equal(state['since_shot'], [5., 5., 5.])


@pytest.mark.parametrize('invalid_id', ['0', 'nan', 'None', '<NA>', None, np.nan, pd.NA])
def test_invalid_id_rows_and_events_are_skipped(parser_boundary, invalid_id):
    parser_boundary['ticks'].loc[0, 'steamid'] = invalid_id
    parser_boundary['events']['player_hurt'] = pd.DataFrame([
        {**_event('player_hurt'), 'attacker_steamid': invalid_id},
        {**_event('player_hurt'), 'user_steamid': invalid_id},
    ])
    players, hurt, _, _, _ = _parse()
    assert list(players) == ['76561198000000001']
    np.testing.assert_array_equal(players['76561198000000001']['tick'], [1, 2])
    assert hurt == []


@pytest.mark.parametrize('property_name', ['health', 'is_warmup_period', 'aim_punch_angle'])
def test_unavailable_property_fails_without_defaults(parser_boundary, property_name):
    parser_boundary['unavailable'] = set(dict(dem_extractor._PROP_CANDIDATES)[property_name])
    with pytest.raises(ValueError, match='Required tick properties unavailable'):
        _parse()


@pytest.mark.parametrize('property_name', ['health', 'is_warmup_period', 'aim_punch_angle', 'tick', 'steamid'])
def test_missing_output_column_fails_even_if_probe_succeeds(parser_boundary, property_name):
    parser_boundary['ticks'] = parser_boundary['ticks'].drop(columns=property_name)
    with pytest.raises(ValueError, match='missing required columns'):
        _parse()


def test_nan_telemetry_preserved(parser_boundary):
    parser_boundary['ticks']['health'] = pd.array([np.nan, 100., 100.], dtype='Float32')
    parser_boundary['ticks'].at[0, 'aim_punch_angle'] = None
    players, _, header, _, approximated = _parse()
    state = players['76561198000000001']
    assert np.isnan(state['health'][-1])
    assert np.isnan(state['punch_0'][-1]) and np.isnan(state['punch_1'][-1])
    assert header['null_counts']['health'] == header['null_counts']['aim_punch_angle'] == 1
    assert approximated == []


@pytest.mark.parametrize('tick', [np.nan, np.inf, 'invalid', 2.5])
def test_invalid_telemetry_tick_fails(parser_boundary, tick):
    parser_boundary['ticks']['tick'] = [1, tick, 3]
    with pytest.raises(ValueError, match='tick'):
        _parse()


@pytest.mark.parametrize('name', EVENT_NAMES)
def test_present_but_failed_event_is_not_absent(parser_boundary, name):
    parser_boundary['events'][name] = pd.DataFrame([_event(name)])
    parser_boundary['failed_events'].add(name)
    with pytest.raises(ValueError, match='could not be parsed'):
        _parse()


@pytest.mark.parametrize('name', EVENT_NAMES)
@pytest.mark.parametrize('tick', [np.nan, np.inf, -np.inf, 'invalid', 2.5])
def test_event_ticks_must_be_finite_numeric_integers(parser_boundary, name, tick):
    parser_boundary['events'][name] = pd.DataFrame([_event(name, tick)])
    with pytest.raises(ValueError, match='tick'):
        _parse()


@pytest.mark.parametrize('duration', [np.nan, np.inf, -np.inf, -1., 'invalid'])
def test_blind_duration_must_be_finite_nonnegative(parser_boundary, duration):
    parser_boundary['events']['player_blind'] = pd.DataFrame([
        {**_event('player_blind'), 'blind_duration': duration}])
    with pytest.raises(ValueError, match='duration|required event'):
        _parse()


@pytest.mark.parametrize('frame', [None, pd.DataFrame(), pd.DataFrame({'tick': [2]})])
def test_present_event_missing_schema_fails(parser_boundary, frame):
    parser_boundary['events']['weapon_fire'] = frame
    with pytest.raises(ValueError, match='event columns unavailable'):
        _parse()


def test_present_empty_events_with_schema_are_valid(parser_boundary):
    parser_boundary['events'] = {
        name: pd.DataFrame(columns=_event(name)) for name in EVENT_NAMES
    }
    _, hurt, _, warnings, _ = _parse()
    assert hurt == warnings == []


@pytest.mark.parametrize('timing,header,valid', [
    ({'playback_ticks': 640, 'playback_time': 10.}, {}, True),
    ({'playback_ticks': 1280, 'playback_time': 10.}, {}, False),
    ({}, {}, False),
    ({}, {'tickrate': 64}, True),
    ({}, {'tick_rate': 128}, False),
    ({}, {'tick_interval': 1 / 64}, True),
    ({'playback_ticks': 640, 'playback_time': 10.}, {'tickrate': 128}, False),
    ({'playback_ticks': 640, 'playback_time': 0.}, {}, False),
    ({}, {'tickrate': np.nan}, False),
])
def test_tick_rate_requires_consistent_explicit_64hz(parser_boundary, timing, header, valid):
    parser_boundary['timing'] = timing
    parser_boundary['header'] = header
    if valid:
        assert _parse()[2]['verified_tick_rate'] == 64
    else:
        with pytest.raises(ValueError, match='tick rate|duration'):
            _parse()
        assert parser_boundary['requests'] == []


def test_unordered_events_and_future_events_are_causal(parser_boundary):
    parser_boundary['events'] = {
        name: pd.DataFrame([_event(name, tick) for tick in (100, 2)]) for name in EVENT_NAMES
    }
    first_players, hurt, _, warnings, _ = _parse()
    state = first_players['76561198000000001']
    assert [event['tick'] for event in hurt] == [2, 100]
    assert all(event['weapon'] == 'ak47' for event in hurt)
    assert warnings == []
    np.testing.assert_array_equal(state['shot'], [0., 1., 0.])
    np.testing.assert_array_equal(state['footstep'], [0., 1., 0.])
    np.testing.assert_allclose(state['since_shot'], [5., 0., 1 / 64])
    np.testing.assert_allclose(state['since_noise'], [5., 0., 1 / 64])
    np.testing.assert_allclose(state['flash'], [0., 1., 1 - 1 / 64])
    parser_boundary['events'] = {
        name: pd.DataFrame([_event(name, 2)]) for name in EVENT_NAMES
    }
    second_players, _, _, _, _ = _parse()
    for name in state:
        np.testing.assert_array_equal(state[name], second_players['76561198000000001'][name])


def test_property_alias_fallback(parser_boundary):
    parser_boundary['unavailable'] = {'aim_punch_angle'}
    parser_boundary['ticks'] = parser_boundary['ticks'].rename(
        columns={'aim_punch_angle': 'CCSPlayerPawn.m_aimPunchAngle'})
    players, _, header, _, _ = _parse()
    assert header['property_mapping']['aim_punch_angle'] == 'CCSPlayerPawn.m_aimPunchAngle'
    np.testing.assert_array_equal(players['76561198000000001']['punch_1'], [10, 20, 30])


def test_explicit_property_override(parser_boundary):
    parser_boundary['ticks'] = parser_boundary['ticks'].rename(columns={'health': 'custom_health'})
    players, _, header, _, _ = dem_extractor.parse_demo('synthetic.dem', {'health': 'custom_health'})
    assert header['property_mapping']['health'] == 'custom_health'
    assert 'health' in players['76561198000000001']


@pytest.mark.parametrize('mapping', [{'unknown': 'health'}, {'health': ''}, {'health': 1}, []])
def test_invalid_property_override_fails(parser_boundary, mapping):
    with pytest.raises(ValueError, match='prop_map'):
        dem_extractor.parse_demo('synthetic.dem', mapping)
