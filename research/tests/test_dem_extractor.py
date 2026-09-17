"""Focused tests for the private-server demo extraction pipeline.

Tests use synthetic player dicts to exercise build_encounters() without
requiring demoparser2 or a real .dem file.  The regression test uses the
included example NPZ to verify the inference path is unchanged.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from anticheat.cs2_data import FEATURES, wrap
from anticheat.dem_extractor import build_encounters

ROOT = Path(__file__).resolve().parents[1]


# ── Test fixtures ────────────────────────────────────────────────────

def _player_array(n, **overrides):
    """Create a default player state dict spanning *n* contiguous ticks."""
    defaults = dict(
        tick=np.arange(1, n + 1, dtype=np.float32),
        X=np.zeros(n, dtype=np.float32),
        Y=np.zeros(n, dtype=np.float32),
        Z=np.zeros(n, dtype=np.float32),
        pitch=np.zeros(n, dtype=np.float32),
        yaw=np.zeros(n, dtype=np.float32),
        velocity_X=np.zeros(n, dtype=np.float32),
        velocity_Y=np.zeros(n, dtype=np.float32),
        velocity_Z=np.zeros(n, dtype=np.float32),
        health=np.full(n, 100., dtype=np.float32),
        armor_value=np.full(n, 100., dtype=np.float32),
        is_alive=np.ones(n, dtype=np.float32),
        team_num=np.full(n, 2., dtype=np.float32),
        is_scoped=np.zeros(n, dtype=np.float32),
        duck_amount=np.zeros(n, dtype=np.float32),
        is_airborne=np.zeros(n, dtype=np.float32),
        fl_recoil_idx=np.zeros(n, dtype=np.float32),
        punch_0=np.zeros(n, dtype=np.float32),
        punch_1=np.zeros(n, dtype=np.float32),
        shots_fired=np.zeros(n, dtype=np.float32),
        active_weapon_ammo=np.full(n, 30., dtype=np.float32),
        is_warmup_period=np.zeros(n, dtype=np.float32),
        total_rounds_played=np.ones(n, dtype=np.float32),
        weapon=np.full(n, 'AK-47', dtype=object),
        shot=np.zeros(n, dtype=np.float32),
        footstep=np.zeros(n, dtype=np.float32),
        since_shot=np.full(n, 5.0, dtype=np.float32),
        since_noise=np.full(n, 5.0, dtype=np.float32),
        flash=np.zeros(n, dtype=np.float32),
    )
    defaults.update(overrides)
    return defaults


def _standard_fixture():
    """Two players, 300 ticks, one valid encounter at tick 257."""
    attacker = _player_array(300, team_num=np.full(300, 2., dtype=np.float32))
    victim = _player_array(300, team_num=np.full(300, 3., dtype=np.float32))
    # Place victim at a known position for geometry checks
    victim['X'][:] = -100.
    victim['Z'][:] = 100.
    players = {'A': attacker, 'B': victim}
    events = [{'tick': 257, 'attacker': 'A', 'victim': 'B', 'weapon': 'ak47'}]
    return players, events


# ── Feature ordering ─────────────────────────────────────────────────

def test_feature_order_matches_schema():
    """Column indices must match input_schema.json exactly."""
    schema = json.loads((ROOT / 'examples/input_schema.json').read_text())
    assert schema['feature_order'] == FEATURES
    assert len(FEATURES) == 39


def test_output_has_correct_feature_count():
    players, events = _standard_fixture()
    x, meta, _, _ = build_encounters(players, events)
    assert x.shape == (1, 256, 39)
    assert len(meta) == 1


# ── Shape and dtype ──────────────────────────────────────────────────

def test_output_shape_and_dtype():
    players, events = _standard_fixture()
    x, _, _, _ = build_encounters(players, events)
    assert x.ndim == 3
    assert x.shape[1:] == (256, 39)
    assert x.dtype == np.float32


def test_all_values_finite():
    players, events = _standard_fixture()
    x, _, _, _ = build_encounters(players, events)
    assert np.isfinite(x).all()


# ── Angle wrapping ───────────────────────────────────────────────────

def test_wrap_boundary_values():
    np.testing.assert_allclose(wrap(np.array([359, -359, 181, -181])),
                               [-1, 1, -179, 179])


def test_yaw_wrap_in_features():
    """A yaw change across the ±180 boundary must wrap correctly."""
    players, events = _standard_fixture()
    # Attacker yaw crosses ±180 boundary at tick 129.
    players['A']['yaw'][:128] = 179.
    players['A']['yaw'][128:] = -179.
    x, _, _, _ = build_encounters(players, events)
    # Index 128 is tick 129: 179→-179 wraps to +2.
    assert x[0, 128, 0] == pytest.approx(2.0, abs=1e-5)


# ── Chronological ordering ──────────────────────────────────────────

def test_ticks_are_chronological():
    players, events = _standard_fixture()
    _, meta, _, _ = build_encounters(players, events)
    ticks = np.array([m['tick'] for m in meta])
    assert np.all(np.diff(ticks) >= 0)


# ── Damage event alignment ──────────────────────────────────────────

def test_window_ends_strictly_before_damage():
    """The 256-tick window must end at tick = anchor - 1."""
    players, events = _standard_fixture()
    damage_tick = events[0]['tick']  # 257
    x, _, _, _ = build_encounters(players, events)
    # The window covers ticks [1..256], so tick 257 is excluded.
    # With yaw=0 everywhere, the shot feature (index 4) at the last
    # tick (index 255 = tick 256) should not include the damage tick.
    # Set a weapon_fire at tick 257 (the damage tick itself).
    players['A']['shot'][256] = 1.0  # tick 257 — outside window
    x2, _, _, _ = build_encounters(players, events)
    assert x2[0, :, 4].sum() == 0.0  # No shots inside the window


def test_no_future_information():
    """Modifying data after the anchor tick must not change the window."""
    players, events = _standard_fixture()
    x1, _, _, _ = build_encounters(players, events)
    # Corrupt everything at and after the anchor tick
    for arr in [players['A']['yaw'], players['A']['pitch'],
                players['A']['X'], players['B']['X']]:
        arr[256:] = 9999.  # tick 257+ (0-indexed 256+)
    x2, _, _, _ = build_encounters(players, events)
    np.testing.assert_array_equal(x1, x2)


# ── Player filtering ────────────────────────────────────────────────

def test_consent_filter():
    players, events = _standard_fixture()
    # Only score player 'A' (attacker) — should work
    x, meta, lmap, _ = build_encounters(players, events, consent_players={'A'})
    assert len(meta) == 1
    assert 'A' in lmap

    # Only consent player 'B' (victim only) — no encounters as attacker
    x2, meta2, lmap2, _ = build_encounters(players, events, consent_players={'B'})
    assert len(meta2) == 0


def test_player_local_ids():
    players, events = _standard_fixture()
    _, meta, lmap, _ = build_encounters(players, events)
    # Players get sequential local IDs
    assert lmap['A'] == 'Player_1'
    assert lmap['B'] == 'Player_2'
    # Encounter metadata uses local IDs
    assert meta[0]['player'] == 'Player_1'
    assert meta[0]['victim'] == 'Player_2'


# ── Missing telemetry ────────────────────────────────────────────────

def test_missing_ticks_rejected():
    """If the attacker is missing ticks in the 256-tick window, reject."""
    players, events = _standard_fixture()
    # Remove tick 100 from attacker (create a gap)
    mask = players['A']['tick'] != 100
    players['A'] = {k: v[mask] for k, v in players['A'].items()}
    x, meta, _, audit = build_encounters(players, events)
    assert len(meta) == 0
    assert audit['rejected'].get('missing_tick', 0) >= 1


def test_victim_missing_ticks_rejected():
    """If the victim is missing ticks, the encounter is rejected."""
    players, events = _standard_fixture()
    mask = players['B']['tick'] != 200
    players['B'] = {k: v[mask] for k, v in players['B'].items()}
    x, meta, _, audit = build_encounters(players, events)
    assert len(meta) == 0
    assert audit['rejected'].get('missing_tick', 0) >= 1


# ── Encounter rejection criteria ────────────────────────────────────

def test_warmup_excluded():
    players, events = _standard_fixture()
    players['A']['is_warmup_period'][:] = 1.0
    x, meta, _, audit = build_encounters(players, events)
    assert len(meta) == 0
    assert audit['rejected'].get('warmup', 0) >= 1


def test_team_damage_excluded():
    players, events = _standard_fixture()
    players['B']['team_num'][:] = 2.0  # Same team as attacker
    x, meta, _, audit = build_encounters(players, events)
    assert len(meta) == 0
    assert audit['rejected'].get('team_damage', 0) >= 1


def test_round_boundary_excluded():
    players, events = _standard_fixture()
    players['A']['total_rounds_played'][200:] = 2.0  # Round changes mid-window
    x, meta, _, audit = build_encounters(players, events)
    assert len(meta) == 0
    assert audit['rejected'].get('round_boundary', 0) >= 1


def test_not_alive_excluded():
    players, events = _standard_fixture()
    players['A']['is_alive'][50] = 0.0
    x, meta, _, audit = build_encounters(players, events)
    assert len(meta) == 0
    assert audit['rejected'].get('not_alive_throughout', 0) >= 1


def test_self_damage_excluded():
    """Attacker == victim should never produce an encounter."""
    players, events = _standard_fixture()
    events = [{'tick': 257, 'attacker': 'A', 'victim': 'A', 'weapon': 'ak47'}]
    x, meta, _, audit = build_encounters(players, events)
    assert len(meta) == 0


def test_nongun_excluded():
    """Knife, grenade etc. should not trigger encounters."""
    players, events = _standard_fixture()
    events = [{'tick': 257, 'attacker': 'A', 'victim': 'B', 'weapon': 'knife'}]
    x, meta, _, audit = build_encounters(players, events)
    assert len(meta) == 0


# ── NPZ compatibility ───────────────────────────────────────────────

def test_npz_structure(tmp_path):
    """Output NPZ must have keys x, player, tick with correct shapes."""
    players, events = _standard_fixture()
    x, meta, lmap, _ = build_encounters(players, events)
    player_arr = np.array([m['player'] for m in meta])
    tick_arr = np.array([m['tick'] for m in meta], dtype=np.int64)
    npz_path = tmp_path / 'test.npz'
    np.savez(npz_path, x=x, player=player_arr, tick=tick_arr)

    with np.load(npz_path, allow_pickle=False) as data:
        assert 'x' in data and 'player' in data and 'tick' in data
        assert data['x'].shape == (1, 256, 39)
        assert len(data['player']) == 1
        assert len(data['tick']) == 1


# ── Regression test with existing example ────────────────────────────

def test_example_npz_regression():
    """The included validation example must still produce expected scores."""
    example = ROOT / 'examples/example_validation_player.npz'
    expected_csv = ROOT / 'examples/example_expected_scores.csv'
    if not example.exists() or not expected_csv.exists():
        pytest.skip('Example files not present.')

    with np.load(example, allow_pickle=False) as data:
        x, players, ticks = data['x'], data['player'], data['tick']
    assert x.shape == (16, 256, 39)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()

    # Read expected scores
    import csv
    with expected_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    expected_raw = float(rows[0]['raw_score'])
    expected_cal = float(rows[0]['calibrated_score'])

    # Run inference if model artifacts are available
    model_dir = ROOT / 'experiments/cs2cd_v1'
    if not (model_dir / 'selection.json').exists():
        pytest.skip('Model artifacts not present.')

    from anticheat.cs2_inference import predict
    results = predict(x, players, ticks, str(model_dir), 'cpu')
    assert len(results) == 1
    np.testing.assert_allclose(results[0]['raw_score'], expected_raw, atol=1e-4)
    np.testing.assert_allclose(results[0]['calibrated_score'], expected_cal, atol=1e-4)


# ── Deterministic output ────────────────────────────────────────────

def test_deterministic_extraction():
    """Same input must produce identical output."""
    players, events = _standard_fixture()
    x1, meta1, _, _ = build_encounters(players, events)
    x2, meta2, _, _ = build_encounters(players, events)
    np.testing.assert_array_equal(x1, x2)
    assert meta1 == meta2


# ── Geometry checks ─────────────────────────────────────────────────

def test_target_error_geometry():
    """Verify target_error_pitch matches the standing-height approximation."""
    players, events = _standard_fixture()
    # Victim at X=-100, Z=100 (eye height = 100+64=164)
    # Attacker at X=0, Z=0 (eye height = 0+64=64)
    # dz = 164 - 64 = 100, horizontal = 100
    # elevation = -degrees(arctan2(100, 100)) = -45
    # pitch = 0 → target_error_pitch = -45 - 0 = -45
    x, _, _, _ = build_encounters(players, events)
    np.testing.assert_allclose(x[0, :, 3], -45, atol=1e-4)


def test_weapon_one_hot():
    """AK-47 must set weapon_auto=1 and all others=0."""
    players, events = _standard_fixture()
    x, _, _, _ = build_encounters(players, events)
    # weapon_auto (idx 34) should be 1, rest should be 0
    assert x[0, 0, 34] == 1.0  # weapon_auto
    assert x[0, 0, 35] == 0.0  # weapon_sniper
    assert x[0, 0, 36] == 0.0  # weapon_pistol
    assert x[0, 0, 37] == 0.0  # weapon_smg
    assert x[0, 0, 38] == 0.0  # weapon_shotgun


def test_burst_debounce():
    """Hits within 128 ticks of the same pair share one anchor."""
    players = {
        'A': _player_array(600, team_num=np.full(600, 2., dtype=np.float32)),
        'B': _player_array(600, team_num=np.full(600, 3., dtype=np.float32)),
    }
    players['B']['X'][:] = -100.
    events = [
        {'tick': 300, 'attacker': 'A', 'victim': 'B', 'weapon': 'ak47'},
        {'tick': 310, 'attacker': 'A', 'victim': 'B', 'weapon': 'ak47'},  # within 128
        {'tick': 500, 'attacker': 'A', 'victim': 'B', 'weapon': 'ak47'},  # new burst
    ]
    x, meta, _, audit = build_encounters(players, events)
    assert audit['anchor_count'] == 2  # 310 is debounced


@pytest.mark.parametrize('player', ['A', 'B'])
@pytest.mark.parametrize('duplicate_tick', [1, 100, 256, 257, 280])
def test_duplicate_ticks_are_rejected_only_inside_window(player, duplicate_tick):
    players, events = _standard_fixture()
    expected, expected_meta, _, _ = build_encounters(players, events)
    state = players[player]
    index = duplicate_tick - 1
    players[player] = {
        name: np.insert(values, index, values[index]) for name, values in state.items()
    }
    actual, meta, _, audit = build_encounters(players, events)
    if duplicate_tick < events[0]['tick']:
        assert len(actual) == 0
        assert audit['rejected'] == {'duplicate_tick': 1}
    else:
        np.testing.assert_array_equal(actual, expected)
        assert meta == expected_meta
        assert audit['rejected'] == {}


def test_victim_round_boundary_rejected():
    players, events = _standard_fixture()
    players['B']['total_rounds_played'][200:] = 2
    x, meta, _, audit = build_encounters(players, events)
    assert len(x) == len(meta) == 0
    assert audit['rejected'] == {'round_boundary': 1}


@pytest.mark.parametrize('player', ['A', 'B'])
@pytest.mark.parametrize('property_name', ['health', 'punch_0', 'is_warmup_period'])
def test_nonfinite_telemetry_is_window_local(player, property_name):
    players, events = _standard_fixture()
    expected, _, _, _ = build_encounters(players, events)
    players[player][property_name][256:] = np.nan
    actual, _, _, audit = build_encounters(players, events)
    np.testing.assert_array_equal(actual, expected)
    assert audit['rejected'] == {}
    players[player][property_name][100] = np.nan
    actual, _, _, audit = build_encounters(players, events)
    assert len(actual) == 0
    assert audit['rejected'] == {'nonfinite_telemetry': 1}


def test_missing_required_filter_rejected():
    players, events = _standard_fixture()
    del players['B']['total_rounds_played']
    x, _, _, audit = build_encounters(players, events)
    assert len(x) == 0
    assert audit['rejected'] == {'missing_telemetry': 1}


def test_empty_consent_and_stable_victim_aliases():
    players, events = _standard_fixture()
    players = dict(reversed(list(players.items())))
    _, meta, aliases, _ = build_encounters(players, events, {'A'})
    assert aliases == {'A': 'Player_1', 'B': 'Player_2'}
    assert meta[0]['victim'] == aliases['B']
    x, meta, empty_aliases, audit = build_encounters(players, events, set())
    assert x.shape == (0, 256, len(FEATURES))
    assert meta == []
    assert empty_aliases == aliases
    assert audit == {'anchor_count': 0, 'encounter_count': 0, 'rejected': {}, 'players': []}


@pytest.mark.parametrize('invalid_id', ['0', 'nan', 'None', '<NA>'])
def test_invalid_players_do_not_get_aliases_or_anchors(invalid_id):
    players, events = _standard_fixture()
    players[invalid_id] = _player_array(300)
    events.append({'tick': 257, 'attacker': invalid_id, 'victim': 'B', 'weapon': 'ak47'})
    x, _, aliases, audit = build_encounters(players, events)
    assert len(x) == 1
    assert invalid_id not in aliases
    assert audit['anchor_count'] == 1


@pytest.mark.parametrize('tick', [np.nan, np.inf, -np.inf, 'invalid', 257.5, None])
def test_build_rejects_invalid_event_ticks(tick):
    players, events = _standard_fixture()
    events[0]['tick'] = tick
    with pytest.raises(ValueError, match='tick'):
        build_encounters(players, events)


def test_unordered_events_and_future_hits_do_not_change_prior_windows():
    players = {'A': _player_array(600), 'B': _player_array(600, team_num=np.full(600, 3.))}
    events = [{'tick': tick, 'attacker': 'A', 'victim': 'B', 'weapon': 'ak47'}
              for tick in (300, 310, 438, 567)]
    expected, meta, _, audit = build_encounters(players, events)
    actual, reordered_meta, _, reordered_audit = build_encounters(players, events[::-1])
    np.testing.assert_array_equal(actual, expected)
    assert reordered_meta == meta
    assert reordered_audit == audit
    assert [row['tick'] for row in meta] == [300, 567]
    earlier, _, _, _ = build_encounters(players, events[:1])
    np.testing.assert_array_equal(expected[:1], earlier)


def test_feature_parity_with_frozen_parquet_extractor(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from anticheat.cs2_data import BASE_COLUMNS, extract_match, time_since

    monkeypatch.chdir(tmp_path)
    players, events = _standard_fixture()
    event_payload = {'player_hurt': [
        {'tick': event['tick'], 'attacker_steamid': event['attacker'],
         'user_steamid': event['victim'], 'weapon': event['weapon']} for event in events
    ], 'weapon_fire': [], 'player_footstep': [], 'player_blind': []}
    columns = {name: [] for name in BASE_COLUMNS}
    for player_index, (player_id, state) in enumerate(players.items()):
        ticks = state['tick']
        state['yaw'][:] = np.where(ticks <= 128, 179., -179.)
        state['pitch'][:] = ticks / 100
        for name in ('velocity_X', 'velocity_Y', 'velocity_Z', 'duck_amount',
                     'is_scoped', 'is_airborne', 'fl_recoil_idx', 'punch_0', 'punch_1',
                     'shots_fired'):
            state[name][:] = (ticks % 5) * (player_index + 1) / 10
        state['Y'][:] = ticks * (player_index + 1)
        shots, steps = [257, 128, 30], [260, 90]
        state['shot'] = np.isin(ticks, shots).astype(np.float32)
        state['footstep'] = np.isin(ticks, steps).astype(np.float32)
        state['since_shot'] = time_since(ticks, shots)
        state['since_noise'] = time_since(ticks, shots + steps)
        state['flash'][:] = np.where(ticks >= 100, np.maximum(0, 2 - (ticks - 100) / 64), 0)
        event_payload['weapon_fire'].extend(
            {'tick': tick, 'user_steamid': player_id, 'weapon': 'weapon_ak47'} for tick in shots)
        event_payload['player_footstep'].extend(
            {'tick': tick, 'user_steamid': player_id} for tick in steps)
        event_payload['player_blind'].append(
            {'tick': 100, 'user_steamid': player_id, 'blind_duration': 2.})
        for name in BASE_COLUMNS:
            if name == 'steamid':
                values = [player_id] * len(ticks)
            elif name == 'active_weapon_name':
                values = state['weapon'].tolist()
            elif name == 'aim_punch_angle':
                values = np.column_stack((state['punch_0'], state['punch_1'], np.zeros(len(ticks)))).tolist()
            else:
                values = state[name].tolist()
            columns[name].extend(values)
    path = tmp_path / 'raw' / 'cohort' / 'sample.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(columns), path)
    path.with_suffix('.json').write_text(json.dumps(event_payload), encoding='utf-8')
    baseline_audit = extract_match(str(path))
    actual, meta, _, audit = build_encounters(players, events)
    with np.load(tmp_path / 'data/cs2cd/processed/matches/cohort_sample.npz') as baseline:
        np.testing.assert_array_equal(actual, baseline['x'])
        assert [row['tick'] for row in meta] == baseline['tick'].tolist()
    assert audit['anchor_count'] == baseline_audit['anchor_count']
    assert audit['rejected'] == baseline_audit['rejected'] == {}


@pytest.mark.parametrize('empty_consent', [False, True])
def test_extract_npz_real_predict_and_reporting(tmp_path, monkeypatch, empty_consent):
    from anticheat import dem_extractor
    from anticheat.cs2_inference import predict
    from anticheat.private_scoring import add_feature_audit

    players, events = _standard_fixture()
    demo = tmp_path / 'synthetic.dem'
    demo.touch()  # Only a path placeholder: no real .dem parsing is exercised.
    monkeypatch.setattr(dem_extractor, 'parse_demo', lambda *args: (
        players, events, {'map_name': 'synthetic', 'verified_tick_rate': 64},
        ['Synthetic parser input; no real demo tested.'], []))
    audit = dem_extractor.extract_dem(
        demo, tmp_path / 'encounters.npz', set() if empty_consent else {'A'})
    saved_audit = json.loads(Path(audit['npz_path']).with_suffix('.json').read_text(encoding='utf-8'))
    assert saved_audit == audit
    with np.load(audit['npz_path'], allow_pickle=False) as payload:
        raw, player_ids, ticks = payload['x'], payload['player'], payload['tick']
    assert len(raw) == audit['encounter_count'] == len(audit['events'])
    assert player_ids.tolist() == [row['player'] for row in audit['events']]
    assert ticks.tolist() == [row['tick'] for row in audit['events']]
    assert audit['warnings'] and audit['known_approximations']
    assert audit['approximated_features'] == []
    assert audit['feature_stats']['delta_yaw']['count'] == raw.size // len(FEATURES)
    assert audit['feature_stats']['delta_yaw']['mean'] == (None if empty_consent else 0.)
    assert (audit['encounter_count'] == 0) == empty_consent
    # The existing reporting layer must accept and preserve the statistics schema.
    reporting_audit = dict(audit)
    add_feature_audit(reporting_audit, raw)
    assert reporting_audit['feature_stats'] == audit['feature_stats']
    model_dir = ROOT / 'experiments/cs2cd_v1'
    results = predict(raw, player_ids, ticks, model_dir, 'cpu')
    if empty_consent:
        assert results == []
    else:
        assert len(results) == 1
        assert results[0]['player'] == 'Player_1'
        assert results[0]['encounters'] == 1
        assert 0 <= results[0]['raw_score'] <= 1
        assert 0 <= results[0]['calibrated_score'] <= 1
        assert results == predict(raw, player_ids, ticks, model_dir, 'cpu')
