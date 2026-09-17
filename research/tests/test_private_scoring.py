"""Focused private-match reporting tests; no demo parser or external downloads."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.special import expit
import torch

from anticheat import private_scoring
from anticheat.cs2_data import FEATURES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'scripts' / 'score_private_match.py'
spec = importlib.util.spec_from_file_location('score_private_match', SCRIPT_PATH)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def _write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / 'models'
    root.mkdir()
    buffer = io.BytesIO()
    torch.save({'config': {'channels': 39, 'architecture': 'tcn'}, 'state_dict': {}}, buffer)
    (root / 'neural.pt').write_bytes(buffer.getvalue())
    (root / 'tree.joblib').write_bytes(b'not deserialized for event scoring')
    _write_json(root / 'normalization.json', {'mean': [0.] * 39, 'std': [1.] * 39})
    selection = {
        'components': {'neural': 0.01, 'tree': 0.99},
        'artifact_sha256': {name: _digest(root / name)
                           for name in ('neural.pt', 'tree.joblib', 'normalization.json')},
    }
    _write_json(root / 'selection.json', selection)
    _write_json(root / 'calibration.json', {'selection_sha256': _digest(root / 'selection.json')})
    return root


@pytest.fixture
def encounters():
    raw = np.zeros((3, 256, len(FEATURES)), dtype=np.float32)
    raw[:, :, 0] = np.array([1., 2., 3.])[:, None]
    players = np.array(['Player_2', 'Player_1', 'Player_1'])
    ticks = np.array([600, 500, 300])
    events = [{'player': str(player), 'tick': int(tick), 'victim': f'Victim_{index}',
               'weapon': 'é_weapon'} for index, (player, tick) in enumerate(zip(players, ticks))]
    return raw, players, ticks, events


@pytest.fixture
def neural(monkeypatch):
    model = Mock()
    model.to.return_value = model
    factory = Mock(return_value=model)
    monkeypatch.setattr(private_scoring, 'MatchDetector', factory)

    def fake_predict(model, x, mean, std, bags, player_indices):
        indices = np.concatenate([bags[index] for index in player_indices])
        # Deliberately unrelated player logits catch accidental player/event substitution.
        logits = x[indices, 0, 0].float().cpu().numpy()
        return np.full(len(bags), 99.), logits, indices

    inference = Mock(side_effect=fake_predict)
    monkeypatch.setattr(private_scoring, 'predict_players', inference)
    return factory, inference


def _invoke(monkeypatch, tmp_path, encounters, artifacts, extra=()):
    raw, players, ticks, events = encounters

    def fake_extract(demo, output_path, consent_players, prop_map):
        np.savez(output_path, x=raw, player=players, tick=ticks)
        audit = {
            'dem_path': str(demo), 'npz_path': str(output_path),
            'encounter_count': len(raw), 'anchor_count': len(raw), 'events': events,
            'players': [{'local_id': player, 'encounters': int(sum(players == player)),
                         'total_ticks': 900} for player in ('Player_1', 'Player_2', 'Player_3')],
            'header': {'map_name': 'de_été'}, 'warnings': [],
            'approximated_features': ['flash_remaining uses duration approximation'],
        }
        _write_json(output_path.with_suffix('.json'), audit)
        return audit

    extraction = Mock(side_effect=fake_extract)
    monkeypatch.setattr(cli, 'extract_dem', extraction)
    prediction = Mock(return_value=[{
        'player': 'Player_1', 'encounters': 2, 'raw_score': .75,
        'calibrated_score': .8, 'flag_review': False,
    }])
    monkeypatch.setattr(cli, 'predict', prediction)
    output = tmp_path / 'résultats'
    argv = ['score_private_match.py', str(tmp_path / 'fake.dem'), '--output', str(output),
            '--model-dir', str(artifacts), *extra]
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    return output, extraction, prediction


def test_mock_demo_end_to_end(monkeypatch, tmp_path, artifacts, encounters, neural):
    output, extraction, prediction = _invoke(
        monkeypatch, tmp_path, encounters, artifacts,
        ['--player', 'Player_1', '--consent-ids', '123', '456'])
    assert extraction.call_args.kwargs['consent_players'] == {'123', '456'}
    assert prediction.call_args.args[1].tolist() == ['Player_1', 'Player_1']
    with (output / 'encounter_scores.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['encounter_index']) for row in rows] == [1, 2]
    assert [int(row['tick']) for row in rows] == [500, 300]
    assert [row['victim'] for row in rows] == ['Victim_1', 'Victim_2']
    assert all(row['weapon'] == 'é_weapon' for row in rows)
    expected = np.arcsinh([2., 3.]).astype(np.float16).astype(float)
    np.testing.assert_allclose([float(row['event_logit']) for row in rows], expected)
    np.testing.assert_allclose([float(row['event_score']) for row in rows], expit(expected))
    audit = json.loads((output / 'extraction_audit.json').read_text(encoding='utf-8'))
    assert audit['feature_stats']['delta_yaw']['count'] == 3 * 256
    assert audit['feature_stats']['delta_yaw']['mean'] == 2.
    assert audit['known_approximations']
    report = (output / 'report.txt').read_text(encoding='utf-8')
    for text in ('de_été', 'Victim_1', 'é_weapon', 'neural', '500',
                 'NOT ensemble probabilities', 'Known Approximations', 'Feature Statistics'):
        assert text in report
    assert '0.800000' in report


def test_event_logits_use_returned_indices_not_player_scores(artifacts, encounters, neural):
    raw, players, ticks, events = encounters
    rows = private_scoring.predict_encounters(raw, players, ticks, events, artifacts)
    expected = np.arcsinh([1., 2., 3.]).astype(np.float16).astype(float)
    assert [row['encounter_index'] for row in rows] == [0, 1, 2]
    np.testing.assert_allclose([row['event_logit'] for row in rows], expected)
    np.testing.assert_allclose([row['event_score'] for row in rows], expit(expected))
    assert all(row['component'] == 'neural' for row in rows)
    assert all('calibrated_score' not in row and 'flag_review' not in row for row in rows)
    bags = neural[1].call_args.args[4]
    assert [bag.tolist() for bag in bags] == [[2, 1], [0]]


@pytest.mark.parametrize('filename', ['selection.json', 'normalization.json', 'neural.pt', 'tree.joblib'])
def test_checksum_tampering_rejected_before_model_load(artifacts, encounters, neural, filename):
    path = artifacts / filename
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='checksum mismatch'):
        private_scoring.predict_encounters(*encounters, artifacts)
    neural[0].assert_not_called()


def test_missing_selected_neural_cannot_fall_back_to_tree(artifacts, encounters, neural):
    (artifacts / 'neural.pt').unlink()
    (artifacts / 'neural.joblib').write_bytes(b'fallback must not be used')
    with pytest.raises(FileNotFoundError):
        private_scoring.predict_encounters(*encounters, artifacts)
    neural[0].assert_not_called()


@pytest.mark.parametrize('invalid_metadata', ['missing', 'reordered'])
def test_metadata_must_align(artifacts, encounters, neural, invalid_metadata):
    raw, players, ticks, events = encounters
    events = [] if invalid_metadata == 'missing' else events[::-1]
    with pytest.raises(ValueError, match='metadata row|metadata mismatch'):
        private_scoring.predict_encounters(raw, players, ticks, events, artifacts)
    neural[0].assert_not_called()


def test_nonfinite_features_rejected(artifacts, encounters, neural):
    encounters[0][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        private_scoring.predict_encounters(*encounters, artifacts)


def test_empty_consent_rejected_before_extraction(monkeypatch, tmp_path, artifacts, encounters):
    with pytest.raises(SystemExit) as error:
        _invoke(monkeypatch, tmp_path, encounters, artifacts, ['--consent-ids'])
    assert error.value.code == 2
    cli.extract_dem.assert_not_called()


@pytest.mark.parametrize('player, message', [
    ('unknown', 'Unknown player'), ('Player_3', 'no eligible encounters'),
])
def test_player_filter_errors(monkeypatch, tmp_path, artifacts, encounters, capsys, player, message):
    with pytest.raises(SystemExit) as error:
        _invoke(monkeypatch, tmp_path, encounters, artifacts, ['--player', player])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    cli.predict.assert_not_called()


def test_empty_match_reports_without_loading_model(monkeypatch, tmp_path, artifacts, encounters):
    raw, players, ticks, _ = encounters
    output, _, prediction = _invoke(
        monkeypatch, tmp_path, (raw[:0], players[:0], ticks[:0], []), artifacts)
    prediction.assert_not_called()
    with (output / 'encounter_scores.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(private_scoring.ENCOUNTER_FIELDS)
        assert list(reader) == []
    audit = json.loads((output / 'extraction_audit.json').read_text(encoding='utf-8'))
    assert audit['feature_stats']['delta_yaw']['mean'] is None
    assert 'No players could be scored' in (output / 'report.txt').read_text(encoding='utf-8')


def test_default_model_directory_is_cwd_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.DEFAULT_MODEL_DIR == ROOT / 'experiments' / 'cs2cd_v1'
    assert cli.DEFAULT_MODEL_DIR.is_absolute()


def test_each_selected_neural_component_has_separate_rows(artifacts, encounters, neural):
    (artifacts / 'second.pt').write_bytes((artifacts / 'neural.pt').read_bytes())
    selection = json.loads((artifacts / 'selection.json').read_text(encoding='utf-8'))
    selection['components']['second'] = .2
    selection['artifact_sha256']['second.pt'] = _digest(artifacts / 'second.pt')
    _write_json(artifacts / 'selection.json', selection)
    _write_json(artifacts / 'calibration.json', {'selection_sha256': _digest(artifacts / 'selection.json')})
    rows = private_scoring.predict_encounters(*encounters, artifacts)
    assert [row['component'] for row in rows] == ['neural'] * 3 + ['second'] * 3
    assert [row['encounter_index'] for row in rows] == [0, 1, 2] * 2
    assert neural[1].call_count == 2


def test_deserializes_verified_snapshot_not_reopened_path(monkeypatch, artifacts, encounters, neural):
    original_load = torch.load

    def load_snapshot(source, **kwargs):
        assert isinstance(source, io.BytesIO)
        assert kwargs['weights_only'] is True
        (artifacts / 'neural.pt').write_bytes(b'changed after verification')
        return original_load(source, **kwargs)

    monkeypatch.setattr(torch, 'load', load_snapshot)
    assert len(private_scoring.predict_encounters(*encounters, artifacts)) == 3


def test_real_selected_model_regression(monkeypatch, encounters):
    from anticheat.cs2_inference import predict

    model_dir = ROOT / 'experiments' / 'cs2cd_v1'
    if not (model_dir / 'selection.json').exists():
        pytest.skip('Real selected model artifacts not installed.')
    raw, players, ticks, events = encounters
    torch.set_num_threads(2)
    before = predict(raw, players, ticks, model_dir)
    inference = Mock(wraps=private_scoring.predict_players)
    monkeypatch.setattr(private_scoring, 'predict_players', inference)
    rows = private_scoring.predict_encounters(raw, players, ticks, events, model_dir)
    assert len(rows) == 3
    assert {row['component'] for row in rows} == {'tcn39_seed123'}
    assert all(np.isfinite(row['event_logit']) and 0 <= row['event_score'] <= 1 for row in rows)
    assert inference.call_count == 1
    # Inference makes no changes to inputs or the existing player-level prediction.
    assert predict(raw, players, ticks, model_dir) == before
