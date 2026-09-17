"""Uncalibrated neural encounter evidence for offline, consented review only."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch

from .cs2_data import FEATURES, transform
from .cs2_models import MatchDetector
from .cs2_training import predict_players

WINDOW_TICKS = 256
ENCOUNTER_FIELDS = (
    'encounter_index', 'player', 'tick', 'victim', 'weapon',
    'component', 'event_logit', 'event_score',
)
EVENT_SCORE_DESCRIPTION = (
    'Event scores are sigmoid(neural component event-head logits) only; '
    'they are NOT ensemble probabilities, calibrated scores, cheating verdicts, or bans.'
)
KNOWN_APPROXIMATIONS = (
    'Target eye/head geometry uses origin and duck amount, not hitboxes or line of sight.',
    'Encounter anchors are first damage in an attacker-victim burst; histories end before impact.',
    'Window-start finite differences have no preceding sample and start at zero.',
)


def _raw_features(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (WINDOW_TICKS, len(FEATURES)):
        raise ValueError(f'Expected [encounters,{WINDOW_TICKS},{len(FEATURES)}] raw features.')
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite feature values are not accepted.')
    return raw


def add_feature_audit(audit, raw):
    """Record raw, pre-transform statistics over all extracted encounter windows."""
    raw = _raw_features(raw)
    statistics = {}
    for index, feature in enumerate(FEATURES):
        values = raw[..., index].astype(np.float64).reshape(-1)
        statistics[feature] = {
            'count': int(values.size),
            'min': float(values.min()) if values.size else None,
            'max': float(values.max()) if values.size else None,
            'mean': float(values.mean()) if values.size else None,
            'std': float(values.std()) if values.size else None,
            'zero_fraction': float(np.mean(values == 0)) if values.size else None,
        }
    audit['feature_stats'] = statistics
    audit['feature_stats_scope'] = (
        'Raw pre-transform features across all extracted windows, before --player filtering; '
        'overlapping ticks are counted once per window.'
    )
    approximations = list(audit.get('known_approximations', []))
    approximations.extend(KNOWN_APPROXIMATIONS)
    approximations.extend(audit.get('approximated_features', []))
    audit['known_approximations'] = list(dict.fromkeys(approximations))
    audit['event_score_description'] = EVENT_SCORE_DESCRIPTION


def _verified_bytes(path, expected):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f'Artifact checksum mismatch: {path.name}')
    return content


def _selected_neural_artifacts(model_dir):
    """Verify the manifest chain and snapshot the exact bytes to deserialize."""
    root = Path(model_dir)
    calibration = json.loads((root / 'calibration.json').read_text(encoding='utf-8'))
    selection_bytes = _verified_bytes(root / 'selection.json', calibration['selection_sha256'])
    selection = json.loads(selection_bytes)
    checksums = selection['artifact_sha256']
    normalization = json.loads(_verified_bytes(
        root / 'normalization.json', checksums['normalization.json']))
    artifacts = []
    for name in selection['components']:
        # Use the manifest, not file existence: a missing selected .pt must fail.
        filename = f'{name}.pt'
        if filename not in checksums:
            filename = f'{name}.joblib'
        if Path(filename).name != filename or filename not in checksums:
            raise ValueError(f'Missing or invalid selected artifact: {name}')
        content = _verified_bytes(root / filename, checksums[filename])
        if filename.endswith('.pt'):
            artifacts.append((name, content))
    return normalization, artifacts


def predict_encounters(raw, player_ids, ticks, events, model_dir, device='cpu'):
    """Return one row per encounter per selected neural component, in input order.

    Component weights and player calibration deliberately do not apply here.
    Metadata must align exactly with NPZ rows; duplicate player/tick pairs are valid.
    """
    raw = _raw_features(raw)
    ids = np.asarray(player_ids).astype(str)
    ticks = np.asarray(ticks)
    if ids.shape != (len(raw),) or ticks.shape != (len(raw),):
        raise ValueError('Metadata lengths must match encounter count.')
    if len(events) != len(raw):
        raise ValueError("audit['events'] must contain one metadata row per encounter.")
    for index, event in enumerate(events):
        if (not {'player', 'tick', 'victim', 'weapon'} <= event.keys()
                or str(event['player']) != ids[index] or event['tick'] != ticks[index]):
            raise ValueError(f"audit['events'] metadata mismatch at encounter {index}.")

    normalization, artifacts = _selected_neural_artifacts(model_dir)
    mean = torch.tensor(normalization['mean'], device=device)
    std = torch.tensor(normalization['std'], device=device)
    if (mean.shape != (len(FEATURES),) or std.shape != mean.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not (std > 0).all()):
        raise ValueError('Normalization requires finite means and positive standard deviations for every feature.')
    if not len(raw) or not artifacts:
        return []

    players = sorted(set(ids))
    bags = [np.flatnonzero(ids == player)[np.argsort(ticks[ids == player], kind='stable')]
            for player in players]
    x = torch.from_numpy(transform(raw).astype(np.float16)).to(device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for component, checkpoint_bytes in artifacts:
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
        config = checkpoint['config']
        model = MatchDetector(config['channels'], config['architecture'],
                              config.get('hierarchical', False)).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        _, event_logits, indices = predict_players(
            model, x, mean, std, bags, np.arange(len(bags)))
        indices = np.asarray(indices)
        event_logits = np.asarray(event_logits)
        if (indices.shape != (len(raw),) or indices.dtype.kind not in 'iu'
                or not np.array_equal(np.sort(indices), np.arange(len(raw)))
                or event_logits.shape != (len(raw),) or not np.isfinite(event_logits).all()):
            raise ValueError('Neural inference returned invalid event logits or encounter indices.')
        ordered_logits = np.empty(len(raw), dtype=np.float64)
        ordered_logits[indices] = event_logits
        for index, (logit, score) in enumerate(zip(ordered_logits, expit(ordered_logits))):
            event = events[index]
            rows.append({
                'encounter_index': index, 'player': str(ids[index]),
                'tick': int(event['tick']), 'victim': str(event['victim']),
                'weapon': str(event['weapon']), 'component': component,
                'event_logit': float(logit), 'event_score': float(score),
            })
    return rows
