"""Inference on extracted, chronological pre-impact CS2 encounter arrays."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
import torch

from .cs2_data import FEATURES, transform
from .cs2_features import summarize_events, summarize_player
from .cs2_models import MatchDetector
from .cs2_training import predict_players
from .reference_lstm import apply_calibration


def predict(raw, player_ids, ticks, model_dir, device='cpu'):
    root=Path(model_dir)
    selection=json.loads((root/'selection.json').read_text())
    calibration=json.loads((root/'calibration.json').read_text())
    if hashlib.sha256((root/'selection.json').read_bytes()).hexdigest()!=calibration['selection_sha256']:
        raise ValueError('Calibration does not belong to this model selection.')
    if hashlib.sha256((root/'normalization.json').read_bytes()).hexdigest()!=selection['artifact_sha256']['normalization.json']:
        raise ValueError('Normalization checksum mismatch.')
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=3 or raw.shape[1:]!=(256,len(FEATURES)):
        raise ValueError(f'Expected [encounters,256,{len(FEATURES)}] raw feature array in the documented order.')
    if not np.isfinite(raw).all():raise ValueError('Nonfinite feature values are not accepted.')
    if len(raw)==0:return []
    ids=np.asarray(player_ids).astype(str);ticks=np.asarray(ticks)
    if len(ids)!=len(raw) or len(ticks)!=len(raw):raise ValueError('Metadata lengths must match encounter count.')
    unique=sorted(set(ids))
    bags=[np.flatnonzero(ids==player)[np.argsort(ticks[ids==player],kind='stable')] for player in unique]
    transformed=transform(raw)
    event_features=summarize_events(transformed)
    tabular=np.stack([summarize_player(event_features[bag]) for bag in bags])
    x=torch.from_numpy(transformed.astype(np.float16)).to(device)
    norm=json.loads((root/'normalization.json').read_text())
    mean=torch.tensor(norm['mean'],device=device);std=torch.tensor(norm['std'],device=device)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    scores=np.zeros(len(unique),dtype=np.float64)
    for name,weight in selection['components'].items():
        filename=name+('.pt' if (root/f'{name}.pt').exists() else '.joblib')
        path=root/filename
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=selection['artifact_sha256'][filename]:raise ValueError(f'Model checksum mismatch: {name}')
        if path.suffix=='.pt':
            checkpoint=torch.load(path,map_location='cpu',weights_only=True);config=checkpoint['config']
            model=MatchDetector(config['channels'],config['architecture'],config.get('hierarchical',False)).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            logits,_,_=predict_players(model,x,mean,std,bags,np.arange(len(bags)))
            p=expit(logits)
        else:
            saved=joblib.load(path)
            p=saved['model'].predict_proba(tabular[:,saved['feature_indices']])[:,1]
        scores+=weight*p
    cal=calibration['models']['selected'];calp=apply_calibration(scores,cal['platt'])
    return [{'player':player,'encounters':len(bag),'raw_score':float(score),'calibrated_score':float(probability),
             **{'flag_'+name:bool(probability>=threshold) for name,threshold in cal['thresholds'].items()}}
            for player,bag,score,probability in zip(unique,bags,scores,calp)]
