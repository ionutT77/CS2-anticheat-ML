"""Load saved candidates and reproduce record-level scores on new arrays."""
import sys
import json
from pathlib import Path
import numpy as np
import joblib
import torch
from anticheat.data import ROOT
from anticheat.features import engagement_features,aggregate_features,player_features
from anticheat.neural import PlayerModel,sequence_channels

if (ROOT/".vendor").exists():sys.path.insert(0,str(ROOT/".vendor"))


def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-40,40)))


def combine(predictions,components,mode="probability"):
    values=[predictions[n] if mode=="probability" else logit(predictions[n]) for n in components]
    p=sum(components[n]*v for n,v in zip(components,values))
    return p if mode=="probability" else sigmoid(p)


def predict_component(name,raw,features=None,event_features=None,device="cpu",batch_size=24):
    tree_path=ROOT/f"models/{name}.joblib"
    if tree_path.exists():
        bundle=joblib.load(tree_path)
        if bundle.get("level")=="engagement":
            if event_features is None:
                ee,_=engagement_features(raw.reshape(-1,192,5));event_features=ee.reshape(len(raw),30,-1)
            p=bundle["model"].predict_proba(event_features.reshape(-1,event_features.shape[-1]))[:,1].reshape(len(raw),30)
            pool=bundle["pool"]
            if pool=="mean":return p.mean(1)
            if pool=="top10":return np.sort(p,axis=1)[:,-10:].mean(1)
            return sigmoid(logit(p).mean(1))
        if features is None:features,_=player_features(raw)
        return bundle["model"].predict_proba(features[:,bundle["columns"]])[:,1]
    ckpt=torch.load(ROOT/f"models/{name}.pt",map_location="cpu",weights_only=True)
    cfg=ckpt["config"]
    channels=5 if cfg["kind"]=="lstm" or cfg.get("raw_five",False) else 12
    model=PlayerModel(cfg["kind"],ckpt["feature_dim"],cfg["width"],channels).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    ff=None
    if ckpt["feature_dim"]:
        if features is None:features,_=player_features(raw)
        ff=np.clip((np.arcsinh(features)-ckpt["feature_mean"].numpy())/ckpt["feature_std"].numpy(),-8,8).astype(np.float32)
    ps=[]
    with torch.inference_mode():
        for lo in range(0,len(raw),batch_size):
            a=torch.from_numpy(np.asarray(raw[lo:lo+batch_size])).to(device)
            # Match float16 storage used during training before float32 autocast input.
            xx=sequence_channels(a,enriched=channels==12).half().float()
            f=None if ff is None else torch.from_numpy(ff[lo:lo+batch_size]).to(device)
            if str(device).startswith("cuda"):
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not cfg.get("fp32",False)):z=model(xx,f)
            else:z=model(xx,f)
            ps.append(z.float().sigmoid().cpu().numpy())
    return np.concatenate(ps)


def predict(raw,device="cpu"):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim==3:raw=raw[None]
    if raw.ndim!=4 or raw.shape[1:]!=(30,192,5) or not np.isfinite(raw).all():
        raise ValueError("Expected finite float32 array [N,30,192,5]")
    if not np.isin(raw[...,4],[0,1]).all():raise ValueError("Firing must be binary (0 or 1)")
    selection=json.loads((ROOT/"models/selection.json").read_text())
    calibration=json.loads((ROOT/"models/calibration.json").read_text())
    e,en=engagement_features(raw.reshape(-1,192,5));e=e.reshape(len(raw),30,-1)
    f,_=aggregate_features(e,en)
    parts={name:predict_component(name,raw,f,e,device=device) for name in selection["components"]}
    score=combine(parts,selection["components"],selection["mode"])
    probability=sigmoid(calibration["slope"]*logit(score)+calibration["intercept"])
    return {"cheater_probability":probability,"review_flag":probability>=calibration["thresholds"]["f1"],
            "low_fpr_review_flag":probability>=calibration["thresholds"]["fpr_1pct"],"component_scores":parts}
