"""Freeze model/ensemble selection before calibration or test scoring."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import datetime
import numpy as np
import pandas as pd
from anticheat.data import ROOT,load_splits
from anticheat.inference import combine
from anticheat.metrics import ranking


def main():
    if (ROOT/"results/test_predictions.csv").exists():
        raise RuntimeError("Test already evaluated. Do not retune selection against an exposed holdout.")
    frame,splits=load_splits();ids=splits["validation"];y=frame.label.to_numpy()[ids]
    predictions={}
    failed={}
    for path in sorted((ROOT/"results/validation").glob("*.npz")):
        a=np.load(path,allow_pickle=False)
        np.testing.assert_array_equal(a["ids"],ids)
        metadata=json.loads((ROOT/f"models/{path.stem}.json").read_text())
        if metadata.get("train",{}).get("roc_auc",1.)<=.55:
            failed[path.stem]={"reason":"Training failed to learn above chance; retained for diagnosis, not a representative baseline.","metrics":metadata}
            continue
        predictions[path.stem]=a["p"]
    if not predictions:raise RuntimeError("No validation predictions")
    candidates=[]
    def add(name,components,mode="probability"):
        p=combine(predictions,components,mode)
        candidates.append({"name":name,"components":components,"mode":mode,**ranking(y,p)})
    for name in predictions:add(name,{name:1.})
    tree_names=[n for n in predictions if n.startswith(("lgbm_leaves","xgb_","catboost_"))]
    tree_names=sorted(tree_names,key=lambda n:ranking(y,predictions[n])["roc_auc"],reverse=True)
    for count in sorted(set([min(3,len(tree_names)),min(5,len(tree_names)),len(tree_names)])):
        if count:
            components={n:1/count for n in tree_names[:count]}
            for mode in ["probability","logit"]:add(f"top{count}_trees_{mode}",components,mode)
    deep=[n for n in predictions if n.startswith(("cnn_","hybrid_","lstm_"))]
    deep=sorted(deep,key=lambda n:ranking(y,predictions[n])["roc_auc"],reverse=True)
    if len(deep)>1:add("top2_neural",{n:.5 for n in deep[:2]})
    for kind in ["cnn","hybrid"]:
        family=[n for n in deep if n.startswith(kind+"_")]
        if len(family)>1:
            add(kind+"_all_seeds",{n:1/len(family) for n in family})
            add(kind+"_all_seeds_logit",{n:1/len(family) for n in family},"logit")
    base=max(candidates,key=lambda a:a["roc_auc"])
    if deep:
        deep_best=max([c for c in candidates if all(n in deep for n in c["components"])],key=lambda a:a["roc_auc"])
        if base["name"]!=deep_best["name"]:
            for weight in [.15,.3,.5]:
                parts={n:w*(1-weight) for n,w in base["components"].items()}
                for n,w in deep_best["components"].items():parts[n]=parts.get(n,0)+w*weight
                for mode in ["probability","logit"]:add(f"best_plus_neural_{weight}_{mode}",parts,mode)
    winner=max(candidates,key=lambda a:a["roc_auc"])
    hashes={}
    for name in winner["components"]:
        path=ROOT/f"models/{name}.joblib"
        if not path.exists():path=ROOT/f"models/{name}.pt"
        with path.open("rb") as handle:hashes[str(path.relative_to(ROOT))]=hashlib.file_digest(handle,"sha256").hexdigest()
    for rel in ["data/processed/splits.csv","data/processed/feature_names.json","reports/protocol.md",
                "anticheat/features.py","anticheat/neural.py","anticheat/inference.py","data/raw/download_manifest.json"]:
        with (ROOT/rel).open("rb") as handle:hashes[rel]=hashlib.file_digest(handle,"sha256").hexdigest()
    winner={**winner,"selected_on":"validation ROC-AUC only","frozen_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"sha256":hashes,
            "candidate_count":len(candidates),"validation_records":len(ids),"evaluated_candidates":sorted(predictions)}
    (ROOT/"models/selection.json").write_text(json.dumps(winner,indent=2))
    (ROOT/"results/validation_candidates.json").write_text(json.dumps(candidates,indent=2))
    (ROOT/"results/failed_training_runs.json").write_text(json.dumps(failed,indent=2))
    pd.DataFrame([{k:v for k,v in c.items() if k!="components"} for c in candidates]).sort_values("roc_auc",ascending=False).to_csv(ROOT/"results/validation_leaderboard.csv",index=False)
    print(json.dumps(winner,indent=2))


if __name__=="__main__":main()
