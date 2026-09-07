"""Calibrate frozen models, then evaluate the untouched test reserve once."""
import os
os.environ.setdefault("MPLCONFIGDIR","/tmp/csgo-mpl")
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve,accuracy_score,roc_auc_score,average_precision_score
from anticheat.data import ROOT,load_arrays,load_splits
from anticheat.inference import predict_component,combine,logit,sigmoid
from anticheat.metrics import evaluate,ranking


def calibrate(y,p):
    model=LogisticRegression(C=1e6,max_iter=1000).fit(logit(p)[:,None],y)
    slope=float(model.coef_[0,0]);intercept=float(model.intercept_[0])
    if slope<=0:raise RuntimeError("Non-monotone calibration; investigate training before evaluation")
    return slope,intercept,sigmoid(slope*logit(p)+intercept)


def thresholds(y,p):
    precision,recall,t=precision_recall_curve(y,p)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    # Resolve equal F1 in favor of the higher threshold.
    best=np.flatnonzero(f1==f1.max())[-1]
    out={"f1":float(t[best]),"default_0_5":.5}
    candidates=np.r_[0.,np.unique(p),np.nextafter(p.max(),np.inf)]
    accuracy=np.array([accuracy_score(y,p>=z) for z in candidates])
    out["accuracy"]=float(candidates[np.flatnonzero(accuracy==accuracy.max())[-1]])
    negatives=np.sort(p[y==0])[::-1]
    for name,alpha in [("fpr_1pct",.01),("fpr_0_1pct",.001)]:
        permitted=int(np.floor(alpha*len(negatives)))
        out[name]=float(np.nextafter(negatives[permitted],np.inf))
    return out


def bootstrap(y,p,groups,threshold,iterations=2000,references=None):
    references={} if references is None else references
    rng=np.random.default_rng(20260907)
    unique=np.unique(groups);members=[np.where(groups==g)[0] for g in unique]
    samples=[]
    for _ in range(iterations):
        ix=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))])
        yy=y[ix];pp=p[ix];pred=pp>=threshold
        tp=np.sum(pred&(yy==1));fp=np.sum(pred&(yy==0));fn=np.sum((~pred)&(yy==1))
        auc=roc_auc_score(yy,pp)
        samples.append([auc,average_precision_score(yy,pp),np.mean(pred==yy),
                        tp/max(tp+fp,1),tp/max(tp+fn,1),2*tp/max(2*tp+fp+fn,1),fp/max(np.sum(yy==0),1)]
                       +[auc-roc_auc_score(yy,ref[ix]) for ref in references.values()])
    ci=np.quantile(samples,[.025,.975],axis=0).T
    return {k:[float(a),float(b)] for k,(a,b) in zip(["roc_auc","average_precision","accuracy","precision","recall","f1","fpr"]+list(references),ci)}


def main():
    out=ROOT/"results"
    if (out/"test_predictions.csv").exists():raise RuntimeError("Test already evaluated; use the saved results instead of tuning on it")
    selection=json.loads((ROOT/"models/selection.json").read_text())
    for rel,expected in selection["sha256"].items():
        with (ROOT/rel).open("rb") as handle:actual=hashlib.file_digest(handle,"sha256").hexdigest()
        if expected!=actual:raise RuntimeError(f"Frozen artifact changed: {rel}")
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    raw,y=load_arrays();frame,splits=load_splits()
    cal,te=splits["calibration"],splits["test"]
    x=np.load(ROOT/"data/processed/player_features.npy",mmap_mode="r",allow_pickle=False)
    e=np.load(ROOT/"data/processed/engagement_features.npy",mmap_mode="r",allow_pickle=False)
    names=selection["evaluated_candidates"]
    cp,tp={},{}
    indices=np.r_[cal,te]
    rr=raw[indices];ff=x[indices];ee=e[indices]
    for name in names:
        start=time.monotonic()
        p=predict_component(name,rr,ff,ee,device=device)
        cp[name]=p[:len(cal)];tp[name]=p[len(cal):]
        # No test metric is printed or used for any model selection decision.
        print("Scored frozen candidate",name,"seconds",round(time.monotonic()-start,2),flush=True)
    cp["selected"]=combine(cp,selection["components"],selection["mode"])
    tp["selected"]=combine(tp,selection["components"],selection["mode"])
    models={};predictions=frame.loc[te,["record_id","group","label"]].copy()
    calibration_records={}
    for name in [*names,"selected"]:
        slope,intercept,pc=calibrate(y[cal],cp[name])
        pt=sigmoid(slope*logit(tp[name])+intercept)
        th=thresholds(y[cal],pc)
        calibration_records[name]={"slope":slope,"intercept":intercept,"thresholds":th,
            "calibration_records":len(cal),"calibration_negatives":int((y[cal]==0).sum()),
            "calibration_operating_points":{k:evaluate(y[cal],pc,v) for k,v in th.items()}}
        models[name]={"ranking":ranking(y[te],pt),"operating_points":{k:evaluate(y[te],pt,v) for k,v in th.items()}}
        predictions[name]=pt
    selected_p=predictions.selected.to_numpy()
    selected_calibration=calibration_records["selected"]
    (ROOT/"models/calibration.json").write_text(json.dumps(selected_calibration,indent=2))
    (out/"calibration_all_models.json").write_text(json.dumps(calibration_records,indent=2))
    predictions.to_csv(out/"test_predictions.csv",index=False)
    pd.DataFrame({"record_id":cal,"label":y[cal],"score":cp["selected"],
                  "probability":sigmoid(selected_calibration["slope"]*logit(cp["selected"])+selected_calibration["intercept"])}).to_csv(out/"calibration_predictions.csv",index=False)
    print("Computing group bootstrap intervals",flush=True)
    reference_names={}
    for family,prefixes in [("lstm",("lstm_",)),("tree",("lgbm_","xgb_","catboost_","extratrees"))]:
        members=[n for n in names if n.startswith(prefixes)]
        if members:
            reference_names[family]=max(members,key=lambda n:json.loads((ROOT/f"models/{n}.json").read_text())["validation"]["roc_auc"])
    ci=bootstrap(y[te],selected_p,frame.group.to_numpy()[te],selected_calibration["thresholds"]["f1"],
                 references={f"roc_auc_gain_vs_{family}":predictions[n].to_numpy() for family,n in reference_names.items()})
    grouped=predictions.groupby("group").agg(label=("label","first"),score=("selected","mean"))
    majority=evaluate(y[te],np.full(len(te),float(y[splits["train"]].mean())),.5)
    result={"selected_model":selection,"test_records":len(te),"test_cheaters":int(y[te].sum()),
            "test_legit":int((y[te]==0).sum()),"test_groups":frame.loc[te,"group"].nunique(),
            "models":models,"selected_group_bootstrap_95pct":ci,"bootstrap_iterations":2000,
            "paired_comparison_references_selected_on_validation":reference_names,
            "one_vote_per_duplicate_group":evaluate(grouped.label.to_numpy(),grouped.score.to_numpy(),selected_calibration["thresholds"]["f1"]),
            "majority_baseline":majority,"evaluation_device":device}
    (out/"test_metrics.json").write_text(json.dumps(result,indent=2))
    rows=[]
    for name,m in models.items():
        point=m["operating_points"]["f1"]
        rows.append({"model":name,**m["ranking"],**{k:point[k] for k in ["accuracy","balanced_accuracy","precision","recall","f1"]},
                     "recall_at_calibrated_1pct_threshold":m["operating_points"]["fpr_1pct"]["recall"],
                     "actual_fpr_at_calibrated_1pct_threshold":m["operating_points"]["fpr_1pct"]["fpr"]})
    pd.DataFrame(rows).to_csv(out/"test_leaderboard.csv",index=False)
    print(json.dumps({"selected":models["selected"],"confidence_intervals":ci,"group_level":result["one_vote_per_duplicate_group"]},indent=2),flush=True)


if __name__=="__main__":main()
