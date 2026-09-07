"""Additional tree families and weak-label engagement-to-player aggregation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import joblib
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from anticheat.data import ROOT,load_splits
from anticheat.metrics import ranking


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--only",default="all");args=ap.parse_args()
    frame,splits=load_splits();y=frame.label.to_numpy()
    tr,va=splits["train"],splits["validation"]
    x=np.load(ROOT/"data/processed/player_features.npy",allow_pickle=False)
    names=json.loads((ROOT/"data/processed/feature_names.json").read_text())
    configs=[("xgb_depth3",{"max_depth":3,"min_child_weight":10}),
             ("xgb_depth5",{"max_depth":5,"min_child_weight":20}),
             ("extratrees",{}),("engagement_lgbm",{})]
    for name,cfg in configs:
        if args.only!="all" and args.only!=name:continue
        if (ROOT/f"models/{name}.joblib").exists():continue
        start=time.monotonic()
        if name.startswith("xgb"):
            model=XGBClassifier(n_estimators=1500,learning_rate=.035,tree_method="hist",n_jobs=10,
                subsample=.85,colsample_bytree=.8,reg_lambda=15,reg_alpha=.1,
                objective="binary:logistic",eval_metric="auc",early_stopping_rounds=100,random_state=42,**cfg)
            model.fit(x[tr],y[tr],eval_set=[(x[va],y[va])],verbose=False)
        elif name=="extratrees":
            model=ExtraTreesClassifier(n_estimators=700,max_features=.6,min_samples_leaf=3,n_jobs=10,random_state=42)
            model.fit(x[tr],y[tr])
        else:
            e=np.load(ROOT/"data/processed/engagement_features.npy",allow_pickle=False)
            model=lgb.LGBMClassifier(n_estimators=800,learning_rate=.035,metric="auc",num_leaves=31,
                min_child_samples=150,reg_lambda=10,colsample_bytree=.85,n_jobs=10,verbosity=-1,random_state=42)
            # Player labels are weak engagement labels; assess both levels explicitly.
            model.fit(e[tr].reshape(-1,e.shape[-1]),np.repeat(y[tr],30),
                eval_set=[(e[va].reshape(-1,e.shape[-1]),np.repeat(y[va],30))],eval_metric="auc",
                callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False)])
            ep=model.predict_proba(e[va].reshape(-1,e.shape[-1]))[:,1].reshape(-1,30)
            pools={"mean":ep.mean(1),"top10":np.sort(ep,axis=1)[:,-10:].mean(1),
                   "logitmean":1/(1+np.exp(-np.log(np.clip(ep,1e-6,1-1e-6)/(1-np.clip(ep,1e-6,1-1e-6))).mean(1)))}
            pool=max(pools,key=lambda k:ranking(y[va],pools[k])["roc_auc"])
            pv=pools[pool]
            result={"model":name,"seconds":time.monotonic()-start,"pool":pool,
                "validation":ranking(y[va],pv),"engagement_validation":ranking(np.repeat(y[va],30),ep.flatten()),
                "aggregation_candidates":{k:ranking(y[va],v) for k,v in pools.items()},"best_iteration":model.best_iteration_}
            joblib.dump({"model":model,"level":"engagement","pool":pool,"feature_names":names["engagement"]},ROOT/f"models/{name}.joblib")
            np.savez(ROOT/f"results/validation/{name}.npz",ids=va,p=pv,engagement_p=ep)
            (ROOT/f"models/{name}.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
            continue
        pv=model.predict_proba(x[va])[:,1]
        result={"model":name,"seconds":time.monotonic()-start,"validation":ranking(y[va],pv),
                "train":ranking(y[tr],model.predict_proba(x[tr])[:,1]),"best_iteration":getattr(model,"best_iteration",None)}
        joblib.dump({"model":model,"columns":np.arange(x.shape[1]),"feature_names":names["player"]},ROOT/f"models/{name}.joblib")
        np.savez(ROOT/f"results/validation/{name}.npz",ids=va,p=pv)
        (ROOT/f"models/{name}.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=="__main__":main()
