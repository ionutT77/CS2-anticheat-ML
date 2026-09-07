"""Train candidates using only train/validation; final holdout is never scored here."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from anticheat.data import ROOT,load_splits
from anticheat.metrics import ranking


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--only",default="all")
    args=parser.parse_args()
    frame,splits=load_splits()
    y=frame.label.to_numpy()
    x=np.load(ROOT/"data/processed/player_features.npy",allow_pickle=False)
    names=json.loads((ROOT/"data/processed/feature_names.json").read_text())["player"]
    tr,va=splits["train"],splits["validation"]
    models=ROOT/"models"
    models.mkdir(exist_ok=True)
    preds=ROOT/"results/validation"
    preds.mkdir(parents=True,exist_ok=True)
    records=[]
    configs=[("lgbm_leaves15",{"num_leaves":15,"min_child_samples":40,"reg_lambda":5.}),
             ("lgbm_leaves31",{"num_leaves":31,"min_child_samples":40,"reg_lambda":5.}),
             ("lgbm_leaves63",{"num_leaves":63,"min_child_samples":60,"reg_lambda":10.}),
             ("lgbm_pre_only",{"num_leaves":15,"min_child_samples":40,"reg_lambda":5.}),
             ("lgbm_movement_only",{"num_leaves":15,"min_child_samples":40,"reg_lambda":5.}),
             ("lgbm_no_zero_features",{"num_leaves":15,"min_child_samples":40,"reg_lambda":5.})]
    if args.only in ["all","logistic"]:
        configs.insert(0,("logistic",{}))
    for name,config in configs:
        if args.only!="all" and name!=args.only:continue
        if (models/f"{name}.joblib").exists():
            print("Already trained",name,flush=True)
            continue
        start=time.monotonic()
        columns=np.arange(len(names))
        if name=="lgbm_pre_only":
            columns=np.array([i for i,n in enumerate(names) if n.startswith(("pre.","near."))])
        elif name=="lgbm_movement_only":
            columns=np.array([i for i,n in enumerate(names) if n.split('.')[1] in ["delta_yaw","delta_pitch","speed","acceleration","jerk"] and "firing" not in n])
        elif name=="lgbm_no_zero_features":
            columns=np.array([i for i,n in enumerate(names) if "allzero" not in n and "idle_fraction" not in n])
        xx=x[:,columns]
        if name=="logistic":
            model=make_pipeline(FunctionTransformer(np.arcsinh),StandardScaler(),LogisticRegression(C=.03,max_iter=2000))
            model.fit(xx[tr],y[tr])
        else:
            model=lgb.LGBMClassifier(n_estimators=1800,learning_rate=.025,metric="auc",verbosity=-1,n_jobs=10,
                colsample_bytree=.8,subsample=.85,subsample_freq=1,reg_alpha=.1,
                random_state=42,**config)
            model.fit(xx[tr],y[tr],eval_set=[(xx[va],y[va])],eval_metric="auc",
                callbacks=[lgb.early_stopping(120,first_metric_only=True,verbose=False)])
        pv=model.predict_proba(xx[va])[:,1]
        pt=model.predict_proba(xx[tr])[:,1]
        result={"model":name,"seconds":time.monotonic()-start,"features":len(columns),
                "best_iteration":getattr(model,"best_iteration_",None),
                "validation":ranking(y[va],pv),"train":ranking(y[tr],pt)}
        records.append(result)
        joblib.dump({"model":model,"columns":columns,"feature_names":[names[i] for i in columns]},models/f"{name}.joblib")
        np.savez(preds/f"{name}.npz",ids=va,p=pv)
        (models/f"{name}.json").write_text(json.dumps(result,indent=2))
        if hasattr(model,"booster_"):
            pd.DataFrame({"feature":[names[i] for i in columns],"gain":model.booster_.feature_importance(importance_type="gain")}).sort_values("gain",ascending=False).to_csv(models/f"{name}_importance.csv",index=False)
        print(json.dumps(result),flush=True)
    print("Completed tabular candidates",flush=True)


if __name__=="__main__":main()
