import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/".vendor"))
import json
import time
import joblib
import numpy as np
from catboost import CatBoostClassifier
from anticheat.data import load_splits
from anticheat.metrics import ranking


def main():
    frame,splits=load_splits();y=frame.label.to_numpy();tr=splits["train"];va=splits["validation"]
    x=np.load(ROOT/"data/processed/player_features.npy",allow_pickle=False)
    names=json.loads((ROOT/"data/processed/feature_names.json").read_text())["player"]
    for depth in [4,6]:
        name=f"catboost_depth{depth}"
        if (ROOT/f"models/{name}.joblib").exists():continue
        start=time.monotonic()
        model=CatBoostClassifier(iterations=2200,depth=depth,learning_rate=.035,l2_leaf_reg=10,
            loss_function="Logloss",eval_metric="AUC",thread_count=10,random_seed=42,
            border_count=64,random_strength=.5,verbose=False,allow_writing_files=False)
        model.fit(x[tr],y[tr],eval_set=(x[va],y[va]),early_stopping_rounds=150,use_best_model=True,verbose=False)
        pv=model.predict_proba(x[va])[:,1]
        result={"model":name,"seconds":time.monotonic()-start,"best_iteration":model.get_best_iteration(),
                "validation":ranking(y[va],pv),"train":ranking(y[tr],model.predict_proba(x[tr])[:,1])}
        model.save_model(str(ROOT/f"models/{name}.cbm"))
        joblib.dump({"model":model,"columns":np.arange(len(names)),"feature_names":names},ROOT/f"models/{name}.joblib")
        np.savez(ROOT/f"results/validation/{name}.npz",ids=va,p=pv)
        (ROOT/f"models/{name}.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=="__main__":main()
