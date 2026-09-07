import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anticheat.cs2_training import validation_metrics


def main():
    root=Path('experiments/cs2cd_v1');cache=Path('data/cs2cd/processed/cache')
    assert (root/'cache.json').exists()
    assert not (root/'selection.json').exists()
    players=pd.read_csv(root/'player_index.csv');x=np.load(cache/'player_features.npy')
    y=players.label.to_numpy();reviewed=(players.source=='with_cheater_present').to_numpy()
    tr=np.flatnonzero(players.split=='train');va=np.flatnonzero(players.split=='validation')
    configs=[('lgbm39_leaves15',39,15),('lgbm39_leaves31',39,31),('lgbm5_leaves15',5,15)]
    for name,channels,leaves in configs:
        if (root/f'{name}_done.json').exists():continue
        features=np.flatnonzero(np.arange(x.shape[1])%39<channels);xx=x[:,features]
        def objective(y_true,pred):
            return 'cohort_mean_auc',validation_metrics(y_true,pred,reviewed[va])['selection_score'],True
        model=lgb.LGBMClassifier(n_estimators=2000,num_leaves=leaves,max_depth=-1,learning_rate=.025,
                                 min_child_samples=35,colsample_bytree=.8,subsample=.8,subsample_freq=1,
                                 reg_lambda=10,reg_alpha=.3,n_jobs=8,random_state=42,verbosity=-1,metric='None')
        t0=time.time()
        model.fit(xx[tr],y[tr],eval_set=[(xx[va],y[va])],eval_metric=objective,
                  callbacks=[lgb.early_stopping(100,first_metric_only=True),lgb.log_evaluation(100)])
        p=model.predict_proba(xx[va])[:,1];metric=validation_metrics(y[va],p,reviewed[va])
        joblib.dump({'model':model,'feature_indices':features,'channels':channels},root/f'{name}.joblib')
        np.savez(root/f'{name}_validation.npz',player_index=va,scores=p)
        report={'name':name,'channels':channels,'leaves':leaves,'best_iteration':model.best_iteration_,
                **metric,'seconds':time.time()-t0,'feature_count':len(features)}
        (root/f'{name}_done.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    # Diagnostics only. Metadata models are never eligible for the deployed blend.
    diagnostics={}
    for fields in [['source'],['map'],['source','map']]:
        model=make_pipeline(OneHotEncoder(handle_unknown='ignore'),LogisticRegression(C=1,max_iter=1000))
        model.fit(players.iloc[tr][fields],y[tr]);p=model.predict_proba(players.iloc[va][fields])[:,1]
        name='diagnostic_'+'_'.join(fields)
        joblib.dump({'model':model,'fields':fields},root/f'{name}.joblib')
        np.savez(root/f'{name}_validation.npz',player_index=va,scores=p)
        diagnostics[name]=validation_metrics(y[va],p,reviewed[va])
    (root/'metadata_diagnostics_validation.json').write_text(json.dumps(diagnostics,indent=2))
    print('Metadata diagnostics',json.dumps(diagnostics),flush=True)


if __name__=='__main__':main()
