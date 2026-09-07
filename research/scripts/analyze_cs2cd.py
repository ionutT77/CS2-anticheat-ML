"""Paired feature-ablation intervals and an unchanged-label review queue.

Reads frozen predictions only; never changes models, thresholds or test membership.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anticheat.cs2_evaluation import weighted_auc
from anticheat.reference_lstm import apply_calibration


def main():
    root=Path('experiments/cs2cd_v1');pred=pd.read_csv(root/'test_predictions.csv')
    cal=json.loads((root/'calibration.json').read_text())['models']['selected']
    results={}
    for cohort,frame in [('all',pred),('reviewed_only',pred[pred.source=='with_cheater_present'])]:
        y=frame.label.to_numpy();groups,inverse=np.unique(frame.group,return_inverse=True)
        result={}
        for before,after in [('lstm5','lstm39'),('lgbm5_leaves15','lgbm39_leaves15')]:
            a,b=weighted_auc(y,frame[before].to_numpy()),weighted_auc(y,frame[after].to_numpy())
            rng=np.random.default_rng(9072026);values=[]
            for _ in range(2000):
                w=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))[inverse]
                if (w*y).sum()==0 or (w*(1-y)).sum()==0:continue
                values.append(b(w)-a(w))
            result[after+'_minus_'+before]={'auc_difference':float(roc_auc_score(y,frame[after])-roc_auc_score(y,frame[before])),
                                          'ci95':np.quantile(values,[.025,.975]).tolist()}
        results[cohort]=result
    results['note']='Supplementary paired analysis of preregistered feature ablations; no new fitting, selection or threshold changes.'
    (root/'feature_ablation_intervals.json').write_text(json.dumps(results,indent=2))
    scores=apply_calibration(pred.selected.to_numpy(),cal['platt'])
    flagged=scores>=cal['thresholds']['fpr_1pct']
    queue=pred[(pred.label==0)&flagged].copy()
    queue['calibrated_score']=scores[(pred.label==0)&flagged]
    columns=['match_id','player','source','map','label','encounters','selected','calibrated_score',
             'tcn39_seed123','lgbm39_leaves15']
    queue[columns].sort_values('calibrated_score',ascending=False).to_csv(root/'strict_policy_negative_label_review_queue.csv',index=False)
    print(json.dumps(results,indent=2));print('Negative-label review queue:',len(queue),'records')


if __name__=='__main__':main()
