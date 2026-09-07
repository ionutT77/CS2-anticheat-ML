"""Freeze calibration first, then run a single evaluation on the reserved matches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anticheat.cs2_models import MatchDetector
from anticheat.cs2_training import predict_players
from anticheat.cs2_features import summarize_player
from anticheat.cs2_evaluation import ranking,threshold_metrics,cluster_bootstrap,exploratory_precision_threshold
from anticheat.reference_lstm import fit_calibration,apply_calibration,choose_thresholds


def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--test',action='store_true');args=ap.parse_args()
    root=Path('experiments/cs2cd_v1');cache=Path('data/cs2cd/processed/cache')
    selection=json.loads((root/'selection.json').read_text())
    for p,sha in selection['artifact_sha256'].items():assert file_hash(root/p)==sha,p
    for p,sha in selection['code_sha256'].items():assert file_hash(p)==sha,p
    assert not (root/'test_results.json').exists(),'Reserved test already evaluated; do not retune.'
    if args.test:
        calibration=json.loads((root/'calibration.json').read_text())
        assert calibration['selection_sha256']==file_hash(root/'selection.json')
    else:
        assert not (root/'calibration.json').exists(),'Calibration is already frozen.'
    split='test' if args.test else 'calibration'
    players=pd.read_csv(root/'player_index.csv');chosen=np.flatnonzero(players.split==split)
    y=players.label.to_numpy()[chosen];reviewed=(players.source=='with_cheater_present').to_numpy()[chosen]
    packed=np.load(cache/'bags.npz');ix,starts=packed['indices'],packed['starts']
    bags=[ix[a:b] for a,b in zip(starts[:-1],starts[1:])]
    xtab=np.load(cache/'player_features.npy')
    x=torch.from_numpy(np.load(cache/'x.npy')).cuda()
    norm=json.loads((root/'normalization.json').read_text())
    mean=torch.tensor(norm['mean'],device='cuda');std=torch.tensor(norm['std'],device='cuda')
    torch.set_num_threads(6);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    probabilities={};horizons={h:{} for h in [1,5,10]};parity={}
    va=np.flatnonzero(players.split=='validation')
    for name in selection['all_candidates']:
        if (root/f'{name}.pt').exists():
            checkpoint=torch.load(root/f'{name}.pt',map_location='cpu',weights_only=True);config=checkpoint['config']
            model=MatchDetector(config['channels'],config['architecture'],config.get('hierarchical',False)).cuda()
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            if not args.test:
                logits,_,_=predict_players(model,x,mean,std,bags,va)
                previous=np.load(root/f'{name}_validation.npz')['scores']
                parity[name]=float(np.max(np.abs(expit(logits)-previous)));assert parity[name]<2e-5,(name,parity[name])
            logits,event_logits,event_indices=predict_players(model,x,mean,std,bags,chosen)
            p=expit(logits)
            if args.test and name in selection['components']:
                for h in horizons:
                    logits,_,_=predict_players(model,x,mean,std,bags,chosen,horizon=h)
                    horizons[h][name]=expit(logits)
                np.savez(root/f'{name}_test_events.npz',event_index=event_indices,scores=expit(event_logits))
            del model;torch.cuda.empty_cache()
        else:
            saved=joblib.load(root/f'{name}.joblib');features=saved['feature_indices']
            p=saved['model'].predict_proba(xtab[chosen][:,features])[:,1]
            if args.test and name in selection['components']:
                event_features=np.load(cache/'event_features.npy',mmap_mode='r')
                for h in horizons:
                    hx=np.stack([summarize_player(event_features[bags[i][:h]]) for i in chosen])
                    horizons[h][name]=saved['model'].predict_proba(hx[:,features])[:,1]
        probabilities[name]=p
        print(split,'predictions complete',name,flush=True)
    probabilities['selected']=sum(weight*probabilities[name] for name,weight in selection['components'].items())
    for path in root.glob('diagnostic_*.joblib'):
        saved=joblib.load(path)
        probabilities[path.stem]=saved['model'].predict_proba(players.iloc[chosen][saved['fields']])[:,1]
    if not args.test:
        calibration={'selection_sha256':file_hash(root/'selection.json'),'models':{},'split':split,
                     'evaluation_code_sha256':file_hash(__file__),'parity':parity}
        for name,p in probabilities.items():
            fit=fit_calibration(y,p);calp=apply_calibration(p,fit);thresholds=choose_thresholds(y,calp)
            thresholds['precision99_exploratory']=exploratory_precision_threshold(y,calp)
            calibration['models'][name]={'platt':fit,'thresholds':thresholds,
                'calibration_metrics':{k:threshold_metrics(y,calp,t) for k,t in thresholds.items()}}
        (root/'calibration.json').write_text(json.dumps(calibration,indent=2))
        np.savez(root/'calibration_predictions.npz',player_index=chosen,**probabilities)
        print('Calibration and thresholds frozen; maximum validation replay difference',max(parity.values()),flush=True)
        return
    assert calibration['evaluation_code_sha256']==file_hash(__file__),'Evaluation changed after calibration.'
    result={'selection':selection['components'],'decision_unit':'eligible player-match','cohorts':{},
            'horizons':{},'calibration_sha256':file_hash(root/'calibration.json'),
            'selection_sha256':file_hash(root/'selection.json')}
    for cohort,mask in [('all',np.ones(len(y),dtype=bool)),('reviewed_only',reviewed)]:
        yy=y[mask];models={}
        for name,p in probabilities.items():
            pp=p[mask];cal=calibration['models'][name];calp=apply_calibration(pp,cal['platt'])
            models[name]={**ranking(yy,pp),'policies':{k:threshold_metrics(yy,calp,t) for k,t in cal['thresholds'].items()}}
        result['cohorts'][cohort]={'models':models,'all_noncheater':{**ranking(yy,np.zeros(len(yy))),
                     **threshold_metrics(yy,np.zeros(len(yy)),.5)},
                     'bootstrap':cluster_bootstrap(yy,{k:v[mask] for k,v in probabilities.items()
                                        if k in ['selected','lstm5','lstm39','tcn39','transformer39','hierarchical39',selection['best_tree']]},
                                       players.iloc[chosen].group.to_numpy()[mask])}
    for h,components in horizons.items():
        p=sum(weight*components[name] for name,weight in selection['components'].items())
        cal=calibration['models']['selected'];calp=apply_calibration(p,cal['platt'])
        result['horizons'][h]={'all':ranking(y,p),'reviewed_only':ranking(y[reviewed],p[reviewed]),
              'policies':{k:threshold_metrics(y,calp,t) for k,t in cal['thresholds'].items()},
              'note':'Uses the first up-to-N eligible encounters and the frozen full-match calibrator/thresholds.'}
    out=players.iloc[chosen].copy()
    for name,p in probabilities.items():out[name]=p
    out.to_csv(root/'test_predictions.csv',index=False)
    (root/'test_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({cohort:{'selected':data['models']['selected'],'all_noncheater':data['all_noncheater']}
                      for cohort,data in result['cohorts'].items()},indent=2),flush=True)


if __name__=='__main__':main()
