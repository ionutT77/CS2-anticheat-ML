"""Validation-only selection; this command never reads calibration/test scores."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anticheat.cs2_training import validation_metrics
from scripts.train_cs2cd_neural import CONFIGS


def main():
    root=Path('experiments/cs2cd_v1')
    assert not (root/'selection.json').exists(),'Selection already frozen.'
    players=pd.read_csv(root/'player_index.csv');va=np.flatnonzero(players.split=='validation')
    y=players.label.to_numpy()[va];reviewed=(players.source=='with_cheater_present').to_numpy()[va]
    neural=[c['name'] for c in CONFIGS];trees=['lgbm39_leaves15','lgbm39_leaves31','lgbm5_leaves15']
    scores={};metrics={}
    for name in neural+trees:
        assert (root/f'{name}_done.json').exists(),name
        d=np.load(root/f'{name}_validation.npz');assert np.array_equal(d['player_index'],va)
        scores[name]=d['scores'];metrics[name]=validation_metrics(y,d['scores'],reviewed)
    def choose(names):
        best=None
        for name in names:
            a=metrics[name]
            if best is None:
                best=name;continue
            b=metrics[best]
            if a['selection_score']>b['selection_score']+.001 or (
               abs(a['selection_score']-b['selection_score'])<=.001 and a['reviewed_auc']>b['reviewed_auc']):
                best=name
        return best
    best_neural,best_tree=choose(neural),choose(trees)
    blends=[]
    for weight in [0.,.25,.5,.75,1.]:
        p=weight*scores[best_neural]+(1-weight)*scores[best_tree]
        metric=validation_metrics(y,p,reviewed)
        blends.append({'neural_weight':weight,**metric})
    winner=max(blends,key=lambda r:(r['selection_score'],r['reviewed_auc']))
    components={best_neural:winner['neural_weight'],best_tree:1-winner['neural_weight']}
    components={k:v for k,v in components.items() if v>0}
    artifacts=['protocol.json','match_splits.csv','normalization.json','data_audit.json','player_index.csv']
    artifacts += [name+('.pt' if name in neural else '.joblib') for name in neural+trees]
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in artifacts}
    code=['anticheat/cs2_data.py','anticheat/cs2_models.py','anticheat/cs2_features.py',
          'anticheat/cs2_training.py','scripts/train_cs2cd_neural.py','scripts/train_cs2cd_tabular.py']
    selection={'components':components,'best_neural':best_neural,'best_tree':best_tree,
               'validation_models':metrics,'validation_blends':blends,'winner':winner,'all_candidates':neural+trees,
               'artifact_sha256':hashes,'code_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code}}
    (root/'selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
