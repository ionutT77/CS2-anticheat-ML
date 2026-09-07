import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anticheat.cs2_models import MatchDetector
from anticheat.cs2_training import normalize, predict_players, validation_metrics

CONFIGS=[
    {'name':'lstm5','architecture':'lstm','channels':5,'seed':42},
    {'name':'lstm39','architecture':'lstm','channels':39,'seed':42},
    {'name':'tcn39','architecture':'tcn','channels':39,'seed':42},
    {'name':'transformer39','architecture':'transformer','channels':39,'seed':42},
    {'name':'hierarchical39','architecture':'tcn','channels':39,'seed':42,'hierarchical':True},
    {'name':'tcn39_seed123','architecture':'tcn','channels':39,'seed':123},
    {'name':'tcn39_reviewed','architecture':'tcn','channels':39,'seed':42,'reviewed_only':True},
]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only');ap.add_argument('--epochs',type=int,default=40)
    args=ap.parse_args();root=Path('experiments/cs2cd_v1');cache=Path('data/cs2cd/processed/cache')
    assert (root/'cache.json').exists()
    assert not (root/'selection.json').exists(),'Selection frozen; no more training.'
    protocol=json.loads((root/'protocol.json').read_text())
    assert hashlib.sha256(Path('anticheat/cs2_data.py').read_bytes()).hexdigest()==protocol['extraction_source_sha256']
    torch.set_num_threads(6);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    players=pd.read_csv(root/'player_index.csv');y=players.label.to_numpy().astype(np.float32)
    reviewed=(players.source=='with_cheater_present').to_numpy()
    tr=np.flatnonzero(players.split=='train');va=np.flatnonzero(players.split=='validation')
    packed=np.load(cache/'bags.npz');ix,starts=packed['indices'],packed['starts']
    bags=[ix[a:b] for a,b in zip(starts[:-1],starts[1:])]
    x=torch.from_numpy(np.load(cache/'x.npy')).cuda()
    norm=json.loads((root/'normalization.json').read_text())
    mean=torch.tensor(norm['mean'],device='cuda');std=torch.tensor(norm['std'],device='cuda')
    configs=[c for c in CONFIGS if args.only is None or args.only==c['name']]
    manifest={"configs":configs,"epochs":args.epochs,"batch_players":32,"bag_size":16,
              "auxiliary_event_loss":.25,"mixed_precision_training":"bfloat16",'validation_precision':'float32 TF32 disabled',
              'code_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                 ['scripts/train_cs2cd_neural.py','anticheat/cs2_models.py','anticheat/cs2_training.py']}}
    (root/('neural_plan'+('_'+args.only if args.only else '')+'.json')).write_text(json.dumps(manifest,indent=2))
    for config in configs:
        name=config['name']
        if (root/f'{name}_done.json').exists():
            print(name,'already complete',flush=True);continue
        seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed)
        train=tr[reviewed[tr]] if config.get('reviewed_only') else tr
        model=MatchDetector(config['channels'],config['architecture'],config.get('hierarchical',False)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=.00002)
        best=-1;best_reviewed=-1;best_epoch=0;history=[];start=time.time()
        print('Training',name,'players',len(train),'parameters',sum(p.numel() for p in model.parameters()),flush=True)
        for epoch in range(1,args.epochs+1):
            model.train();losses=[];t0=time.time()
            for offset in range(0,len(train),32):
                if offset==0:order=rng.permutation(train)
                batch=order[offset:offset+32]
                bidx=np.zeros((len(batch),16),dtype=np.int64);mask=np.zeros(bidx.shape,dtype=bool)
                for j,player in enumerate(batch):
                    bag=bags[player]
                    bidx[j,:]=bag[0]
                    chosen=np.sort(rng.choice(len(bag),16,replace=False)) if len(bag)>16 else np.arange(len(bag))
                    bidx[j,:len(chosen)]=bag[chosen];mask[j,:len(chosen)]=True
                xx=normalize(x[bidx],mean,std,config['channels']);mm=torch.from_numpy(mask).cuda()
                target=torch.from_numpy(y[batch]).cuda()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,event_logits=model(xx,mm)
                    primary=nn.functional.binary_cross_entropy_with_logits(logits.float(),target)
                    event_loss=nn.functional.binary_cross_entropy_with_logits(event_logits.float(),target[:,None].expand_as(event_logits),reduction='none')
                    auxiliary=((event_loss*mm).sum(1)/mm.sum(1)).mean()
                    loss=primary+.25*auxiliary
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                losses.append(float(loss.detach()))
            schedule.step()
            val_logits,_,_=predict_players(model,x,mean,std,bags,va)
            metric=validation_metrics(y[va],val_logits,reviewed[va])
            row={'epoch':epoch,'loss':float(np.mean(losses)),**metric,'seconds':time.time()-t0};history.append(row)
            score,review=metric['selection_score'],metric['reviewed_auc']
            improved=score>best+.001 or (abs(score-best)<=.001 and review>best_reviewed)
            if improved:
                best=score;best_reviewed=review;best_epoch=epoch
                torch.save({'config':config,'state_dict':model.state_dict(),'normalization':norm,'epoch':epoch,
                            'validation':metric,'features':norm['features'][:config['channels']]},root/f'{name}.pt')
                np.savez(root/f'{name}_validation.npz',player_index=va,logits=val_logits,scores=expit(val_logits))
            (root/f'{name}_history.json').write_text(json.dumps(history,indent=2))
            print(name,json.dumps(row),flush=True)
            if epoch-best_epoch>=8:break
        (root/f'{name}_done.json').write_text(json.dumps({'config':config,'best_epoch':best_epoch,
            'epochs':len(history),'selection_score':best,'reviewed_auc':best_reviewed,'seconds':time.time()-start,
            'parameters':sum(p.numel() for p in model.parameters()),'train_players':len(train)},indent=2))
        del model,opt;torch.cuda.empty_cache()


if __name__=='__main__':main()
