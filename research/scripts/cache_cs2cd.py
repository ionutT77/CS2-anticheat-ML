import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anticheat.cs2_data import transform, FEATURES
from anticheat.cs2_features import summarize_events, summarize_player


def main():
    root = Path('experiments/cs2cd_v1')
    assert (root/'data_audit.json').exists()
    assert not (root/'selection.json').exists()
    dest = Path('data/cs2cd/processed/cache');dest.mkdir(parents=True,exist_ok=True)
    events = pd.read_parquet(root/'events.parquet')
    players = pd.read_csv(root/'players.csv');players=players[players.encounters>0].reset_index(drop=True)
    players.to_csv(root/'player_index.csv',index=False)
    n,c=len(events),len(FEATURES)
    xout=np.lib.format.open_memmap(dest/'x.npy',mode='w+',dtype=np.float16,shape=(n,256,c))
    fout=np.lib.format.open_memmap(dest/'event_features.npy',mode='w+',dtype=np.float32,shape=(n,14*c))
    total=np.zeros(c);total2=np.zeros(c);count=0;t0=time.time()
    for j,(path,g) in enumerate(events.groupby('npz',sort=False),1):
        with np.load(path) as d:
            raw=d['x'][g.local_index.to_numpy()]
        x=transform(raw); ix=g.index.to_numpy()
        xout[ix]=x.astype(np.float16);fout[ix]=summarize_events(x)
        if g.split.iloc[0]=='train':
            total+=x.sum((0,1),dtype=np.float64);total2+=np.square(x,dtype=np.float64).sum((0,1));count+=x.shape[0]*256
        if j%100==0:
            print(f'Cached {j} matches; {time.time()-t0:.0f}s',flush=True)
    mean=total/count;std=np.sqrt(np.maximum(total2/count-mean**2,.0001))
    (root/'normalization.json').write_text(json.dumps({'features':FEATURES,'mean':mean.tolist(),
        'std':std.tolist(),'training_ticks':count,'transform':'fixed asinh scales then training-only z-score'},indent=2))
    groups={key:g.sort_values('tick').index.to_numpy() for key,g in events.groupby('player_match')}
    bags=[groups[p] for p in players.player_match]
    np.savez(dest/'bags.npz',starts=np.cumsum([0]+[len(b) for b in bags]),indices=np.concatenate(bags))
    player_features=np.stack([summarize_player(fout[b]) for b in bags])
    np.save(dest/'player_features.npy',player_features)
    xout.flush();fout.flush()
    meta={'events':n,'eligible_players':len(players),'channels':c,'event_features':14*c,
          'player_features':player_features.shape[1],'seconds':time.time()-t0,
          'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                            ['anticheat/cs2_data.py','anticheat/cs2_features.py','scripts/cache_cs2cd.py']}}
    (root/'cache.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
