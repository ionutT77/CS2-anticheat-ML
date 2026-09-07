import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
from anticheat.data import ROOT, load_arrays
from anticheat.features import engagement_features, aggregate_features

def main():
    start=time.monotonic()
    x,y=load_arrays()
    out=ROOT/"data/processed"
    out.mkdir(parents=True,exist_ok=True)
    player_out=event_out=None
    for lo in range(0,len(y),100):
        a=x[lo:lo+100]
        e,en=engagement_features(a.reshape(-1,192,5))
        e=e.reshape(len(a),30,-1)
        p,pn=aggregate_features(e,en)
        if player_out is None:
            player_out=np.lib.format.open_memmap(out/"player_features.npy",mode="w+",dtype="float32",shape=(len(y),p.shape[-1]))
            event_out=np.lib.format.open_memmap(out/"engagement_features.npy",mode="w+",dtype="float32",shape=(len(y),30,e.shape[-1]))
            (out/"feature_names.json").write_text(json.dumps({"player":pn,"engagement":en},indent=2))
        player_out[lo:lo+len(a)]=p
        event_out[lo:lo+len(a)]=e
        if lo % 1000 == 0:print(f"Features {lo}/{len(y)} elapsed={time.monotonic()-start:.1f}s D={p.shape[1]}",flush=True)
    player_out.flush()
    event_out.flush()
    print("Done",time.monotonic()-start,flush=True)

if __name__=="__main__":main()
