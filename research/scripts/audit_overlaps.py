"""Find shared 64-tick mouse paths, including shifted or different-victim windows.

64-bit rolling hashes nominate candidates; exact float equality verifies each link.
Constant/common input patterns are rejected by activity and uniqueness checks.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import time
import numpy as np
from anticheat.data import ROOT,load_arrays


def main():
    start=time.monotonic()
    x,y=load_arrays()
    v=np.ascontiguousarray(x.reshape(-1,192,5)[...,[0,1,4]])
    del x
    n=len(v);w=64
    bits=v.view(np.uint32).astype(np.uint64)
    # Mix each exact tick; hashing only nominates candidates, never establishes identity.
    tick=(bits[...,0]*np.uint64(0x9E3779B185EBCA87))^(bits[...,1]*np.uint64(0xC2B2AE3D27D4EB4F))^(bits[...,2]*np.uint64(0x165667B19E3779F9))
    del bits
    base=np.uint64(1000000007)
    power=np.uint64(pow(int(base),w,2**64))
    h=np.zeros(n,dtype=np.uint64)
    for j in range(w):h=h*base+tick[:,j]
    hashes=np.empty((n,192-w+1),dtype=np.uint64)
    hashes[:,0]=h
    for j in range(w,192):
        h=h*base+tick[:,j]-tick[:,j-w]*power
        hashes[:,j-w+1]=h
    del tick,h
    activity=np.any(np.abs(v[...,:2])>.01,axis=-1)
    changed=np.any(np.diff(v[...,:2],axis=1,prepend=v[:,:1,:2])!=0,axis=-1)
    for gate in [activity,changed]:
        cs=np.pad(gate.cumsum(1,dtype=np.int16),((0,0),(1,0)))
        count=cs[:,w:]-cs[:,:-w]
        hashes[count<24]=0
    del activity,changed,cs,count
    flat=hashes.ravel()
    order=np.argsort(flat,kind="stable")
    sorted_h=flat[order]
    matches=np.flatnonzero((sorted_h[1:]==sorted_h[:-1])&(sorted_h[1:]!=0))
    print("Rolling hashes sorted",len(flat),"candidate neighbors",len(matches),"seconds",time.monotonic()-start,flush=True)
    edges=set();candidates=0
    for k in matches:
        a,b=int(order[k]),int(order[k+1])
        ea,sa=divmod(a,129);eb,sb=divmod(b,129)
        pa,pb=ea//30,eb//30
        if pa==pb:continue
        edge=tuple(sorted((pa,pb)))
        if edge in edges:continue
        aa=v[ea,sa:sa+w];bb=v[eb,sb:sb+w]
        if np.array_equal(aa,bb) and len(np.unique(aa[:,:2],axis=0))>=16:
            edges.add(edge)
        candidates+=1
    out=ROOT/"data/processed"
    out.mkdir(parents=True,exist_ok=True)
    # Full-window attacker paths catch sparse matches that fail the 64-tick gate.
    full_seen={};full_edges=set()
    active_count=np.count_nonzero(np.abs(v[...,:2]).sum(-1)>.01,axis=1)
    for ei in np.flatnonzero(active_count>=16):
        pi=int(ei)//30
        key=hashlib.blake2b(v[ei].tobytes(),digest_size=16).digest()
        if key in full_seen and full_seen[key]!=pi:
            full_edges.add(tuple(sorted((pi,full_seen[key]))))
        else:full_seen[key]=pi
    np.save(out/"extra_movement_duplicate_edges.npy",np.array(sorted(full_edges),dtype=np.int64).reshape(-1,2))
    np.save(out/"overlap_duplicate_edges.npy",np.array(sorted(edges),dtype=np.int64).reshape(-1,2))
    report={"window_ticks":w,"shift_stride":1,"input_channels":[0,1,4],"minimum_active_ticks":24,
            "minimum_changed_ticks":24,"minimum_distinct_angle_pairs":16,"verified_record_edges":len(edges),
            "candidate_comparisons":candidates,"seconds":time.monotonic()-start,
            "method":"64-tick rolling hash at every offset, exact float equality verification; adjacent equal-hash records are connected transitively."}
    (ROOT/"reports/overlap_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
