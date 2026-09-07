"""GPU training with player-level labels, validation-only early stopping."""
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
import random
import time
import numpy as np
import torch
from torch import nn
from anticheat.data import ROOT,load_arrays,load_splits
from anticheat.neural import PlayerModel,sequence_channels
from anticheat.metrics import ranking


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--kind",choices=["lstm","cnn","hybrid"],default="hybrid")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=35)
    p.add_argument("--patience",type=int,default=9)
    p.add_argument("--batch-size",type=int,default=24)
    p.add_argument("--width",type=int,default=48)
    p.add_argument("--lr",type=float,default=.0007)
    p.add_argument("--weight-decay",type=float,default=.03)
    p.add_argument("--tag",default="")
    p.add_argument("--ema",action="store_true")
    p.add_argument("--fp32",action="store_true")
    p.add_argument("--raw-five",action="store_true")
    args=p.parse_args()
    start=time.monotonic()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError("This training run requires the available CUDA GPU")
    torch.backends.cudnn.benchmark=True
    torch.set_float32_matmul_precision("high")
    device="cuda"
    raw,y=load_arrays()
    frame,splits=load_splits()
    tr,va=splits["train"],splits["validation"]
    # Only training and validation rows are materialized for neural training.
    ids=np.r_[tr,va]
    channels=5 if args.kind=="lstm" or args.raw_five else 12
    x=torch.empty((len(ids),30,channels,192),device=device,dtype=torch.float16)
    for lo in range(0,len(ids),128):
        r=torch.from_numpy(raw[ids[lo:lo+128]]).to(device)
        x[lo:lo+len(r)]=sequence_channels(r,enriched=channels==12).half()
    del raw,r
    labels=torch.tensor(y[ids],device=device,dtype=torch.float32)
    f=torch.zeros((len(ids),1),device=device)
    mu=std=None
    feature_dim=0
    if args.kind=="hybrid":
        features=np.arcsinh(np.load(ROOT/"data/processed/player_features.npy",allow_pickle=False))
        mu=features[tr].mean(0);std=features[tr].std(0).clip(.05)
        features=np.clip((features[ids]-mu)/std,-8,8).astype(np.float32)
        f=torch.from_numpy(features).to(device)
        feature_dim=f.shape[1]
    model=PlayerModel(args.kind,feature_dim,args.width,channels).to(device)
    ema=copy.deepcopy(model).eval() if args.ema else None
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr/20)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.,device=device))
    # BF16 avoids loss scaling on supported CUDA devices.
    val_indices=torch.arange(len(tr),len(ids),device=device)
    def predict(indices,network=None):
        network=model if network is None else network
        network.eval()
        ps=[]
        with torch.inference_mode():
            for index in indices.split(args.batch_size):
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not args.fp32):z=network(x[index].float(),f[index])
                ps.append(z.float().sigmoid().cpu().numpy())
        return np.concatenate(ps)
    best=-1
    best_epoch=0
    history=[]
    name=f"{args.kind}_seed{args.seed}"+(f"_{args.tag}" if args.tag else "")
    (ROOT/"models").mkdir(exist_ok=True)
    (ROOT/"results/validation").mkdir(parents=True,exist_ok=True)
    print("Training",name,"params",sum(v.numel() for v in model.parameters()),"loaded",x.shape,flush=True)
    for epoch in range(1,args.epochs+1):
        tick=time.monotonic()
        model.train()
        order=torch.randperm(len(tr),device=device)
        losses=[]
        for index in order.split(args.batch_size):
            opt.zero_grad(set_to_none=True)
            xb=x[index].float()
            # Whole-player mirror: signs change together; target label is invariant.
            mirror=(torch.rand((len(index),1,1),device=device)>.5).float()*2-1
            xb[:,:,0,:]*=mirror;xb[:,:,2,:]*=mirror
            if channels==12:xb[:,:,5,:]*=mirror;xb[:,:,7,:]*=mirror
            # Raw-only models use mirroring; hybrid summaries must transform consistently.
            if args.kind=="hybrid":xb=x[index].float()
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not args.fp32):
                z=model(xb,f[index])
                loss=criterion(z.float(),labels[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,.01)
                    for eb,mb in zip(ema.buffers(),model.buffers()):
                        if eb.is_floating_point():eb.lerp_(mb,.01)
                        else:eb.copy_(mb)
            losses.append(loss.detach())
        scheduler.step()
        validation_model=model if ema is None else ema
        pv=predict(val_indices,validation_model)
        metrics=ranking(y[va],pv)
        row={"epoch":epoch,"loss":float(torch.stack(losses).mean()),"seconds":time.monotonic()-tick,**metrics}
        history.append(row)
        print(json.dumps(row),flush=True)
        if metrics["roc_auc"]>best+1e-5:
            best=metrics["roc_auc"];best_epoch=epoch
            checkpoint={"state_dict":{k:v.detach().cpu() for k,v in validation_model.state_dict().items()},
                        "config":vars(args),"feature_dim":feature_dim,
                        "feature_mean":torch.from_numpy(mu) if mu is not None else None,
                        "feature_std":torch.from_numpy(std) if std is not None else None,
                        "best_epoch":epoch,"validation":metrics}
            torch.save(checkpoint,ROOT/f"models/{name}.pt")
            np.savez(ROOT/f"results/validation/{name}.npz",ids=va,p=pv)
        (ROOT/f"models/{name}_history.json").write_text(json.dumps(history,indent=2))
        if epoch-best_epoch>=args.patience:break
    checkpoint=torch.load(ROOT/f"models/{name}.pt",weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    pt=predict(torch.arange(len(tr),device=device))
    result={"model":name,"seconds":time.monotonic()-start,"best_epoch":best_epoch,
            "parameters":sum(v.numel() for v in model.parameters()),"config":vars(args),
            "validation":checkpoint["validation"],"train":ranking(y[tr],pt),
            "torch_version":str(torch.__version__).split("+")[0]}
    (ROOT/f"models/{name}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
