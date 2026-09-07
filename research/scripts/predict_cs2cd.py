import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anticheat.cs2_inference import predict


def main():
    ap=argparse.ArgumentParser(description='Score extracted CS2 pre-impact encounters per player-match.')
    ap.add_argument('input',type=Path,help='NPZ containing x, player and tick arrays, or NPY of one player\'s windows.')
    ap.add_argument('--model-dir',default='experiments/cs2cd_v1')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--player',help='Optionally score one match-local player ID only.')
    args=ap.parse_args();torch.set_num_threads(6)
    if args.input.suffix=='.npz':
        with np.load(args.input,allow_pickle=False) as data:
            x=data['x'];players=data['player'];ticks=data['tick']
    else:
        x=np.load(args.input,allow_pickle=False);players=np.repeat('player',len(x));ticks=np.arange(len(x))
    if args.player:
        mask=players==args.player;x=x[mask];players=players[mask];ticks=ticks[mask]
    rows=predict(x,players,ticks,args.model_dir,args.device)
    if not rows:
        raise SystemExit('No eligible encounters: no player decision was produced.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} player-match scores to {args.output}')


if __name__=='__main__':main()
