import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from anticheat.inference import predict

def main():
    p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("--output",default="predictions.csv");p.add_argument("--device",default="cpu");args=p.parse_args()
    result=predict(np.load(args.input,allow_pickle=False),args.device)
    frame=pd.DataFrame({k:v for k,v in result.items() if k!="component_scores"})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output,index=False);print(frame.to_string(index=False))

if __name__=="__main__":main()
