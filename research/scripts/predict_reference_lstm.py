"""Run Ionuț’s revised LSTM on engagements or 30-engagement records."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from anticheat.reference_lstm import predict_checkpoint, pool_scores, apply_calibration

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reference_lstm_predictions.csv"))
    parser.add_argument("--artifacts", type=Path, default=ROOT / "experiments/lstm_revision")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    selection = json.loads((args.artifacts / "selection.json").read_text())
    calibration = json.loads((args.artifacts / "calibration.json").read_text())["improved"]
    filename = selection["models"]["improved"] + ".pt"
    model_path = args.artifacts / filename
    with model_path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != selection["sha256"][filename]:
            raise RuntimeError("Checkpoint does not match the frozen selection")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    raw = np.load(args.input, allow_pickle=False)
    engagement_p = predict_checkpoint(raw, checkpoint, device=args.device)
    unit = "player" if raw.ndim == 4 else "engagement"
    uncalibrated = (pool_scores(engagement_p, selection["player_pooling"]["improved"])
                    if unit == "player" else engagement_p)
    p = apply_calibration(uncalibrated, calibration[unit])
    threshold = calibration[unit]["thresholds"]
    out = pd.DataFrame({"array_row": np.arange(len(p)), "calibrated_score": p})
    for name in ["accuracy", "f1", "fpr_1pct"]:
        out[f"flag_{name}"] = p >= threshold[name]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} {unit} predictions to {args.output}")


if __name__ == "__main__":
    main()
