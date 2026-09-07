from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["delta_yaw", "delta_pitch", "error_yaw", "error_pitch", "firing"]


def load_raw():
    base = ROOT / "data/raw/extracted"
    clean = np.load(base / "legit/legit.npy", mmap_mode="r", allow_pickle=False)
    cheat = np.load(base / "cheaters/cheaters.npy", mmap_mode="r", allow_pickle=False)
    return clean, cheat


def load_arrays():
    clean, cheat = load_raw()
    return np.concatenate([clean, cheat]), np.r_[np.zeros(len(clean)), np.ones(len(cheat))].astype(np.int64)


def load_splits():
    frame = pd.read_csv(ROOT / "data/processed/splits.csv")
    return frame, {s: frame.loc[frame.split == s, "record_id"].to_numpy() for s in ["train", "validation", "calibration", "test"]}
