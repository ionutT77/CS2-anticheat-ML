# =============================================================================
# CS2 Anti-Cheat — LSTM Training Script for Kaggle
# =============================================================================
# HOW TO USE:
#   1. On Kaggle, create a new Notebook.
#   2. Add your dataset: "csgo-cheating-dataset" (or wherever cheaters.npy /
#      legit.npy live). The data will be at /kaggle/input/<dataset-name>/
#   3. Copy-paste this entire file into a single code cell (or upload as a
#      script and call: !python kaggle_training.py)
#   4. Settings → Accelerator → GPU T4 x2
#   5. Run All. Training ~15-30s/epoch instead of ~5-10 min on CPU.
#   6. After training, download best_lstm.pt from the Output panel.
#      Put it in your local  CS2-anticheat-ML/models/  folder.
# =============================================================================

import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)
from tqdm.auto import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# !! Change this to match your Kaggle dataset input path !!
DATA_DIR = "/kaggle/input/datasets/emstatsl/csgo-cheating-dataset"   # folder with cheaters.npy & legit.npy
OUTPUT_DIR = "/kaggle/working/"                      # Kaggle output — files here are downloadable

BATCH_SIZE  = 512     # Larger batches work better on GPU (256 was for CPU)
N_EPOCHS    = 50
LR          = 5e-4
POS_WEIGHT  = 2.0     # 5:1 class imbalance compensation
PATIENCE    = 10      # Early stopping patience (epochs)
SEED        = 42
DROPOUT = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── FEATURE METADATA ────────────────────────────────────────────────────────

FEATURE_NAMES      = ["AttackerDeltaYaw", "AttackerDeltaPitch",
                       "CrosshairToVictimYaw", "CrosshairToVictimPitch", "Firing"]
CONTINUOUS_FEATURES = [0, 1, 2, 3]
BINARY_FEATURES     = [4]

# =============================================================================
# DATA PIPELINE  (mirrors src/data/dataset.py exactly)
# =============================================================================

def load_raw_data(data_dir):
    cheaters = np.load(os.path.join(data_dir, "cheaters/cheaters.npy"))  # (2000, 30, 192, 5)
    legit    = np.load(os.path.join(data_dir, "legit/legit.npy"))     # (10000, 30, 192, 5)
    print(f"Loaded cheaters: {cheaters.shape}  legit: {legit.shape}")

    data   = np.concatenate([cheaters, legit], axis=0)
    labels = np.concatenate([np.ones(len(cheaters), dtype=np.int64),
                              np.zeros(len(legit),   dtype=np.int64)])
    print(f"Combined: {data.shape} | Cheaters: {(labels==1).sum()}  Legit: {(labels==0).sum()}")
    return data, labels


def split_by_player(data, labels, val_size=0.15, test_size=0.15, seed=42):
    indices = np.arange(len(data))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size,
                                               stratify=labels, random_state=seed)
    val_relative = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_relative,
                                          stratify=labels[train_val_idx], random_state=seed)
    splits = {
        "train": {"data": data[train_idx], "labels": labels[train_idx]},
        "val":   {"data": data[val_idx],   "labels": labels[val_idx]},
        "test":  {"data": data[test_idx],  "labels": labels[test_idx]},
    }
    for name, s in splits.items():
        n, n_ch = len(s["labels"]), (s["labels"]==1).sum()
        print(f"  {name:5s}: {n:5d} players ({n_ch:4d} cheaters) → {n*30:6d} engagements")
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    return splits


def compute_normalization_stats(train_data):
    flat = train_data.reshape(-1, 5)
    mean = np.zeros(5, dtype=np.float32)
    std  = np.ones(5,  dtype=np.float32)
    for i in CONTINUOUS_FEATURES:
        mean[i] = flat[:, i].mean()
        std[i]  = max(flat[:, i].std(), 1e-8)
    print("Normalization stats:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:30s}  mean={mean[i]:8.4f}  std={std[i]:8.4f}")
    return {"mean": mean, "std": std}


class EngagementDataset(Dataset):
    def __init__(self, player_data, player_labels, norm_stats=None):
        n_players, n_eng, n_ticks, n_feat = player_data.shape
        self.engagements   = player_data.reshape(-1, n_ticks, n_feat).astype(np.float32)
        self.labels        = np.repeat(player_labels, n_eng).astype(np.float32)
        self.player_indices = np.repeat(np.arange(n_players), n_eng)

        if norm_stats is not None:
            mean = norm_stats["mean"].reshape(1, 1, -1)
            std  = norm_stats["std"].reshape(1, 1, -1)
            self.engagements = (self.engagements - mean) / std

        self.engagements = torch.tensor(self.engagements, dtype=torch.float32)
        self.labels      = torch.tensor(self.labels,      dtype=torch.float32)
        print(f"  Dataset: {len(self)} samples "
              f"({int((self.labels==1).sum())} cheater / {int((self.labels==0).sum())} legit)")

    def __len__(self):  return len(self.labels)
    def __getitem__(self, idx): return self.engagements[idx], self.labels[idx]


def get_dataloaders(data_dir, batch_size=512, seed=42, num_workers=2):
    print("=" * 60)
    data, labels = load_raw_data(data_dir)

    print("\nSplitting by player...")
    splits = split_by_player(data, labels, seed=seed)

    print("\nComputing normalization stats...")
    norm_stats = compute_normalization_stats(splits["train"]["data"])

    print("\nCreating datasets...")
    datasets = {name: EngagementDataset(splits[name]["data"],
                                        splits[name]["labels"],
                                        norm_stats=norm_stats)
                for name in ["train", "val", "test"]}

    # Balanced sampler for training
    train_labels  = datasets["train"].labels.numpy().astype(int)
    class_weights = 1.0 / np.bincount(train_labels)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.float64),
                                    num_samples=len(datasets["train"]), replacement=True)

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=True),
        "val":   DataLoader(datasets["val"],   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(datasets["test"],  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }
    print("\nDataLoaders ready:")
    for name, loader in loaders.items():
        print(f"  {name:5s}: {len(loader):4d} batches (batch_size={batch_size})")
    print("=" * 60)
    return loaders, norm_stats, datasets


# =============================================================================
# MODEL  (mirrors src/models/lstm_detector.py exactly)
# =============================================================================

class LSTMAimbotDetector(nn.Module):
    def __init__(self, input_size=5, hidden_size_1=128, hidden_size_2=64,
                 fc_size=32, dropout=DROPOUT):
        super().__init__()
        self.lstm1      = nn.LSTM(input_size, hidden_size_1, batch_first=True, num_layers=1)
        self.dropout1   = nn.Dropout(dropout)
        self.lstm2      = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_2, fc_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_size, 1), nn.Sigmoid(),
        )
        self.config = dict(input_size=input_size, hidden_size_1=hidden_size_1,
                           hidden_size_2=hidden_size_2, fc_size=fc_size, dropout=dropout)

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1    = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        return self.classifier(out2[:, -1, :])  # last hidden state → (batch, 1)

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# =============================================================================
# TRAINER  (mirrors src/training/trainer.py exactly)
# =============================================================================

class Trainer:
    def __init__(self, model, device, lr=1e-3, pos_weight=None,
                 patience=10, checkpoint_dir=OUTPUT_DIR):
        self.device    = device
        self.model     = model.to(device)
        self.pos_weight = pos_weight
        self.criterion  = nn.BCELoss(reduction="none")
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) #added weight decay to prevent overfitting
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(  # if the model memorizes data, this weight_decay shrinks every weight towards 0, forcing it to generalize
            self.optimizer, mode="min", factor=0.5, patience=5)
        self.patience          = patience
        self.best_val_loss     = float("inf")
        self.epochs_no_improve = 0
        self.best_epoch        = 0
        self.checkpoint_dir    = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(checkpoint_dir, "best_lstm.pt")
        self.history = {k: [] for k in ["train_loss", "val_loss", "val_accuracy",
                                         "val_precision", "val_recall", "val_f1",
                                         "val_roc_auc", "lr", "epoch_time"]}

    def _weighted_loss(self, preds, targets):
        loss = self.criterion(preds, targets)
        if self.pos_weight is not None:
            w = torch.where(targets == 1.0,
                            torch.tensor(self.pos_weight, device=self.device),
                            torch.tensor(1.0, device=self.device))
            loss = loss * w
        return loss.mean()

    def train_one_epoch(self, loader, epoch, n_epochs):
        self.model.train()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{n_epochs} [train]",
                    unit="batch", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x).squeeze(-1)
            loss  = self._weighted_loss(preds, y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item(); n += 1
            pbar.set_postfix(loss=f"{total/n:.4f}")
        return total / n

    @torch.no_grad()
    def evaluate(self, loader, desc="val"):
        self.model.eval()
        all_probs, all_labels, total, n = [], [], 0.0, 0
        pbar = tqdm(loader, desc=f"           [{desc}]", unit="batch", leave=False)
        for x, y in pbar:
            x, y  = x.to(self.device), y.to(self.device)
            preds = self.model(x).squeeze(-1)
            loss  = self._weighted_loss(preds, y)
            total += loss.item(); n += 1
            all_probs.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            pbar.set_postfix(loss=f"{total/n:.4f}")
        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        all_preds  = (all_probs >= 0.5).astype(int)
        metrics = {
            "loss":      total / max(n, 1),
            "accuracy":  accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall":    recall_score(all_labels, all_preds, zero_division=0),
            "f1":        f1_score(all_labels, all_preds, zero_division=0),
            "roc_auc":   roc_auc_score(all_labels, all_probs)
                         if len(np.unique(all_labels)) > 1 else 0.0,
        }
        return metrics, all_probs, all_labels

    def train(self, train_loader, val_loader, n_epochs=50):
        print("=" * 70)
        print(f"Training on {self.device} | Max epochs: {n_epochs} | Patience: {self.patience}")
        print(f"Pos weight: {self.pos_weight} | LR: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 70)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch, n_epochs)
            val_metrics, _, _ = self.evaluate(val_loader, desc="val")
            val_loss = val_metrics["loss"]
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            for k, v in [("train_loss", train_loss), ("val_loss", val_loss),
                         ("val_accuracy", val_metrics["accuracy"]),
                         ("val_precision", val_metrics["precision"]),
                         ("val_recall", val_metrics["recall"]),
                         ("val_f1", val_metrics["f1"]),
                         ("val_roc_auc", val_metrics["roc_auc"]),
                         ("lr", current_lr), ("epoch_time", elapsed)]:
                self.history[k].append(v)

            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.3f} | Val AUC: {val_metrics['roc_auc']:.3f} | "
                  f"LR: {current_lr:.1e} | {elapsed:.1f}s")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch
                self.epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss, "val_metrics": val_metrics,
                    "model_config": self.model.config,
                    "history": self.history,
                }, self.best_model_path)
                print(f"  ^ New best model saved (val_loss={val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"\n[STOP] Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    print(f"   Best epoch: {self.best_epoch} (val_loss={self.best_val_loss:.4f})")
                    break

        print(f"\nLoading best model from epoch {self.best_epoch}...")
        ckpt = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print("Training complete!")
        return self.history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__" or True:   # "or True" makes it run in a notebook cell too

    # 1. Data
    loaders, norm_stats, datasets = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seed=SEED,
        num_workers=0,          # Kaggle supports workers (unlike Windows)
    )

    # 2. Model
    model = LSTMAimbotDetector()
    total, trainable = model.count_parameters()
    print(f"\nModel parameters: {total:,} total  ({trainable:,} trainable)")

    # 3. Train
    trainer = Trainer(
        model=model,
        device=DEVICE,
        lr=LR,
        pos_weight=POS_WEIGHT,
        patience=PATIENCE,
        checkpoint_dir=OUTPUT_DIR,
    )
    history = trainer.train(loaders["train"], loaders["val"], n_epochs=N_EPOCHS)

    # 4. Final test evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)
    test_metrics, _, _ = trainer.evaluate(loaders["test"], desc="test")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # 5. Save norm stats alongside the model (needed for inference)
    np.save(os.path.join(OUTPUT_DIR, "norm_stats_mean.npy"), norm_stats["mean"])
    np.save(os.path.join(OUTPUT_DIR, "norm_stats_std.npy"),  norm_stats["std"])

    print(f"\nOutputs saved to {OUTPUT_DIR}:")
    print(f"  best_lstm.pt         ← model checkpoint")
    print(f"  norm_stats_mean.npy  ← normalization mean (needed for inference)")
    print(f"  norm_stats_std.npy   ← normalization std  (needed for inference)")
    print("\nDownload these 3 files from the Kaggle Output panel.")
