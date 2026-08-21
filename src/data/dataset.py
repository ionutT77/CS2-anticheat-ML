"""
Dataset loading, splitting, and PyTorch Dataset for the Kaggle CSGO Cheating Dataset.

Design decisions documented in PROJECT_JOURNAL.md Entry 9:
  - Engagement-level classification: each (192, 5) engagement is one sample
  - Player-level splitting: all 30 engagements from one player stay in the same split
  - Global z-score normalization on 4 continuous features, Firing left as-is
  - Player verdict requires multiple flagged engagements (not just one)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# ─── Feature metadata ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "AttackerDeltaYaw",
    "AttackerDeltaPitch",
    "CrosshairToVictimYaw",
    "CrosshairToVictimPitch",
    "Firing",
]

# Indices of continuous features (to be normalized) vs binary features (left as-is)
CONTINUOUS_FEATURES = [0, 1, 2, 3]  # DeltaYaw, DeltaPitch, C2VYaw, C2VPitch
BINARY_FEATURES = [4]               # Firing


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_raw_data(data_dir="data/raw"):
    """
    Load the raw .npy files and create labels.

    Returns
    -------
    data : np.ndarray, shape (12000, 30, 192, 5)
        All players concatenated (cheaters first, then legit).
    labels : np.ndarray, shape (12000,)
        1 = cheater, 0 = legit.
    """
    cheaters_path = os.path.join(data_dir, "cheaters.npy")
    legit_path = os.path.join(data_dir, "legit.npy")

    cheaters = np.load(cheaters_path)  # (2000, 30, 192, 5)
    legit = np.load(legit_path)        # (10000, 30, 192, 5)

    print(f"Loaded cheaters: {cheaters.shape}  dtype={cheaters.dtype}")
    print(f"Loaded legit:    {legit.shape}  dtype={legit.dtype}")

    # Concatenate: cheaters first, then legit
    data = np.concatenate([cheaters, legit], axis=0)  # (12000, 30, 192, 5)

    # Labels: 1 = cheater, 0 = legit
    labels = np.concatenate([
        np.ones(len(cheaters), dtype=np.int64),
        np.zeros(len(legit), dtype=np.int64),
    ])

    print(f"Combined: {data.shape}  labels: {labels.shape}")
    print(f"  Cheaters: {(labels == 1).sum()}  Legit: {(labels == 0).sum()}")

    return data, labels


# ─── Splitting ────────────────────────────────────────────────────────────────

def split_by_player(data, labels, val_size=0.15, test_size=0.15, seed=42):
    """
    Split data by PLAYER to prevent data leakage.

    All 30 engagements from a single player go into the same split.
    Stratified by label to preserve class ratios across splits.

    Parameters
    ----------
    data : np.ndarray, shape (N_players, 30, 192, 5)
    labels : np.ndarray, shape (N_players,)
    val_size : float — fraction of players for validation
    test_size : float — fraction of players for test
    seed : int — random seed for reproducibility

    Returns
    -------
    splits : dict with keys 'train', 'val', 'test'
        Each value is a dict with 'data' and 'labels' (player-level arrays)
    """
    n_players = len(data)
    indices = np.arange(n_players)

    # First split: separate test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    # Second split: separate validation from training
    # val_size is relative to original, so adjust for the remaining data
    val_relative = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        stratify=labels[train_val_idx],
        random_state=seed,
    )

    splits = {
        "train": {"data": data[train_idx], "labels": labels[train_idx]},
        "val":   {"data": data[val_idx],   "labels": labels[val_idx]},
        "test":  {"data": data[test_idx],  "labels": labels[test_idx]},
    }

    # Print split statistics
    for name, split in splits.items():
        n = len(split["labels"])
        n_ch = (split["labels"] == 1).sum()
        n_lg = (split["labels"] == 0).sum()
        print(f"  {name:5s}: {n:5d} players  "
              f"({n_ch:4d} cheaters, {n_lg:4d} legit)  "
              f"-> {n * 30:6d} engagements")

    # Verify no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_players

    return splits


# ─── Normalization ────────────────────────────────────────────────────────────

def compute_normalization_stats(train_data):
    """
    Compute per-feature mean and std from TRAINING data only.

    Only normalizes the 4 continuous features. Firing (binary) is excluded.

    Parameters
    ----------
    train_data : np.ndarray, shape (N_players, 30, 192, 5)

    Returns
    -------
    stats : dict with 'mean' and 'std', each shape (5,)
        Binary feature (index 4) has mean=0, std=1 so normalization is a no-op.
    """
    # Flatten to (all_ticks, 5)
    flat = train_data.reshape(-1, 5)

    mean = np.zeros(5, dtype=np.float32)
    std = np.ones(5, dtype=np.float32)

    for i in CONTINUOUS_FEATURES:
        mean[i] = flat[:, i].mean()
        std[i] = flat[:, i].std()
        # Guard against zero std (shouldn't happen, but safety first)
        if std[i] < 1e-8:
            std[i] = 1.0

    print(f"Normalization stats (training set):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:30s}  mean={mean[i]:8.4f}  std={std[i]:8.4f}")

    return {"mean": mean, "std": std}


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class EngagementDataset(Dataset):
    """
    PyTorch Dataset that serves individual engagements.

    Takes player-level data (N_players, 30, 192, 5) and flattens it to
    engagement-level (N_players * 30, 192, 5). Each sample is one engagement.

    Parameters
    ----------
    player_data : np.ndarray, shape (N_players, 30, 192, 5)
    player_labels : np.ndarray, shape (N_players,)
        Each player's label is applied to all 30 of their engagements.
    norm_stats : dict or None
        If provided, applies z-score normalization using 'mean' and 'std'.
    """

    def __init__(self, player_data, player_labels, norm_stats=None):
        n_players, n_engagements, n_ticks, n_features = player_data.shape

        # Flatten: (N_players, 30, 192, 5) → (N_players*30, 192, 5)
        self.engagements = player_data.reshape(-1, n_ticks, n_features)

        # Repeat each player's label 30 times (one per engagement)
        self.labels = np.repeat(player_labels, n_engagements)

        # Store player index for each engagement (for player-level aggregation)
        self.player_indices = np.repeat(np.arange(n_players), n_engagements)

        # Apply normalization
        if norm_stats is not None:
            mean = norm_stats["mean"].reshape(1, 1, -1)  # (1, 1, 5)
            std = norm_stats["std"].reshape(1, 1, -1)
            self.engagements = (self.engagements - mean) / std

        # Convert to float32 tensors
        self.engagements = torch.tensor(self.engagements, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        print(f"  EngagementDataset: {len(self)} samples  "
              f"({(self.labels == 1).sum().item()} cheater, "
              f"{(self.labels == 0).sum().item()} legit)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns
        -------
        engagement : torch.Tensor, shape (192, 5)
        label : torch.Tensor, scalar (0.0 or 1.0)
        """
        return self.engagements[idx], self.labels[idx]

    def get_player_indices(self):
        """Return the player index for each engagement (for aggregation)."""
        return self.player_indices


# ─── DataLoader factory ──────────────────────────────────────────────────────

def get_dataloaders(data_dir="data/raw", batch_size=256, seed=42,
                    val_size=0.15, test_size=0.15, num_workers=0):
    """
    Complete data pipeline: load → split → normalize → DataLoaders.

    Uses WeightedRandomSampler on the training set to handle class imbalance,
    so the model sees roughly equal numbers of cheater/legit engagements per epoch.

    Parameters
    ----------
    data_dir : str — path to directory containing cheaters.npy and legit.npy
    batch_size : int — batch size for all DataLoaders
    seed : int — random seed for reproducibility
    val_size : float — fraction of players for validation
    test_size : float — fraction of players for test
    num_workers : int — DataLoader workers (0 = main process)

    Returns
    -------
    loaders : dict with 'train', 'val', 'test' DataLoaders
    norm_stats : dict with 'mean' and 'std'
    datasets : dict with 'train', 'val', 'test' EngagementDataset objects
    """
    # 1. Load raw data
    print("=" * 60)
    print("Loading raw data...")
    data, labels = load_raw_data(data_dir)

    # 2. Split by player
    print("\nSplitting by player (stratified)...")
    splits = split_by_player(data, labels, val_size, test_size, seed)

    # 3. Compute normalization from training data only
    print("\nComputing normalization stats from training set...")
    norm_stats = compute_normalization_stats(splits["train"]["data"])

    # 4. Create datasets
    print("\nCreating PyTorch datasets...")
    datasets = {}
    for name in ["train", "val", "test"]:
        print(f"  {name}:")
        datasets[name] = EngagementDataset(
            splits[name]["data"],
            splits[name]["labels"],
            norm_stats=norm_stats,
        )

    # 5. Create DataLoaders
    # Training: use WeightedRandomSampler to balance classes
    train_labels = datasets["train"].labels.numpy()
    class_counts = np.bincount(train_labels.astype(int))
    # Weight = 1 / class_count  →  minority class gets higher weight
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(datasets["train"]),
        replacement=True,
    )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,           # Balanced sampling
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,            # Drop incomplete last batch for stable training
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    print(f"\nDataLoaders ready:")
    for name, loader in loaders.items():
        print(f"  {name:5s}: {len(loader):4d} batches  "
              f"(batch_size={batch_size})")
    print("=" * 60)

    return loaders, norm_stats, datasets
