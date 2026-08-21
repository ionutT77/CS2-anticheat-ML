"""
Training loop for CS2 anti-cheat LSTM detector.

Features:
    - BCE loss with class weighting for 5:1 imbalance
    - Adam optimizer with ReduceLROnPlateau scheduler
    - Early stopping on validation loss
    - Best model checkpointing
    - Comprehensive metric logging per epoch
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
)


class Trainer:
    """
    Handles the full training lifecycle for the LSTM detector.

    Parameters
    ----------
    model : nn.Module
        The LSTM model to train.
    device : torch.device
        Device to train on (cpu or cuda).
    lr : float
        Initial learning rate (default: 1e-3).
    pos_weight : float or None
        Weight for the positive (cheater) class in BCE loss.
        If None, auto-computed from the training data.
        The 5:1 imbalance means cheaters should get ~5× weight.
    patience : int
        Number of epochs to wait for improvement before early stopping.
    checkpoint_dir : str
        Directory to save model checkpoints.
    """

    def __init__(
        self,
        model,
        device=None,
        lr=1e-3,
        pos_weight=None,
        patience=10,
        checkpoint_dir="models",
    ):
        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)

        # Loss function: BCEWithLogitsLoss is more numerically stable than
        # BCE + Sigmoid, but our model already has Sigmoid in it.
        # So we use plain BCELoss with manual weight compensation.
        #
        # However, for class weighting, BCEWithLogitsLoss is cleaner.
        # We'll remove Sigmoid from forward pass and use BCEWithLogitsLoss.
        # Actually — to keep the model architecture clean and as documented,
        # we use BCE loss and apply class weighting via sample weights.
        #
        # Decision: Use BCELoss. The WeightedRandomSampler in the DataLoader
        # already balances the class frequencies. We add pos_weight as an
        # additional safety net via manual weighting in the loss computation.
        self.pos_weight = pos_weight
        self.criterion = nn.BCELoss(reduction="none")

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # LR scheduler: reduce by half when val loss plateaus for 5 epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # Early stopping
        self.patience = patience
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_epoch = 0

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(checkpoint_dir, "best_lstm.pt")

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_roc_auc": [],
            "lr": [],
            "epoch_time": [],
        }

    def _compute_weighted_loss(self, predictions, targets):
        """
        Compute BCE loss with optional class weighting.

        If pos_weight is set, positive (cheater) samples get higher loss weight
        to compensate for class imbalance.
        """
        loss_per_sample = self.criterion(predictions, targets)

        if self.pos_weight is not None:
            # Create weight tensor: 1.0 for negatives, pos_weight for positives
            weights = torch.where(
                targets == 1.0,
                torch.tensor(self.pos_weight, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            loss_per_sample = loss_per_sample * weights

        return loss_per_sample.mean()

    def train_one_epoch(self, train_loader):
        """Train for one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            predictions = self.model(batch_x).squeeze(-1)  # (batch,)
            loss = self._compute_weighted_loss(predictions, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTMs
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Evaluate model on a DataLoader.

        Returns
        -------
        metrics : dict
            loss, accuracy, precision, recall, f1, roc_auc
        all_probs : np.ndarray
            Raw predicted probabilities for each sample
        all_labels : np.ndarray
            True labels for each sample
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x).squeeze(-1)
            loss = self._compute_weighted_loss(predictions, batch_y)

            total_loss += loss.item()
            n_batches += 1

            all_probs.append(predictions.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        all_preds = (all_probs >= 0.5).astype(int)

        metrics = {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "roc_auc": roc_auc_score(all_labels, all_probs)
                       if len(np.unique(all_labels)) > 1 else 0.0,
        }

        return metrics, all_probs, all_labels

    def train(self, train_loader, val_loader, n_epochs=50):
        """
        Full training loop with validation, early stopping, and checkpointing.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        n_epochs : int — maximum number of epochs

        Returns
        -------
        history : dict — training history (all metrics per epoch)
        """
        print("=" * 70)
        print(f"Training on {self.device} | Max epochs: {n_epochs} | "
              f"Patience: {self.patience}")
        print(f"Pos weight: {self.pos_weight} | "
              f"LR: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 70)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            # ── Train ──────────────────────────────────────────────────
            train_loss = self.train_one_epoch(train_loader)

            # ── Validate ───────────────────────────────────────────────
            val_metrics, _, _ = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]

            # ── LR scheduler step ──────────────────────────────────────
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Record history ─────────────────────────────────────────
            elapsed = time.time() - t0
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_roc_auc"].append(val_metrics["roc_auc"])
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(elapsed)

            # ── Print epoch summary ────────────────────────────────────
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.3f} | "
                f"Val F1: {val_metrics['f1']:.3f} | "
                f"Val AUC: {val_metrics['roc_auc']:.3f} | "
                f"LR: {current_lr:.1e} | "
                f"{elapsed:.1f}s"
            )

            # ── Early stopping / checkpointing ────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0

                # Save best model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "model_config": self.model.config,
                    "history": self.history,
                }, self.best_model_path)
                print(f"  ^ New best model saved (val_loss={val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"\n[STOP] Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    print(f"   Best epoch: {self.best_epoch} "
                          f"(val_loss={self.best_val_loss:.4f})")
                    break

        # Load best model weights
        print(f"\nLoading best model from epoch {self.best_epoch}...")
        checkpoint = torch.load(self.best_model_path, map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        print("Training complete!")
        return self.history

    def load_best_model(self):
        """Load the best saved model checkpoint."""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(
                self.best_model_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch']} "
                  f"(val_loss={checkpoint['val_loss']:.4f})")
            return checkpoint
        else:
            print(f"No checkpoint found at {self.best_model_path}")
            return None
