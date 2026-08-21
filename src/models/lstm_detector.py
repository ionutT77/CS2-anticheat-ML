"""
LSTM-based Aimbot Detector for CS2 Anti-Cheat.

Architecture designed in PROJECT_JOURNAL.md Entry 3:
    Input (192 ticks × 5 features)
        → LSTM Layer 1 (128 hidden) — learns low-level patterns (single snaps, aim anomalies)
        → Dropout (0.3) — regularization
        → LSTM Layer 2 (64 hidden) — learns higher-order patterns (patterns of snaps, consistency)
        → Take last hidden state — compresses 192 ticks into a 64-dim summary vector
        → FC (64 → 32) + ReLU + Dropout(0.3) — decision features
        → FC (32 → 1) + Sigmoid — binary probability P(cheater)

Design decisions:
    - 2 LSTM layers (not 1 or 3): 1 is too shallow; 3+ has diminishing returns
    - 128→64 hidden dims (decreasing): forces compression/abstraction at higher layers
    - Dropout 0.3: standard regularization for the 12K player dataset
    - Sigmoid output: binary classification (cheater vs legit)
    - batch_first=True: easier tensor handling (batch, seq, features)
"""

import torch
import torch.nn as nn


class LSTMAimbotDetector(nn.Module):
    """
    Two-layer LSTM for engagement-level aimbot detection.

    Input shape:  (batch_size, seq_len=192, n_features=5)
    Output shape: (batch_size, 1) — P(cheater) in [0, 1]

    Parameters
    ----------
    input_size : int
        Number of input features per tick (default: 5).
    hidden_size_1 : int
        Hidden units in the first LSTM layer (default: 128).
    hidden_size_2 : int
        Hidden units in the second LSTM layer (default: 64).
    fc_size : int
        Size of the intermediate fully-connected layer (default: 32).
    dropout : float
        Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        input_size=5,
        hidden_size_1=128,
        hidden_size_2=64,
        fc_size=32,
        dropout=0.3,
    ):
        super().__init__()

        # ── LSTM backbone ─────────────────────────────────────────────────
        # Layer 1: learns low-level temporal patterns
        # Individual tick-level anomalies: single snaps, sudden aim changes,
        # fire-on-align correlations within a few ticks
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            num_layers=1,
        )

        # Dropout between LSTM layers
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2: learns higher-order temporal patterns
        # Patterns of patterns: repeated snaps, inhuman consistency across
        # an engagement, overall aiming "rhythm" that distinguishes human
        # from machine
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True,
            num_layers=1,
        )

        # ── Classification head ───────────────────────────────────────────
        # Takes the last hidden state (64-dim summary of the entire sequence)
        # and maps it to a cheater probability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_2, fc_size),   # 64 → 32
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, 1),                # 32 → 1
            nn.Sigmoid(),                          # → P(cheater) ∈ [0, 1]
        )

        # Store config for logging / reconstruction
        self.config = {
            "input_size": input_size,
            "hidden_size_1": hidden_size_1,
            "hidden_size_2": hidden_size_2,
            "fc_size": fc_size,
            "dropout": dropout,
        }

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)
            Batch of engagement sequences. Default: (batch, 192, 5)

        Returns
        -------
        probs : torch.Tensor, shape (batch, 1)
            Predicted probability of cheating for each engagement.
        """
        # LSTM Layer 1: process the full sequence
        # out1 shape: (batch, seq_len, hidden_size_1=128)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)

        # LSTM Layer 2: process the Layer 1 output sequence
        # out2 shape: (batch, seq_len, hidden_size_2=64)
        out2, _ = self.lstm2(out1)

        # Take only the LAST hidden state
        # This is the 64-dimensional summary of the entire 192-tick sequence
        # It captures the cumulative temporal patterns the LSTM learned
        last_hidden = out2[:, -1, :]  # (batch, 64)

        # Classification head: 64 → 32 → 1 → sigmoid
        probs = self.classifier(last_hidden)  # (batch, 1)

        return probs

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def summary(self):
        """Print a human-readable model summary."""
        total, trainable = self.count_parameters()
        print("=" * 60)
        print("LSTM Aimbot Detector -- Model Summary")
        print("=" * 60)
        print(f"Architecture:")
        print(f"  Input:   (batch, 192, {self.config['input_size']})")
        print(f"  LSTM 1:  {self.config['input_size']} -> {self.config['hidden_size_1']} hidden")
        print(f"  Dropout: {self.config['dropout']}")
        print(f"  LSTM 2:  {self.config['hidden_size_1']} -> {self.config['hidden_size_2']} hidden")
        print(f"  FC:      {self.config['hidden_size_2']} -> {self.config['fc_size']} -> 1")
        print(f"  Output:  Sigmoid -> P(cheater)")
        print(f"\nParameters:")
        print(f"  Total:     {total:>10,}")
        print(f"  Trainable: {trainable:>10,}")
        print("=" * 60)
        return total, trainable
