"""
Run this script to generate notebooks/02_lstm_training.ipynb

Usage:
    cd notebooks
    python generate_02_notebook.py
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python (CS2 Anti-Cheat)",
    "language": "python",
    "name": "cs2-anticheat",
}

cells = []

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)


# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""# 02 — LSTM Aimbot Detector: Training & Evaluation

**Goal:** Train the 2-layer LSTM model on the Kaggle CSGO Cheating Dataset and evaluate its ability to detect aimbot/triggerbot behavior.

**Architecture (from Journal Entry 3):**
```
Input (192 ticks × 5 features)
    → LSTM Layer 1 (128 hidden) — low-level temporal patterns
    → Dropout (0.3)
    → LSTM Layer 2 (64 hidden) — higher-order patterns
    → Last hidden state → (64,)
    → FC (64 → 32) + ReLU + Dropout(0.3)
    → FC (32 → 1) + Sigmoid → P(cheater)
```

**Key decisions (from Journal Entry 9):**
1. **Engagement-level** classification — each (192, 5) engagement is one sample
2. **Player-level splitting** — no data leakage between train/val/test
3. **Global z-score normalization** — on 4 continuous features, from training stats only
4. **Player verdict** — a player is flagged only if multiple engagements are flagged (not just one)
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Setup
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("## 1. Imports & Setup"))
cells.append(code("""\
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path so we can import our src modules
sys.path.insert(0, os.path.abspath('..'))

from src.data import load_raw_data, split_by_player, EngagementDataset, compute_normalization_stats, get_dataloaders
from src.models import LSTMAimbotDetector
from src.training import Trainer
from src.evaluation import (
    compute_metrics, compute_player_verdicts,
    plot_roc_curve, plot_precision_recall_curve,
    plot_confusion_matrix, plot_training_history,
    full_evaluation_report,
)

plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_theme(style='darkgrid')

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load & Split Data
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 2. Load Data & Create Splits

We split by **player** (not by engagement) to prevent data leakage.
All 30 engagements from a single player stay in the same split.

- **Train:** 70% of players → used for gradient updates
- **Validation:** 15% → used for early stopping & hyperparameter tuning
- **Test:** 15% → held out completely, evaluated only once at the end
"""))

cells.append(code("""\
# This single function handles the entire pipeline:
# load → split by player → normalize → create PyTorch datasets → DataLoaders
loaders, norm_stats, datasets = get_dataloaders(
    data_dir='../data/raw',
    batch_size=256,
    seed=SEED,
    num_workers=0,  # Windows compatibility
)

# Quick sanity check: grab one batch
batch_x, batch_y = next(iter(loaders['train']))
print(f"\\nSample batch:")
print(f"  Input shape:  {batch_x.shape}  (batch, ticks, features)")
print(f"  Labels shape: {batch_y.shape}")
print(f"  Label distribution in batch: "
      f"{(batch_y==1).sum().item()} cheater, {(batch_y==0).sum().item()} legit")
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Initialize Model
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 3. Initialize the LSTM Model

The model architecture was designed in Journal Entry 3.
Let's instantiate it and verify the parameter count.
"""))

cells.append(code("""\
model = LSTMAimbotDetector(
    input_size=5,       # 5 features per tick
    hidden_size_1=128,  # LSTM Layer 1
    hidden_size_2=64,   # LSTM Layer 2
    fc_size=32,         # FC hidden layer
    dropout=0.3,
)

# Print model summary
model.summary()

# Verify forward pass shape
dummy = torch.randn(4, 192, 5)  # 4 samples, 192 ticks, 5 features
out = model(dummy)
print(f"\\nForward pass check:")
print(f"  Input:  {dummy.shape}")
print(f"  Output: {out.shape} → {out.squeeze().tolist()}")
print(f"  Values in [0,1]: {(out >= 0).all() and (out <= 1).all()}")
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 4. Training
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 4. Train the Model

**Training configuration:**
- **Loss:** BCE with class weighting (pos_weight=5.0 for the 5:1 imbalance)
- **Optimizer:** Adam, lr=1e-3
- **LR Scheduler:** ReduceLROnPlateau (halve LR when val loss stalls for 5 epochs)
- **Early Stopping:** patience=10 epochs
- **Gradient Clipping:** max_norm=1.0 (prevents exploding gradients in LSTMs)

The `WeightedRandomSampler` in the DataLoader already balances sampling,
and `pos_weight=5.0` additionally weights the loss for cheater samples.
This dual approach ensures the model doesn't simply predict "legit" for everything.

⏱️ **Expected time:** ~2-5 minutes per epoch on CPU (depends on hardware).
Training will run up to 50 epochs but will stop early if validation loss plateaus.
"""))

cells.append(code("""\
trainer = Trainer(
    model=model,
    device=DEVICE,
    lr=1e-3,
    pos_weight=5.0,     # 5:1 class imbalance compensation
    patience=10,        # Stop if no improvement for 10 epochs
    checkpoint_dir='../models',
)

# Train!
history = trainer.train(
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    n_epochs=50,
)
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Training Curves
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 5. Training Curves

Let's visualize how training progressed. We're looking for:
- **Loss:** Train and val should decrease together. If train << val, we're overfitting.
- **Accuracy/F1/AUC:** Should improve and plateau. Val metrics shouldn't be far below train.
"""))

cells.append(code("""\
plot_training_history(history, save_path='../data/processed/10_training_history.png')
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 6. Test Set Evaluation
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 6. Test Set Evaluation

Now we evaluate on the **held-out test set** (15% of players, never seen during training).

This section reports:
1. **Engagement-level metrics** — how well the model classifies individual engagements
2. **Player-level verdicts** — a player is flagged ONLY if multiple engagements are flagged
   (we test different thresholds: 1, 2, 3, 5, 8, 10 out of 30 engagements)
3. **FPR @ 95% Recall** — the critical anti-cheat metric:
   "How many legit players get wrongly flagged to catch 95% of cheaters?"
"""))

cells.append(code("""\
report = full_evaluation_report(
    model=model,
    test_loader=loaders['test'],
    test_dataset=datasets['test'],
    device=DEVICE,
    save_dir='../data/processed',
    min_flagged_values=[1, 2, 3, 5, 8, 10, 15],
)
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Misclassification Analysis
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 7. Misclassification Analysis

Let's look at the model's mistakes:
- **False Positives:** Legit players flagged as cheaters — are they high-skill players?
- **False Negatives:** Cheaters that slipped through — are they subtle cheaters?

Understanding these failures helps improve the model and set better thresholds.
"""))

cells.append(code("""\
all_probs = report['all_probs']
all_labels = report['all_labels']
all_preds = (all_probs >= 0.5).astype(int)

# False positives: legit players predicted as cheater
fp_mask = (all_labels == 0) & (all_preds == 1)
fp_probs = all_probs[fp_mask]

# False negatives: cheaters predicted as legit
fn_mask = (all_labels == 1) & (all_preds == 0)
fn_probs = all_probs[fn_mask]

print(f"False Positives: {fp_mask.sum()} engagements")
print(f"  Avg P(cheater): {fp_probs.mean():.3f}" if len(fp_probs) > 0 else "  None!")
print(f"False Negatives: {fn_mask.sum()} engagements")
print(f"  Avg P(cheater): {fn_probs.mean():.3f}" if len(fn_probs) > 0 else "  None!")

# Distribution of predicted probabilities
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Predicted Probability Distribution by True Label', fontsize=13, fontweight='bold')

legit_probs = all_probs[all_labels == 0]
cheater_probs = all_probs[all_labels == 1]

axes[0].hist(legit_probs, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].set_title(f'Legit Players (n={len(legit_probs)})')
axes[0].set_xlabel('P(cheater)')
axes[0].set_ylabel('Count')
axes[0].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[0].legend()

axes[1].hist(cheater_probs, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[1].set_title(f'Cheaters (n={len(cheater_probs)})')
axes[1].set_xlabel('P(cheater)')
axes[1].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig('../data/processed/11_probability_distribution.png', dpi=120)
plt.show()
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 8. Player-Level Verdict Analysis
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 8. Player-Level Verdict Deep Dive

Since we classify at the engagement level, let's see how predictions aggregate per player.

For each player, we have 30 engagement predictions. Let's plot:
- How many of the 30 engagements are flagged for cheaters vs legit players
- This helps us choose the right `min_flagged` threshold
"""))

cells.append(code("""\
player_indices = datasets['test'].get_player_indices()

# Compute flagged count per player
unique_players = np.unique(player_indices)
flagged_counts_cheater = []
flagged_counts_legit = []

for pid in unique_players:
    mask = player_indices == pid
    probs = all_probs[mask]
    label = all_labels[mask][0]  # Same for all engagements
    
    flagged = (probs >= 0.5).sum()
    if label == 1:
        flagged_counts_cheater.append(flagged)
    else:
        flagged_counts_legit.append(flagged)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Number of Flagged Engagements per Player (out of 30)', fontsize=13, fontweight='bold')

axes[0].hist(flagged_counts_legit, bins=range(0, 32), color='#2ecc71', 
             alpha=0.7, edgecolor='black', linewidth=0.5, align='left')
axes[0].set_title(f'Legit Players (n={len(flagged_counts_legit)})')
axes[0].set_xlabel('# Flagged Engagements (out of 30)')
axes[0].set_ylabel('# Players')
axes[0].axvline(3, color='red', linestyle='--', label='min_flagged=3')
axes[0].legend()

axes[1].hist(flagged_counts_cheater, bins=range(0, 32), color='#e74c3c',
             alpha=0.7, edgecolor='black', linewidth=0.5, align='left')
axes[1].set_title(f'Cheaters (n={len(flagged_counts_cheater)})')
axes[1].set_xlabel('# Flagged Engagements (out of 30)')
axes[1].axvline(3, color='red', linestyle='--', label='min_flagged=3')
axes[1].legend()

plt.tight_layout()
plt.savefig('../data/processed/12_player_flagged_distribution.png', dpi=120)
plt.show()

print(f"\\nLegit players: median flagged = {np.median(flagged_counts_legit):.0f}/30")
print(f"Cheaters:      median flagged = {np.median(flagged_counts_cheater):.0f}/30")
"""))


# ══════════════════════════════════════════════════════════════════════════════
# 9. Save Model & Summary
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 9. Summary & Next Steps

### Results at a Glance
Review the metrics above and note:
1. **Engagement-level accuracy** — how well does the LSTM classify individual engagements?
2. **Player-level verdict accuracy** — with `min_flagged=3`, how well do we identify cheaters?
3. **FPR @ 95% recall** — how many legit players would be wrongly flagged?

### Next Steps
- **Threshold tuning:** Adjust `min_flagged` based on the histogram distributions above
- **Feature engineering:** Add derived features (aim speed, acceleration, jerk)
- **Model comparison:** Train a Random Forest baseline on aggregated features
- **Phase 2:** Transformer architecture (AntiCheatPT-style)
"""))

cells.append(code("""\
# Save the final model info
print("Model saved to: ../models/best_lstm.pt")
print(f"\\nFinal test metrics:")
for key, val in report['engagement_metrics'].items():
    if isinstance(val, float):
        print(f"  {key:20s}: {val:.4f}")
    else:
        print(f"  {key:20s}: {val}")

print(f"\\nFPR @ 95% Recall: {report['fpr_at_95_recall']:.4f}")
print(f"  → {report['fpr_at_95_recall']*100:.2f}% of legit players would be wrongly flagged")
print(f"  → to catch 95% of cheaters")
"""))


# ══════════════════════════════════════════════════════════════════════════════
# Save notebook
# ══════════════════════════════════════════════════════════════════════════════
nb.cells = cells

out_path = os.path.join(os.path.dirname(__file__), "02_lstm_training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook written to: {out_path}")
