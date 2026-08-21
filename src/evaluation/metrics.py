"""
Evaluation metrics and visualization for CS2 anti-cheat models.

Includes:
    - Standard classification metrics (accuracy, precision, recall, F1, ROC AUC)
    - Anti-cheat-specific: False Positive Rate at 95% recall
    - Player-level aggregation: convert engagement predictions -> player verdict
    - Publication-quality plots for thesis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true : np.ndarray — true labels (0 or 1)
    y_prob : np.ndarray — predicted probabilities
    threshold : float — classification threshold

    Returns
    -------
    metrics : dict
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
                   if len(np.unique(y_true)) > 1 else 0.0,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fpr": fp / max(fp + tn, 1),  # False Positive Rate
        "fnr": fn / max(fn + tp, 1),  # False Negative Rate
        "threshold": threshold,
    }
    return metrics


def fpr_at_recall(y_true, y_prob, target_recall=0.95):
    """
    Find the False Positive Rate when recall reaches a target value.

    This is THE critical anti-cheat metric:
    "How many legit players do we wrongly ban to catch 95% of cheaters?"

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray
    target_recall : float

    Returns
    -------
    fpr : float — false positive rate at the target recall
    threshold : float — the threshold that achieves this recall
    """
    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_prob)

    # tpr = recall. Find the point where recall >= target_recall
    # with the lowest FPR
    valid = tpr_vals >= target_recall
    if not valid.any():
        return 1.0, 0.0  # Can't achieve target recall

    # Among all points with sufficient recall, pick the lowest FPR
    best_idx = np.where(valid)[0][np.argmin(fpr_vals[valid])]
    best_fpr = fpr_vals[best_idx]

    # thresholds array is one shorter than fpr/tpr arrays
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1]

    return best_fpr, best_threshold


# ─── Player-level aggregation ────────────────────────────────────────────────

def compute_player_verdicts(
    engagement_probs,
    player_indices,
    player_labels,
    min_flagged=3,
    engagement_threshold=0.5,
):
    """
    Aggregate engagement-level predictions to player-level verdicts.

    A player is ONLY classified as a cheater if at least `min_flagged`
    of their 30 engagements are individually flagged as cheating.
    This prevents banning a player based on a single lucky/suspicious shot.

    Parameters
    ----------
    engagement_probs : np.ndarray — P(cheater) for each engagement
    player_indices : np.ndarray — which player each engagement belongs to
    player_labels : np.ndarray — true label per engagement (all same per player)
    min_flagged : int — minimum number of flagged engagements for a player verdict
    engagement_threshold : float — threshold for individual engagement classification

    Returns
    -------
    results : dict
        player_verdicts: 1/0 per player
        player_true_labels: true label per player
        player_flagged_counts: how many of 30 engagements were flagged
        player_mean_probs: mean P(cheater) across all engagements
        metrics: player-level classification metrics
    """
    unique_players = np.unique(player_indices)
    n_players = len(unique_players)

    player_verdicts = np.zeros(n_players, dtype=int)
    player_true_labels = np.zeros(n_players, dtype=int)
    player_flagged_counts = np.zeros(n_players, dtype=int)
    player_mean_probs = np.zeros(n_players, dtype=float)

    for i, pid in enumerate(unique_players):
        mask = player_indices == pid
        probs = engagement_probs[mask]
        labels = player_labels[mask]

        # True label (same for all engagements of this player)
        player_true_labels[i] = int(labels[0])

        # Count how many engagements are flagged
        flagged = (probs >= engagement_threshold).sum()
        player_flagged_counts[i] = flagged

        # Mean probability across all engagements
        player_mean_probs[i] = probs.mean()

        # Verdict: cheater ONLY if enough engagements are flagged
        player_verdicts[i] = 1 if flagged >= min_flagged else 0

    # Compute player-level metrics
    metrics = compute_metrics(player_true_labels, player_mean_probs,
                              threshold=0.5)
    # Override predictions with the multi-engagement verdict
    verdict_metrics = {
        "accuracy": accuracy_score(player_true_labels, player_verdicts),
        "precision": precision_score(player_true_labels, player_verdicts,
                                     zero_division=0),
        "recall": recall_score(player_true_labels, player_verdicts,
                               zero_division=0),
        "f1": f1_score(player_true_labels, player_verdicts, zero_division=0),
        "min_flagged_threshold": min_flagged,
    }

    results = {
        "player_verdicts": player_verdicts,
        "player_true_labels": player_true_labels,
        "player_flagged_counts": player_flagged_counts,
        "player_mean_probs": player_mean_probs,
        "engagement_metrics": metrics,
        "verdict_metrics": verdict_metrics,
    }

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve with AUC score."""
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_vals, tpr_vals, color="#3498db", linewidth=2,
            label=f"LSTM (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5,
            label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Engagement-Level Classification", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved ROC curve to {save_path}")
    plt.show()
    return fig


def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """Plot Precision-Recall curve."""
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_vals, prec_vals, color="#e74c3c", linewidth=2, label="LSTM")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved PR curve to {save_path}")
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """Plot confusion matrix heatmap."""
    if labels is None:
        labels = ["Legit", "Cheater"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()
    return fig


def plot_training_history(history, save_path=None):
    """Plot training curves: loss, accuracy, F1, ROC AUC over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train", color="#e74c3c")
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation", color="#3498db")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(epochs, history["val_accuracy"], label="Val Accuracy",
                    color="#2ecc71")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].legend()

    # F1
    axes[1, 0].plot(epochs, history["val_f1"], label="Val F1", color="#9b59b6")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("Validation F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    # ROC AUC
    axes[1, 1].plot(epochs, history["val_roc_auc"], label="Val ROC AUC",
                    color="#f39c12")
    axes[1, 1].set_ylabel("ROC AUC")
    axes[1, 1].set_title("Validation ROC AUC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved training history to {save_path}")
    plt.show()
    return fig


# ─── Full evaluation report ──────────────────────────────────────────────────

def full_evaluation_report(
    model,
    test_loader,
    test_dataset,
    device,
    save_dir="data/processed",
    min_flagged_values=None,
):
    """
    Run complete evaluation: engagement-level + player-level + plots.

    Parameters
    ----------
    model : nn.Module — trained model
    test_loader : DataLoader
    test_dataset : EngagementDataset — for player index access
    device : torch.device
    save_dir : str — directory to save plots
    min_flagged_values : list of int — test different thresholds for player verdict
                         (default: [1, 2, 3, 5, 8, 10])

    Returns
    -------
    report : dict — all metrics and results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    if min_flagged_values is None:
        min_flagged_values = [1, 2, 3, 5, 8, 10]

    model.eval()
    model.to(device)

    # ── Collect all predictions ────────────────────────────────────────
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            probs = model(batch_x).squeeze(-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch_y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    # ── Engagement-level metrics ──────────────────────────────────────
    print("=" * 70)
    print("ENGAGEMENT-LEVEL EVALUATION")
    print("=" * 70)

    eng_metrics = compute_metrics(all_labels, all_probs)
    for key, val in eng_metrics.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.4f}")
        else:
            print(f"  {key:20s}: {val}")

    # FPR at 95% recall
    fpr_95, thresh_95 = fpr_at_recall(all_labels, all_probs, target_recall=0.95)
    print(f"\n  FPR @ 95% Recall:    {fpr_95:.4f} (threshold={thresh_95:.3f})")
    print(f"  -> At 95% cheat catch rate, {fpr_95*100:.2f}% of legit players "
          f"would be falsely flagged")

    # Classification report
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Legit', 'Cheater'])}")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_roc_curve(all_labels, all_probs,
                   save_path=os.path.join(save_dir, "07_roc_curve.png"))
    plot_precision_recall_curve(all_labels, all_probs,
                                save_path=os.path.join(save_dir, "08_pr_curve.png"))
    plot_confusion_matrix(all_labels, all_preds,
                          save_path=os.path.join(save_dir, "09_confusion_matrix.png"))

    # ── Player-level aggregation ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("PLAYER-LEVEL EVALUATION")
    print("=" * 70)
    print(f"{'min_flagged':>12s} {'Accuracy':>10s} {'Precision':>10s} "
          f"{'Recall':>10s} {'F1':>10s}")
    print("-" * 60)

    player_indices = test_dataset.get_player_indices()
    player_results = {}

    for min_f in min_flagged_values:
        results = compute_player_verdicts(
            all_probs, player_indices, all_labels,
            min_flagged=min_f, engagement_threshold=0.5,
        )
        vm = results["verdict_metrics"]
        print(f"{min_f:>12d} {vm['accuracy']:>10.3f} {vm['precision']:>10.3f} "
              f"{vm['recall']:>10.3f} {vm['f1']:>10.3f}")
        player_results[min_f] = results

    print(f"\nNote: 'min_flagged' = minimum number of the 30 engagements that")
    print(f"must be classified as cheating before the player is flagged.")
    print(f"Higher values = fewer false positives but more missed cheaters.")

    # Build report
    report = {
        "engagement_metrics": eng_metrics,
        "fpr_at_95_recall": fpr_95,
        "threshold_at_95_recall": thresh_95,
        "all_probs": all_probs,
        "all_labels": all_labels,
        "player_results": player_results,
    }

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

    return report
