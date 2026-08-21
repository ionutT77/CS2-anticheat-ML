"""Evaluation metrics and plotting for CS2 anti-cheat models."""
from .metrics import (
    compute_metrics,
    compute_player_verdicts,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_training_history,
    full_evaluation_report,
)
