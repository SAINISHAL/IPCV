"""Confusion Matrix generation and visualization module."""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.logger import setup_logger

logger = setup_logger("ConfusionMatrixVisualizer")


class ConfusionMatrixVisualizer:
    """Generates and plots labeled confusion matrices."""

    @staticmethod
    def plot_matrix(y_true: List[str], y_pred: List[str],
                    classes: Optional[List[str]] = None,
                    title: str = "Confusion Matrix",
                    save_path: Optional[str or Path] = None) -> plt.Figure:
        """Create and optionally save a normalized heatmap confusion matrix."""
        if classes is None:
            classes = sorted(list(set(y_true) | set(y_pred)))
            
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        # Normalize by true row sums
        with np.errstate(all="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

        plt.figure(figsize=(max(8, len(classes) * 0.7), max(6, len(classes) * 0.6)))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, cbar=True)
        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Identity", fontsize=10)
        plt.ylabel("True Identity", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=200)
            plt.close()
            logger.info(f"Confusion matrix saved to {p}")
            
        return plt.gcf()

