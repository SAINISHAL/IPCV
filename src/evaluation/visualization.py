"""Evaluation and PCA Visualizations module."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from src.eigenfaces.pca_from_scratch import PCAFromScratch
from src.utils.logger import setup_logger

logger = setup_logger("EvaluationVisualizer")


class EvaluationVisualizer:
    """Produces diagnostic plots for PCA explained variance, K-experiments, and reconstructions."""

    @staticmethod
    def plot_variance_curves(pca: PCAFromScratch, save_path: Optional[str or Path] = None) -> plt.Figure:
        """Plot explained and cumulative variance ratio curves."""
        if pca.explained_variance_ratio is None:
            raise ValueError("PCA model has no variance ratios.")

        exp = pca.explained_variance_ratio * 100.0
        cum = pca.cumulative_variance_ratio * 100.0
        k_range = np.arange(1, len(exp) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Individual variance
        ax1.plot(k_range[:50], exp[:50], "o-", color="#1f77b4", linewidth=2, markersize=4)
        ax1.set_title("Explained Variance per Eigenface (Scree Plot)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Principal Component Index", fontsize=10)
        ax1.set_ylabel("Explained Variance (%)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Cumulative variance
        ax2.plot(k_range[:50], cum[:50], "s-", color="#2ca02c", linewidth=2, markersize=4)
        ax2.axhline(y=90.0, color="red", linestyle="--", alpha=0.7, label="90% Variance Threshold")
        ax2.axhline(y=95.0, color="orange", linestyle="--", alpha=0.7, label="95% Variance Threshold")
        ax2.set_title("Cumulative Explained Variance", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Number of Components (K)", fontsize=10)
        ax2.set_ylabel("Cumulative Variance (%)", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=200)
            plt.close()
            logger.info(f"Variance plot saved to {p}")
        return fig

    @staticmethod
    def plot_k_experiment_comparison(k_values: List[int],
                                     results_dict: Dict[str, List[float]],
                                     metric_name: str = "Accuracy",
                                     save_path: Optional[str or Path] = None) -> plt.Figure:
        """Plot model performance curves across varying K components."""
        plt.figure(figsize=(9, 5))
        styles = {"Euclidean": ("o--", "#1f77b4"), "KNN": ("s--", "#ff7f0e"), "SVM": ("^--", "#2ca02c")}

        for model_name, values in results_dict.items():
            fmt, color = styles.get(model_name, ("x--", "black"))
            plt.plot(k_values, values, fmt, label=f"Model: {model_name}", linewidth=2, markersize=6, color=color)

        plt.title(f"Model Comparison Across K Components ({metric_name})", fontsize=12, fontweight="bold")
        plt.xlabel("Number of Eigenfaces (K)", fontsize=10)
        plt.ylabel(metric_name, fontsize=10)
        plt.xticks(k_values)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=200)
            plt.close()
        return plt.gcf()

    @staticmethod
    def plot_face_reconstructions(original_face_vec: np.ndarray,
                                  pca: PCAFromScratch,
                                  target_shape: tuple = (100, 100),
                                  k_steps: List[int] = [5, 10, 20, 30, 40],
                                  save_path: Optional[str or Path] = None) -> plt.Figure:
        """Plot original face alongside reconstructions across varying K components."""
        w, h = target_shape
        n_plots = len(k_steps) + 1
        fig, axes = plt.subplots(1, n_plots, figsize=(2.8 * n_plots, 3.2))

        # Original
        orig_img = original_face_vec.reshape((h, w))
        axes[0].imshow(orig_img, cmap="gray")
        axes[0].set_title("Original Face", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        # Reconstructions
        for idx, k in enumerate(k_steps):
            w_k = pca.project(original_face_vec, k=k)
            rec_vec = pca.reconstruct(w_k, k=k)
            rec_img = rec_vec.reshape((h, w))
            mse = np.mean((original_face_vec - rec_vec) ** 2)

            ax = axes[idx + 1]
            ax.imshow(rec_img, cmap="gray")
            ax.set_title(f"K = {k}\nMSE: {mse:.4f}", fontsize=9)
            ax.axis("off")

        plt.suptitle("Eigenface Reconstruction Quality vs. Subspace Dimensionality (K)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=200)
            plt.close()
        return fig

