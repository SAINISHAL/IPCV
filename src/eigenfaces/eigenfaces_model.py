"""Eigenfaces Model Wrapper supporting persistence and diagnostic visualizations."""
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from src.eigenfaces.pca_from_scratch import PCAFromScratch
from src.utils.logger import setup_logger

logger = setup_logger("EigenfacesModel")


class EigenfacesModel:
    """High-level Eigenfaces representation model."""

    def __init__(self, target_shape: Tuple[int, int] = (100, 100), n_components: Optional[int] = 25,
                 face_alignment: str = "eyes", preprocessing_mode: str = "clahe",
                 webcam_retry: bool = True):
        self.target_shape = target_shape  # (width, height)
        self.n_components = n_components
        self.face_alignment = face_alignment
        self.preprocessing_mode = preprocessing_mode
        self.webcam_retry = webcam_retry
        self.pca = PCAFromScratch(n_components=n_components)
        self.is_trained = False

    def train(self, X: np.ndarray) -> "EigenfacesModel":
        """Train PCA model on flattened images matrix of shape (N, D)."""
        self.pca.fit(X)
        self.is_trained = True
        return self

    def get_mean_face_image(self) -> np.ndarray:
        """Return mean face reshaped as 2D image (height, width)."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        w, h = self.target_shape
        return self.pca.mean_vector.reshape((h, w))

    def get_eigenface_image(self, index: int) -> np.ndarray:
        """Return the i-th eigenface reshaped as 2D image (height, width)."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        w, h = self.target_shape
        eig_vec = self.pca.eigenvectors[:, index]
        return eig_vec.reshape((h, w))

    def save(self, filepath: str or Path) -> Path:
        """Save trained Eigenfaces model to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "target_shape": self.target_shape,
            "n_components": self.n_components,
            "face_alignment": self.face_alignment,
            "preprocessing_mode": self.preprocessing_mode,
            "webcam_retry": self.webcam_retry,
            "mean_vector": self.pca.mean_vector,
            "eigenvalues": self.pca.eigenvalues,
            "eigenvectors": self.pca.eigenvectors,
            "explained_variance_ratio": self.pca.explained_variance_ratio,
            "cumulative_variance_ratio": self.pca.cumulative_variance_ratio,
            "n_samples": self.pca.n_samples,
            "n_features": self.pca.n_features
        }
        with open(p, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Eigenfaces model saved to {p}")
        return p

    def load(self, filepath: str or Path) -> "EigenfacesModel":
        """Load trained Eigenfaces model from disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Eigenfaces model file not found: {p}")
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.target_shape = data["target_shape"]
        self.n_components = data["n_components"]
        self.face_alignment = data.get("face_alignment", "eyes")
        self.preprocessing_mode = data.get("preprocessing_mode", "clahe")
        self.webcam_retry = data.get("webcam_retry", True)
        self.pca.n_components = self.n_components
        self.pca.mean_vector = data["mean_vector"]
        self.pca.eigenvalues = data["eigenvalues"]
        self.pca.eigenvectors = data["eigenvectors"]
        self.pca.explained_variance_ratio = data["explained_variance_ratio"]
        self.pca.cumulative_variance_ratio = data["cumulative_variance_ratio"]
        self.pca.n_samples = data["n_samples"]
        self.pca.n_features = data["n_features"]
        self.is_trained = True
        logger.info(f"Eigenfaces model loaded successfully from {p}")
        return self

    def plot_eigenfaces_grid(self, n_eigenfaces: int = 10, save_path: Optional[str or Path] = None) -> plt.Figure:
        """Generate figure showing Mean Face and the top N Eigenfaces."""
        if not self.is_trained:
            raise ValueError("Model is not trained.")
            
        n_show = min(n_eigenfaces, self.pca.eigenvectors.shape[1])
        cols = 5
        rows = int(np.ceil((n_show + 1) / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
        axes = axes.flatten()

        # Plot Mean Face
        mean_img = self.get_mean_face_image()
        axes[0].imshow(mean_img, cmap="gray")
        axes[0].set_title("Average Face (\\mu)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Plot Eigenfaces
        for i in range(n_show):
            ef_img = self.get_eigenface_image(i)
            ax = axes[i + 1]
            # Eigenfaces have both positive and negative values; seismic or coolwarm or gray works
            vmax = max(abs(ef_img.min()), abs(ef_img.max()))
            ax.imshow(ef_img, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            var_pct = self.pca.explained_variance_ratio[i] * 100.0 if self.pca.explained_variance_ratio is not None else 0.0
            ax.set_title(f"Eigenface #{i+1}\n({var_pct:.1f}% var)", fontsize=10)
            ax.axis("off")

        for j in range(n_show + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Principal Component Eigenfaces (Image-Space Basis)", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=200)
            plt.close()
            
        return fig

