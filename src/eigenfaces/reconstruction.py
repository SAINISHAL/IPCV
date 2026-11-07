"""Face reconstruction and residual error computation module."""
import numpy as np
from typing import Optional, Tuple
from src.eigenfaces.pca_from_scratch import PCAFromScratch


class FaceReconstructor:
    """Reconstructs faces from eigenface projections and analyzes residual errors."""

    def __init__(self, pca_model: PCAFromScratch):
        self.pca_model = pca_model

    def reconstruct(self, weights: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Reconstruct original image vector from weights."""
        return self.pca_model.reconstruct(weights, k=k)

    def compute_reconstruction_error(self, original_vector: np.ndarray,
                                     k: Optional[int] = None) -> Tuple[np.ndarray, float, float]:
        """Compute reconstruction, Mean Squared Error (MSE), and Euclidean residual distance.
        
        Returns:
            Tuple of (reconstructed_vector, mse, euclidean_distance).
        """
        weights = self.pca_model.project(original_vector, k=k)
        reconstructed = self.pca_model.reconstruct(weights, k=k)
        diff = original_vector - reconstructed
        mse = float(np.mean(diff ** 2))
        euclidean_dist = float(np.linalg.norm(diff))
        return reconstructed, mse, euclidean_dist

