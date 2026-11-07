"""Subspace projection module for Eigenfaces."""
import numpy as np
from typing import Optional
from src.eigenfaces.pca_from_scratch import PCAFromScratch


class SubspaceProjector:
    """Projects image vectors into the low-dimensional Eigenfaces subspace."""

    def __init__(self, pca_model: PCAFromScratch, k: Optional[int] = None):
        self.pca_model = pca_model
        self.k = k

    def project_single(self, face_vector: np.ndarray) -> np.ndarray:
        """Project a single face vector (D,) to subspace weight vector (K,)."""
        return self.pca_model.project(face_vector, k=self.k)

    def project_batch(self, face_matrix: np.ndarray) -> np.ndarray:
        """Project a batch of face vectors (N, D) to (N, K)."""
        return self.pca_model.project(face_matrix, k=self.k)

