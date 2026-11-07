"""Mathematical PCA from Scratch using Pure NumPy (Dual Covariance Formulation)."""
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("PCAFromScratch")


class PCAFromScratch:
    """Principal Component Analysis implemented from first principles.
    
    Supports the Dual PCA approach (Sirovich and Kirby / Turk and Pentland)
    when feature dimension D >> number of samples N.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.mean_vector: Optional[np.ndarray] = None          # Shape (D,)
        self.eigenvalues: Optional[np.ndarray] = None          # Shape (min(N, D),)
        self.eigenvectors: Optional[np.ndarray] = None         # Shape (D, min(N, D))
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.cumulative_variance_ratio: Optional[np.ndarray] = None
        self.n_samples: int = 0
        self.n_features: int = 0

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        """Compute mean vector, covariance, sorted eigenvalues, and eigenvectors.
        
        Args:
            X: Training data matrix of shape (N, D), where N is samples, D is features.
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        self.n_samples = N
        self.n_features = D
        
        logger.info(f"Fitting PCA on {N} samples with {D} dimensions...")

        # 7.1 Mean Face
        # \mu = \frac{1}{N}\sum_{i=1}^{N} x_i
        self.mean_vector = np.mean(X, axis=0)

        # 7.2 Mean Centering
        # A_matrix: shape (D, N) where each column is (x_i - \mu)
        # Note: X_centered is (N, D), so A = X_centered.T is (D, N)
        X_centered = X - self.mean_vector
        A = X_centered.T  # (D, N)

        # 7.3 Efficient Dual Covariance Matrix
        # Standard covariance: C = (1/N) * A * A^T  (D x D, e.g. 10000 x 10000)
        # Dual covariance:     L = A^T * A           (N x N, e.g. 100 x 100)
        if D > N:
            logger.info(f"Using Dual PCA: computing {N}x{N} Gram matrix L = A^T * A...")
            L = np.dot(X_centered, X_centered.T) / float(N)  # (N, N)
            
            # 7.4 Eigenvalue Decomposition of L
            # L * v_i = \lambda_i * v_i
            # Since L is symmetric positive semi-definite, np.linalg.eigh is numerically optimal
            eig_vals, v = np.linalg.eigh(L)
            
            # Sort eigenvalues in strictly descending order
            idx = np.argsort(eig_vals)[::-1]
            eig_vals = eig_vals[idx]
            v = v[:, idx]
            
            # Filter near-zero or negative eigenvalues due to floating-point imprecision
            valid_mask = eig_vals > 1e-12
            if np.sum(valid_mask) == 0:
                valid_mask[0] = True
            eig_vals = eig_vals[valid_mask]
            v = v[:, valid_mask]
            
            # Transform smaller eigenvectors v into image-space eigenfaces:
            # u_i = A * v_i
            U = np.dot(A, v)  # (D, K_valid)
            
            # Normalize each eigenvector to unit Euclidean length: ||u_i||_2 = 1
            col_norms = np.linalg.norm(U, axis=0, keepdims=True)
            col_norms[col_norms == 0] = 1.0
            U = U / col_norms
            
        else:
            logger.info(f"Computing standard {D}x{D} covariance matrix...")
            C = np.dot(A, A.T) / float(N)
            eig_vals, U = np.linalg.eigh(C)
            idx = np.argsort(eig_vals)[::-1]
            eig_vals = eig_vals[idx]
            U = U[:, idx]
            valid_mask = eig_vals > 1e-12
            eig_vals = eig_vals[valid_mask]
            U = U[:, valid_mask]

        self.eigenvalues = eig_vals
        self.eigenvectors = U

        # Compute explained variance ratio
        total_variance = np.sum(eig_vals)
        if total_variance > 0:
            self.explained_variance_ratio = eig_vals / total_variance
            self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)
        else:
            self.explained_variance_ratio = np.zeros_like(eig_vals)
            self.cumulative_variance_ratio = np.zeros_like(eig_vals)

        logger.info(f"PCA fitted successfully. Extracted {len(self.eigenvalues)} components.")
        return self

    def get_components(self, k: Optional[int] = None) -> np.ndarray:
        """Retrieve top K eigenvectors (eigenfaces matrix of shape (D, K))."""
        if self.eigenvectors is None:
            raise ValueError("PCA model has not been fitted yet.")
        if k is None:
            k = self.n_components if self.n_components is not None else self.eigenvectors.shape[1]
        k = min(k, self.eigenvectors.shape[1])
        return self.eigenvectors[:, :k]

    def project(self, X: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        r"""Project input data onto top K eigenfaces subspace: \omega = U_K^T * (x - \mu).
        
        Args:
            X: Input matrix of shape (M, D) or (D,)
            k: Number of components to retain.
            
        Returns:
            Projected weights of shape (M, K) or (K,).
        """
        if self.mean_vector is None or self.eigenvectors is None:
            raise ValueError("PCA model has not been fitted yet.")
            
        is_1d = (X.ndim == 1)
        X_mat = np.atleast_2d(X)
        X_centered = X_mat - self.mean_vector
        U_k = self.get_components(k)
        
        # \omega = (x - \mu) * U_K  (which is (U_K^T * (x - \mu)^T)^T)
        weights = np.dot(X_centered, U_k)
        
        return weights[0] if is_1d else weights

    def reconstruct(self, weights: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        r"""Reconstruct face from projection weights: \hat{x} = \mu + U_K * \omega.
        
        Args:
            weights: Projected weights of shape (M, K) or (K,)
            k: Number of components used.
            
        Returns:
            Reconstructed vector of shape (M, D) or (D,).
        """
        if self.mean_vector is None or self.eigenvectors is None:
            raise ValueError("PCA model has not been fitted yet.")
            
        is_1d = (weights.ndim == 1)
        W_mat = np.atleast_2d(weights)
        curr_k = W_mat.shape[1]
        U_k = self.get_components(curr_k)
        
        # \hat{x} = \omega * U_K^T + \mu
        reconstructed = np.dot(W_mat, U_k.T) + self.mean_vector
        return reconstructed[0] if is_1d else reconstructed
