"""Unit tests for mathematical PCA and Eigenfaces."""
import pytest
import numpy as np
from src.eigenfaces.pca_from_scratch import PCAFromScratch
from src.eigenfaces.eigenfaces_model import EigenfacesModel


def test_pca_eigenvalue_sorting():
    np.random.seed(42)
    N, D = 30, 200
    X = np.random.randn(N, D)

    pca = PCAFromScratch()
    pca.fit(X)

    # Check strictly descending eigenvalues
    diffs = np.diff(pca.eigenvalues)
    assert np.all(diffs <= 1e-9), "Eigenvalues must be in descending order"


def test_eigenvector_unit_normalization():
    np.random.seed(42)
    N, D = 25, 500
    X = np.random.randn(N, D)

    pca = PCAFromScratch()
    pca.fit(X)

    U = pca.eigenvectors
    norms = np.linalg.norm(U, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-5), "Each eigenface eigenvector must have unit Euclidean length"


def test_projection_and_reconstruction():
    np.random.seed(42)
    N, D = 20, 100
    X = np.random.randn(N, D)

    pca = PCAFromScratch(n_components=10)
    pca.fit(X)

    # Project single vector
    x_test = X[0]
    weights = pca.project(x_test, k=10)
    assert weights.shape == (10,)

    # Reconstruct
    x_rec = pca.reconstruct(weights, k=10)
    assert x_rec.shape == (D,)

    # With full rank components, reconstruction error should be very small
    weights_full = pca.project(x_test, k=N - 1)
    x_rec_full = pca.reconstruct(weights_full, k=N - 1)
    mse = np.mean((x_test - x_rec_full) ** 2)
    assert mse < 1e-4, f"Full component reconstruction error too high: {mse}"


def test_eigenfaces_model_persistence(tmp_path):
    np.random.seed(42)
    N, D = 15, 1600  # 40x40 images
    X = np.random.randn(N, D)

    model = EigenfacesModel(target_shape=(40, 40), n_components=5)
    model.train(X)

    save_file = tmp_path / "test_pca.pkl"
    model.save(save_file)
    assert save_file.exists()

    loaded_model = EigenfacesModel(target_shape=(40, 40))
    loaded_model.load(save_file)
    assert loaded_model.is_trained is True
    assert loaded_model.n_components == 5
    assert np.allclose(model.pca.mean_vector, loaded_model.pca.mean_vector)

