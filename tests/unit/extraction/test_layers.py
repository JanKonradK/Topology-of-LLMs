"""
Tests for LayerAnalyzer.

Validates intrinsic dimensionality estimation, layer similarity,
and anisotropy computation using synthetic data with known properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.extraction.layers import LayerAnalyzer


class TestIntrinsicDimensionality:
    """Tests for LayerAnalyzer.intrinsic_dimensionality()."""

    def test_known_5d_subspace_mle(self, low_dim_subspace: np.ndarray) -> None:
        """MLE should estimate ~5 for data from a 5D subspace of R^100."""
        dim = LayerAnalyzer.intrinsic_dimensionality(low_dim_subspace, method="mle")
        assert 3.0 <= dim <= 8.0, f"Expected ~5, got {dim}"

    def test_known_5d_subspace_twonn(self, low_dim_subspace: np.ndarray) -> None:
        """TwoNN should estimate ~5 for data from a 5D subspace of R^100."""
        dim = LayerAnalyzer.intrinsic_dimensionality(low_dim_subspace, method="twonn")
        assert 3.0 <= dim <= 8.0, f"Expected ~5, got {dim}"

    def test_known_5d_subspace_pca(self, low_dim_subspace: np.ndarray) -> None:
        """PCA should find ~5 components for 95% variance."""
        dim = LayerAnalyzer.intrinsic_dimensionality(low_dim_subspace, method="pca_95")
        assert dim == 5.0, f"Expected 5, got {dim}"

    def test_sphere_is_2d(self, sphere_points: np.ndarray) -> None:
        """Points on a sphere should have intrinsic dim ≈ 2."""
        dim = LayerAnalyzer.intrinsic_dimensionality(sphere_points, method="mle")
        assert 1.5 <= dim <= 3.5, f"Expected ~2, got {dim}"

    def test_circle_is_1d(self, circle_points: np.ndarray) -> None:
        """Points on a circle should have intrinsic dim ≈ 1."""
        dim = LayerAnalyzer.intrinsic_dimensionality(circle_points, method="mle")
        assert 0.5 <= dim <= 2.0, f"Expected ~1, got {dim}"

    def test_positive_result(self, small_embeddings: np.ndarray) -> None:
        """Dimensionality should always be positive."""
        for method in ["mle", "twonn", "pca_95"]:
            dim = LayerAnalyzer.intrinsic_dimensionality(
                small_embeddings, method=method
            )
            assert dim > 0, f"Method {method} returned non-positive: {dim}"

    def test_invalid_method_raises(self, small_embeddings: np.ndarray) -> None:
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError):
            LayerAnalyzer.intrinsic_dimensionality(
                small_embeddings, method="invalid"
            )


class TestLayerSimilarity:
    """Tests for LayerAnalyzer.layer_similarity()."""

    def test_identical_matrices_cka(self, small_embeddings: np.ndarray) -> None:
        """CKA of a matrix with itself should be 1.0."""
        sim = LayerAnalyzer.layer_similarity(
            small_embeddings, small_embeddings, method="cka"
        )
        assert abs(sim - 1.0) < 0.01, f"Expected 1.0, got {sim}"

    def test_identical_matrices_procrustes(self, small_embeddings: np.ndarray) -> None:
        """Procrustes similarity of identical matrices should be ~1.0."""
        sim = LayerAnalyzer.layer_similarity(
            small_embeddings, small_embeddings, method="procrustes"
        )
        assert sim > 0.95, f"Expected ~1.0, got {sim}"

    def test_identical_matrices_cca(self, small_embeddings: np.ndarray) -> None:
        """CCA similarity of identical matrices should be high."""
        sim = LayerAnalyzer.layer_similarity(
            small_embeddings, small_embeddings, method="cca"
        )
        assert sim > 0.9, f"Expected ~1.0, got {sim}"

    def test_random_matrices_lower_similarity(self, rng: np.random.Generator) -> None:
        """Random independent matrices should have lower CKA."""
        X = rng.standard_normal((100, 64))
        Y = rng.standard_normal((100, 64))
        sim = LayerAnalyzer.layer_similarity(X, Y, method="cka")
        assert sim < 0.5, f"Expected low similarity, got {sim}"

    def test_similarity_in_range(self, small_embeddings: np.ndarray, rng: np.random.Generator) -> None:
        """All similarity methods should return values in [0, 1]."""
        other = rng.standard_normal(small_embeddings.shape)
        for method in ["cka", "procrustes", "cca"]:
            sim = LayerAnalyzer.layer_similarity(
                small_embeddings, other, method=method
            )
            assert 0.0 <= sim <= 1.01, f"{method}: {sim} out of range"

    def test_invalid_method_raises(self, small_embeddings: np.ndarray) -> None:
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError):
            LayerAnalyzer.layer_similarity(
                small_embeddings, small_embeddings, method="invalid"
            )


class TestAnisotropy:
    """Tests for LayerAnalyzer.compute_anisotropy()."""

    def test_isotropic_gaussian(self, rng: np.random.Generator) -> None:
        """Isotropic Gaussian should have low mean cosine and high effective rank."""
        data = rng.standard_normal((500, 50))
        result = LayerAnalyzer.compute_anisotropy(data)

        # Mean cosine of isotropic data should be near 0
        assert abs(result["mean_cosine"]) < 0.2, (
            f"Mean cosine too high for isotropic data: {result['mean_cosine']}"
        )
        # Effective rank should be high (close to min(n, d))
        assert result["effective_rank"] > 20, (
            f"Effective rank too low: {result['effective_rank']}"
        )

    def test_anisotropic_data(self, rng: np.random.Generator) -> None:
        """Data with one dominant direction should have high mean cosine."""
        # Create highly anisotropic data (one dominant direction)
        n = 500
        data = rng.standard_normal((n, 50)) * 0.01
        data[:, 0] = rng.standard_normal(n) * 10  # dominant direction
        result = LayerAnalyzer.compute_anisotropy(data)

        # Effective rank should be low
        assert result["effective_rank"] < 10, (
            f"Effective rank too high for anisotropic data: {result['effective_rank']}"
        )

    def test_return_keys(self, small_embeddings: np.ndarray) -> None:
        """Result dict should contain all expected keys."""
        result = LayerAnalyzer.compute_anisotropy(small_embeddings)
        expected_keys = {
            "mean_cosine", "isotropy_score",
            "explained_variance_ratio", "effective_rank",
        }
        assert set(result.keys()) == expected_keys

    def test_explained_variance_sums_to_one(self, small_embeddings: np.ndarray) -> None:
        """Explained variance ratios should sum to <= 1.0 (partial components)."""
        result = LayerAnalyzer.compute_anisotropy(small_embeddings)
        evr = result["explained_variance_ratio"]
        assert evr.sum() <= 1.01

    def test_effective_rank_positive(self, small_embeddings: np.ndarray) -> None:
        """Effective rank should be positive."""
        result = LayerAnalyzer.compute_anisotropy(small_embeddings)
        assert result["effective_rank"] > 0
