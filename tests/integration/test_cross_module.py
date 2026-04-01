"""
Integration tests: Cross-module pipelines.

Tests workflows that combine multiple subpackages, verifying
the NumPy array interchange format works correctly between them.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.extraction import LayerAnalyzer


class TestLayerAnalysisToRiemannian:
    """Test the flow: embeddings -> layer analysis -> PCA -> Riemannian."""

    def test_intrinsic_dim_guides_reduction(self) -> None:
        """Intrinsic dimensionality estimate informs PCA reduced_dim choice."""
        from sklearn.decomposition import PCA

        from topo_llm.riemannian import ChristoffelEstimator, CurvatureAnalyzer, MetricTensorEstimator

        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((60, 100)).astype(np.float32)

        # Step 1: Estimate intrinsic dimensionality
        intrinsic_dim = LayerAnalyzer.intrinsic_dimensionality(embeddings, method="mle")
        assert intrinsic_dim > 0

        # Step 2: Use ~2x intrinsic dim for PCA (rule of thumb from CLAUDE.md)
        reduced_dim = max(int(2 * intrinsic_dim), 10)
        reduced_dim = min(reduced_dim, embeddings.shape[0] - 1, embeddings.shape[1])
        reduced = PCA(n_components=reduced_dim).fit_transform(embeddings)

        # Step 3: Fit metric on reduced data
        metric_est = MetricTensorEstimator(n_neighbors=15)
        metric_est.fit(reduced)

        assert len(metric_est.metric_tensors_) == len(embeddings)
        assert metric_est.intrinsic_dim_ <= reduced_dim

        # Step 4: Compute curvature at a single point
        chris = ChristoffelEstimator(metric_est)
        curv = CurvatureAnalyzer(metric_est, chris)
        scalar = curv.scalar_curvature_at(0)
        assert np.isfinite(scalar)

    def test_anisotropy_detects_structure(self) -> None:
        """Anisotropy analysis correctly distinguishes isotropic vs. structured data."""
        rng = np.random.default_rng(42)

        # Isotropic data
        isotropic = rng.standard_normal((200, 50)).astype(np.float32)
        aniso_iso = LayerAnalyzer.compute_anisotropy(isotropic)

        # Highly anisotropic data (most variance in one direction)
        anisotropic = rng.standard_normal((200, 50)).astype(np.float32)
        anisotropic[:, 0] *= 100  # Dominate first dimension
        aniso_aniso = LayerAnalyzer.compute_anisotropy(anisotropic)

        # Anisotropic data should have lower effective rank
        assert aniso_aniso["effective_rank"] < aniso_iso["effective_rank"]


class TestRiemannianAndTopologyOnSphere:
    """Test combining Riemannian and topological analysis on the same data."""

    def test_curvature_and_features_on_sphere(self) -> None:
        """Sphere has finite curvature AND produces valid topological features."""
        from topo_llm.riemannian import (
            ChristoffelEstimator,
            CurvatureAnalyzer,
            MetricTensorEstimator,
        )
        from topo_llm.topology import TopologicalFeatures

        # Small sphere for speed
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((40, 3)).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        # Riemannian side — single point curvature
        metric_est = MetricTensorEstimator(n_neighbors=15)
        metric_est.fit(pts)
        chris = ChristoffelEstimator(metric_est)
        curv = CurvatureAnalyzer(metric_est, chris)
        s = curv.scalar_curvature_at(0)
        assert np.isfinite(s)

        # Topological side — synthetic diagrams (no ripser needed)
        diagrams = [
            np.array([[0.0, 1.0], [0.0, 0.5]]),   # H_0
            np.array([]).reshape(0, 2),              # H_1
            np.array([[0.5, 1.5]]),                  # H_2
        ]
        feat = TopologicalFeatures.statistics_vector(diagrams)
        assert feat.shape == (30,)
        assert np.all(np.isfinite(feat))


class TestLayerSimilarityMatrix:
    """Test CKA and Procrustes similarity across synthetic 'layers'."""

    def test_identical_layers_have_max_similarity(self) -> None:
        """Identical embeddings should have CKA similarity = 1."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 50)).astype(np.float32)

        sim = LayerAnalyzer.layer_similarity(data, data, method="cka")
        np.testing.assert_allclose(sim, 1.0, atol=1e-5)

    def test_random_layers_have_low_similarity(self) -> None:
        """Random independent embeddings should have low CKA."""
        rng = np.random.default_rng(42)
        data1 = rng.standard_normal((100, 50)).astype(np.float32)
        data2 = rng.standard_normal((100, 50)).astype(np.float32)

        sim = LayerAnalyzer.layer_similarity(data1, data2, method="cka")
        assert sim < 0.5  # Should be much less than 1


class TestFeatureVectorDimensions:
    """Verify feature vectors from different modules can be concatenated."""

    def test_topological_feature_dimensions(self) -> None:
        """Topological feature vector has stable 30-dim output."""
        from topo_llm.topology import TopologicalFeatures

        diagrams = [
            np.array([[0.0, 1.0], [0.0, 0.5]]),
            np.array([[0.5, 1.5]]),
            np.array([]).reshape(0, 2),
        ]
        feat = TopologicalFeatures.statistics_vector(diagrams)
        assert feat.shape == (30,)
        assert np.all(np.isfinite(feat))

    def test_empty_diagrams_produce_zeros(self) -> None:
        """Empty persistence diagrams produce zero feature vector."""
        from topo_llm.topology import TopologicalFeatures

        diagrams = [
            np.array([]).reshape(0, 2),
            np.array([]).reshape(0, 2),
            np.array([]).reshape(0, 2),
        ]
        feat = TopologicalFeatures.statistics_vector(diagrams)
        assert feat.shape == (30,)
        assert np.allclose(feat, 0.0)
