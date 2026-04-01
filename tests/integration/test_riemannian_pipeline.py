"""
Integration tests: Riemannian geometry pipeline.

Tests the full flow from point cloud -> metric -> connection -> curvature -> geodesic
on synthetic manifolds with known properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.riemannian import (
    ChristoffelEstimator,
    CurvatureAnalyzer,
    GeodesicSolver,
    MetricTensorEstimator,
    RiemannianSearch,
)


@pytest.fixture
def small_sphere() -> np.ndarray:
    """50 points on unit sphere — small enough for full curvature stats."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((50, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return (pts / norms).astype(np.float32)


@pytest.fixture
def small_flat() -> np.ndarray:
    """50 points on flat plane."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-5, 5, (50, 2))
    z = np.zeros((50, 1))
    return np.hstack([xy, z]).astype(np.float32)


class TestSphereFullPipeline:
    """End-to-end Riemannian analysis on a unit sphere."""

    def test_metric_to_curvature(self, small_sphere: np.ndarray) -> None:
        """Metric -> Connection -> Curvature pipeline on sphere."""
        metric_est = MetricTensorEstimator(n_neighbors=15)
        metric_est.fit(small_sphere)

        assert len(metric_est.metric_tensors_) == len(small_sphere)

        chris = ChristoffelEstimator(metric_est)
        curv = CurvatureAnalyzer(metric_est, chris)

        # Check single point curvature
        s = curv.scalar_curvature_at(0)
        assert np.isfinite(s)

        # Full statistics on small data
        stats = curv.curvature_statistics(show_progress=False)
        assert stats["mean"] is not None
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_geodesic_on_sphere(self, small_sphere: np.ndarray) -> None:
        """Geodesic solver runs without error on sphere metric."""
        metric_est = MetricTensorEstimator(n_neighbors=15)
        metric_est.fit(small_sphere)
        chris = ChristoffelEstimator(metric_est)

        solver = GeodesicSolver(metric_est, chris, dt=0.01, max_steps=50)

        m = metric_est.intrinsic_dim_
        v0 = np.random.default_rng(42).standard_normal(m) * 0.1

        result = solver.solve(start_idx=0, initial_velocity=v0)

        assert result.ambient_path.shape[1] == small_sphere.shape[1]  # ambient dim
        assert result.arc_length >= 0

    def test_search_euclidean_and_cosine(self, small_sphere: np.ndarray) -> None:
        """RiemannianSearch returns results for Euclidean and cosine queries."""
        metric_est = MetricTensorEstimator(n_neighbors=15)
        metric_est.fit(small_sphere)
        chris = ChristoffelEstimator(metric_est)
        solver = GeodesicSolver(metric_est, chris)

        search = RiemannianSearch(solver, metric_est)

        euc_results = search.query_euclidean(query_idx=0, k=5)
        assert len(euc_results) == 5
        for idx, dist in euc_results:
            assert isinstance(idx, int)
            assert dist >= 0

        cos_results = search.query_cosine(query_idx=0, k=5)
        assert len(cos_results) == 5


class TestFlatPlanePipeline:
    """End-to-end: flat plane should have near-zero curvature."""

    def test_flat_single_point_curvature(self, small_flat: np.ndarray) -> None:
        """Flat plane curvature should be approximately zero at any point."""
        metric_est = MetricTensorEstimator(n_neighbors=15)
        metric_est.fit(small_flat)
        chris = ChristoffelEstimator(metric_est)

        curv = CurvatureAnalyzer(metric_est, chris)
        s = curv.scalar_curvature_at(0)
        assert abs(s) < 5.0  # Near zero with numerical noise


class TestPCAReductionPipeline:
    """Test the mandatory PCA reduction workflow for high-dimensional data."""

    def test_high_dim_to_riemannian(self) -> None:
        """768-dim embeddings -> PCA -> metric -> curvature works end-to-end."""
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 100)).astype(np.float32)

        reduced = PCA(n_components=10).fit_transform(embeddings)
        assert reduced.shape == (50, 10)

        metric_est = MetricTensorEstimator(n_neighbors=15)
        metric_est.fit(reduced)

        chris = ChristoffelEstimator(metric_est)
        curv = CurvatureAnalyzer(metric_est, chris)

        scalar = curv.scalar_curvature_at(0)
        assert np.isfinite(scalar)
