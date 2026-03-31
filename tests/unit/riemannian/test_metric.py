"""
Tests for MetricTensorEstimator.

Validates metric estimation on manifolds with known geometry,
especially the 2-sphere where analytical results are available.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.riemannian.metric import MetricTensorEstimator


class TestMetricTensorEstimator:
    """Tests for the MetricTensorEstimator class."""

    def test_fit_basic(self, sphere_points: np.ndarray) -> None:
        """Fitting should succeed and populate all attributes."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        est.fit(sphere_points)

        assert est.point_cloud_ is not None
        assert len(est.tangent_bases_) == len(sphere_points)
        assert len(est.metric_tensors_) == len(sphere_points)
        assert len(est.metric_inverses_) == len(sphere_points)
        assert est.intrinsic_dim_ == 2

    def test_metric_shape(self, sphere_points: np.ndarray) -> None:
        """Metric tensors should have shape (m, m)."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        est.fit(sphere_points)

        g = est.get_metric_at(0)
        assert g.shape == (2, 2)

    def test_tangent_basis_shape(self, sphere_points: np.ndarray) -> None:
        """Tangent basis should have shape (D, m)."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        est.fit(sphere_points)

        T = est.get_tangent_basis_at(0)
        assert T.shape == (3, 2)  # D=3, m=2

    def test_metric_positive_definite(self, sphere_points: np.ndarray) -> None:
        """All metric tensors should be positive definite."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        est.fit(sphere_points)

        for i in range(len(sphere_points)):
            g = est.get_metric_at(i)
            eigenvalues = np.linalg.eigvalsh(g)
            assert np.all(eigenvalues > 0), (
                f"Point {i}: metric not positive definite, "
                f"eigenvalues = {eigenvalues}"
            )

    def test_metric_determinant_positive(self, sphere_points: np.ndarray) -> None:
        """det(g) > 0 everywhere."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        est.fit(sphere_points)

        for i in range(len(sphere_points)):
            g = est.get_metric_at(i)
            det = np.linalg.det(g)
            assert det > 0, f"Point {i}: det(g) = {det}"

    def test_volume_element_positive(self, sphere_points: np.ndarray) -> None:
        """Volume element sqrt(det(g)) should be positive."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        est.fit(sphere_points)

        for i in range(len(sphere_points)):
            vol = est.volume_element(i)
            assert vol > 0, f"Point {i}: volume element = {vol}"

    def test_interpolate_metric(self, sphere_points: np.ndarray) -> None:
        """Interpolated metric at a known point should be close to fitted."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        est.fit(sphere_points)

        # Interpolate at the first point — should match closely
        g_fitted = est.get_metric_at(0)
        g_interp = est.interpolate_metric(sphere_points[0])

        np.testing.assert_allclose(g_fitted, g_interp, rtol=0.3)

    def test_auto_intrinsic_dim(self, sphere_points: np.ndarray) -> None:
        """Auto-estimated intrinsic dim of sphere should be ~2."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=None)
        est.fit(sphere_points)

        assert 1 <= est.intrinsic_dim_ <= 4, (
            f"Expected ~2, got {est.intrinsic_dim_}"
        )

    def test_flat_plane_metric_near_identity(self, flat_plane_points: np.ndarray) -> None:
        """On a flat plane, metric should be close to the identity."""
        est = MetricTensorEstimator(n_neighbors=30, intrinsic_dim=2)
        est.fit(flat_plane_points)

        # Check a few points
        for i in [0, 10, 50]:
            g = est.get_metric_at(i)
            # Should be close to identity (scaled by local variance)
            np.testing.assert_allclose(
                g / g[0, 0], np.eye(2), atol=0.5,
                err_msg=f"Point {i}: metric far from identity"
            )

    def test_project_and_lift_roundtrip(self, sphere_points: np.ndarray) -> None:
        """Project then lift should approximately recover the original vector."""
        est = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        est.fit(sphere_points)

        # Take a vector in the tangent plane
        T = est.get_tangent_basis_at(0)
        v_tangent = np.array([1.0, 0.5])
        v_ambient = est.lift_from_tangent(0, v_tangent)
        v_recovered = est.project_to_tangent(0, v_ambient)

        np.testing.assert_allclose(v_tangent, v_recovered, atol=0.1)
