"""
Tests for GeodesicSolver.

Validates geodesic computation and exponential/logarithmic maps.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.riemannian.connection import ChristoffelEstimator
from topo_llm.riemannian.geodesic import GeodesicSolver
from topo_llm.riemannian.metric import MetricTensorEstimator


@pytest.fixture
def flat_geodesic(flat_plane_points: np.ndarray) -> GeodesicSolver:
    """Geodesic solver on a flat plane."""
    met = MetricTensorEstimator(n_neighbors=30, intrinsic_dim=2)
    met.fit(flat_plane_points)
    chris = ChristoffelEstimator(met, h=1e-3)
    return GeodesicSolver(met, chris, dt=0.01, max_steps=200)


class TestGeodesicSolver:
    """Tests for geodesic solving."""

    def test_solve_returns_result(self, flat_geodesic: GeodesicSolver) -> None:
        """Solve should return a GeodesicResult with correct fields."""
        v0 = np.array([0.5, 0.3])
        result = flat_geodesic.solve(start_idx=0, initial_velocity=v0)

        assert result.tangent_path.shape[1] == 2
        assert result.ambient_path.shape[1] == 3  # R^3
        assert result.velocities.shape[1] == 2
        assert result.arc_length >= 0
        assert result.n_steps > 0

    def test_geodesic_starts_at_point(self, flat_geodesic: GeodesicSolver) -> None:
        """Geodesic should start at the specified point."""
        v0 = np.array([0.1, 0.2])
        result = flat_geodesic.solve(start_idx=5, initial_velocity=v0)

        start = flat_geodesic.metric.point_cloud_[5]
        np.testing.assert_allclose(result.ambient_path[0], start, atol=1e-6)

    def test_zero_velocity_stays_put(self, flat_geodesic: GeodesicSolver) -> None:
        """Zero initial velocity should stay near the starting point."""
        v0 = np.array([0.0, 0.0])
        result = flat_geodesic.solve(start_idx=0, initial_velocity=v0)

        start = flat_geodesic.metric.point_cloud_[0]
        endpoint = result.ambient_path[-1]
        dist = np.linalg.norm(endpoint - start)
        assert dist < 0.1, f"Moved {dist} with zero velocity"

    def test_arc_length_nonnegative(self, flat_geodesic: GeodesicSolver) -> None:
        """Arc length should be non-negative."""
        v0 = np.array([1.0, 0.0])
        result = flat_geodesic.solve(start_idx=0, initial_velocity=v0)
        assert result.arc_length >= 0

    def test_finite_path(self, flat_geodesic: GeodesicSolver) -> None:
        """All path points should be finite."""
        v0 = np.array([0.3, 0.2])
        result = flat_geodesic.solve(start_idx=0, initial_velocity=v0)
        assert np.all(np.isfinite(result.ambient_path))


class TestExponentialMap:
    """Tests for the exponential map."""

    def test_zero_vector_returns_base(self, flat_geodesic: GeodesicSolver) -> None:
        """Exp_x(0) should return x."""
        v0 = np.array([0.0, 0.0])
        result = flat_geodesic.exponential_map(0, v0)
        base = flat_geodesic.metric.point_cloud_[0]
        np.testing.assert_allclose(result, base, atol=0.1)

    def test_finite_result(self, flat_geodesic: GeodesicSolver) -> None:
        """Exponential map should return finite values."""
        v0 = np.array([0.5, 0.3])
        result = flat_geodesic.exponential_map(0, v0)
        assert np.all(np.isfinite(result))


class TestGeodesicDistance:
    """Tests for geodesic distance computation."""

    def test_self_distance_zero(self, flat_geodesic: GeodesicSolver) -> None:
        """Distance from a point to itself should be ~0."""
        d = flat_geodesic.geodesic_distance(0, 0, n_shooting=3)
        assert d < 0.5, f"Self-distance = {d}"

    def test_distance_nonnegative(self, flat_geodesic: GeodesicSolver) -> None:
        """Geodesic distance should be non-negative."""
        d = flat_geodesic.geodesic_distance(0, 5, n_shooting=3)
        assert d >= 0

    def test_distance_matrix_symmetric(self, flat_geodesic: GeodesicSolver) -> None:
        """Distance matrix should be approximately symmetric."""
        indices = [0, 1, 2, 3]
        D = flat_geodesic.geodesic_distance_matrix(indices, n_shooting=2, show_progress=False)
        np.testing.assert_allclose(D, D.T, atol=0.5)
