"""
Tests for ChristoffelEstimator.

Validates Christoffel symbol computation, especially the lower-index
symmetry Γ^k_{ij} = Γ^k_{ji}.
"""

from __future__ import annotations

import numpy as np

from topo_llm.riemannian.connection import ChristoffelEstimator
from topo_llm.riemannian.metric import MetricTensorEstimator


class TestChristoffelEstimator:
    """Tests for Christoffel symbol computation."""

    def test_shape(self, sphere_points: np.ndarray) -> None:
        """Christoffel symbols should have shape (m, m, m)."""
        met = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        met.fit(sphere_points)
        chris = ChristoffelEstimator(met, h=1e-3)

        gamma = chris.compute_at(0)
        assert gamma.shape == (2, 2, 2)

    def test_symmetry_in_lower_indices(self, sphere_points: np.ndarray) -> None:
        """Γ^k_{ij} should equal Γ^k_{ji} for all k."""
        met = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        met.fit(sphere_points)
        chris = ChristoffelEstimator(met, h=1e-3)

        for idx in [0, 5, 10]:
            assert chris.verify_symmetry(idx, tol=1e-6), (
                f"Christoffel symbols not symmetric at point {idx}"
            )

    def test_flat_plane_near_zero(self, flat_plane_points: np.ndarray) -> None:
        """On a flat plane, Christoffel symbols should be near zero."""
        met = MetricTensorEstimator(n_neighbors=30, intrinsic_dim=2)
        met.fit(flat_plane_points)
        chris = ChristoffelEstimator(met, h=1e-3)

        gamma = chris.compute_at(50)
        max_val = np.abs(gamma).max()
        # Allow some numerical noise but should be small
        assert max_val < 5.0, f"Flat plane Christoffel symbols too large: max = {max_val}"

    def test_compute_all(self, sphere_points: np.ndarray) -> None:
        """compute_all should return one array per point and cache."""
        met = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        met.fit(sphere_points)
        chris = ChristoffelEstimator(met, h=1e-3)

        result = chris.compute_all(show_progress=False)
        assert len(result) == len(sphere_points)
        assert all(g.shape == (2, 2, 2) for g in result)

        # Should be cached
        assert chris.christoffel_symbols_ is result

    def test_finite_values(self, sphere_points: np.ndarray) -> None:
        """All Christoffel values should be finite."""
        met = MetricTensorEstimator(n_neighbors=20, intrinsic_dim=2)
        met.fit(sphere_points)
        chris = ChristoffelEstimator(met, h=1e-3)

        for idx in [0, 10, 50]:
            gamma = chris.compute_at(idx)
            assert np.all(np.isfinite(gamma)), f"Non-finite Christoffel values at point {idx}"
