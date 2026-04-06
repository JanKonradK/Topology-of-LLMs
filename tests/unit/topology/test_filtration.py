"""
Tests for FiltrationBuilder.

Validates filtration construction and maxmin subsampling.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestMaxminSubsample:
    """Tests for furthest point sampling."""

    def test_correct_count(self, sphere_points: np.ndarray) -> None:
        """Should return exactly n_points."""
        from topo_llm.topology.filtration import FiltrationBuilder

        sub, idx = FiltrationBuilder.maxmin_subsample(sphere_points, n_points=50)
        assert sub.shape[0] == 50
        assert len(idx) == 50

    def test_preserves_dimension(self, sphere_points: np.ndarray) -> None:
        """Subsampled points should have same ambient dimension."""
        from topo_llm.topology.filtration import FiltrationBuilder

        sub, _ = FiltrationBuilder.maxmin_subsample(sphere_points, n_points=50)
        assert sub.shape[1] == sphere_points.shape[1]

    def test_unique_indices(self, sphere_points: np.ndarray) -> None:
        """All selected indices should be unique."""
        from topo_llm.topology.filtration import FiltrationBuilder

        _, idx = FiltrationBuilder.maxmin_subsample(sphere_points, n_points=50)
        assert len(set(idx)) == 50

    def test_well_spread(self, sphere_points: np.ndarray) -> None:
        """Maxmin should produce a well-spread sample."""
        from topo_llm.topology.filtration import FiltrationBuilder

        sub, _ = FiltrationBuilder.maxmin_subsample(sphere_points, n_points=50)
        # Minimum pairwise distance should be reasonable
        from scipy.spatial.distance import pdist

        min_dist = pdist(sub).min()
        assert min_dist > 0.01, f"Points too close: min_dist = {min_dist}"


class TestVietorisRips:
    """Tests for Vietoris-Rips filtration."""

    @pytest.mark.slow
    def test_circle_h1(self, circle_points: np.ndarray) -> None:
        """Circle should have 1 persistent H_1 feature."""
        from topo_llm.topology.filtration import FiltrationBuilder

        result = FiltrationBuilder.vietoris_rips(circle_points, max_dimension=1)
        assert len(result.diagrams) >= 2
        assert result.n_points_used == len(circle_points)

    @pytest.mark.slow
    def test_result_structure(self, sphere_points: np.ndarray) -> None:
        """Result should have correct structure."""
        from topo_llm.topology.filtration import FiltrationBuilder

        result = FiltrationBuilder.vietoris_rips(sphere_points, max_dimension=2, n_points=100)
        assert result.n_points_used == 100
        assert result.computation_time > 0
        assert result.backend == "ripser"
        assert len(result.diagrams) >= 1
