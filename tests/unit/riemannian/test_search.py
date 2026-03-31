"""
Tests for RiemannianSearch.

Validates geodesic-aware nearest neighbor search.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.riemannian.metric import MetricTensorEstimator
from topo_llm.riemannian.connection import ChristoffelEstimator
from topo_llm.riemannian.geodesic import GeodesicSolver
from topo_llm.riemannian.search import RiemannianSearch


@pytest.fixture
def flat_search(flat_plane_points: np.ndarray) -> RiemannianSearch:
    """RiemannianSearch on a flat plane."""
    met = MetricTensorEstimator(n_neighbors=30, intrinsic_dim=2)
    met.fit(flat_plane_points)
    chris = ChristoffelEstimator(met, h=1e-3)
    geo = GeodesicSolver(met, chris, dt=0.01, max_steps=100)
    return RiemannianSearch(geo, met)


class TestRiemannianSearch:
    """Tests for geodesic-aware search."""

    def test_euclidean_query_returns_correct_count(
        self, flat_search: RiemannianSearch
    ) -> None:
        """Euclidean query should return exactly k neighbors."""
        result = flat_search.query_euclidean(0, k=5)
        assert len(result) == 5
        for idx, dist in result:
            assert isinstance(idx, int)
            assert dist >= 0

    def test_cosine_query_returns_correct_count(
        self, flat_search: RiemannianSearch
    ) -> None:
        """Cosine query should return exactly k neighbors."""
        result = flat_search.query_cosine(0, k=5)
        assert len(result) == 5

    def test_geodesic_query_returns_correct_count(
        self, flat_search: RiemannianSearch
    ) -> None:
        """Geodesic query should return exactly k neighbors."""
        result = flat_search.query(0, k=3, candidates=10)
        assert len(result) == 3

    def test_compare_metrics_keys(self, flat_search: RiemannianSearch) -> None:
        """compare_metrics should return all expected keys."""
        result = flat_search.compare_metrics(0, k=3, candidates=10)
        expected_keys = {
            "euclidean_neighbors", "cosine_neighbors",
            "geodesic_neighbors", "rank_correlation_euclid_geo",
            "rank_correlation_cosine_geo",
        }
        assert set(result.keys()) == expected_keys

    def test_euclidean_sorted_ascending(
        self, flat_search: RiemannianSearch
    ) -> None:
        """Euclidean neighbors should be sorted by ascending distance."""
        result = flat_search.query_euclidean(0, k=5)
        distances = [d for _, d in result]
        assert distances == sorted(distances)
