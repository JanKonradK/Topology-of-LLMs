"""
Tests for DiagramDistances.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def dgm_a() -> np.ndarray:
    return np.array([[0.0, 1.0], [0.2, 0.8], [0.5, 2.0]])


@pytest.fixture
def dgm_b() -> np.ndarray:
    return np.array([[0.1, 0.9], [0.3, 1.5]])


class TestWasserstein:
    """Tests for Wasserstein distance."""

    @pytest.mark.slow
    def test_self_distance_zero(self, dgm_a: np.ndarray) -> None:
        """W(D, D) = 0."""
        from topo_llm.topology.distances import DiagramDistances

        d = DiagramDistances.wasserstein(dgm_a, dgm_a)
        assert abs(d) < 1e-6

    @pytest.mark.slow
    def test_nonnegative(self, dgm_a: np.ndarray, dgm_b: np.ndarray) -> None:
        """Wasserstein distance is non-negative."""
        from topo_llm.topology.distances import DiagramDistances

        d = DiagramDistances.wasserstein(dgm_a, dgm_b)
        assert d >= 0

    @pytest.mark.slow
    def test_symmetric(self, dgm_a: np.ndarray, dgm_b: np.ndarray) -> None:
        """W(A, B) = W(B, A)."""
        from topo_llm.topology.distances import DiagramDistances

        d1 = DiagramDistances.wasserstein(dgm_a, dgm_b)
        d2 = DiagramDistances.wasserstein(dgm_b, dgm_a)
        assert abs(d1 - d2) < 1e-6

    @pytest.mark.slow
    def test_empty_diagrams(self) -> None:
        """Distance between empty diagrams is 0."""
        from topo_llm.topology.distances import DiagramDistances

        d = DiagramDistances.wasserstein(np.empty((0, 2)), np.empty((0, 2)))
        assert d == 0.0


class TestBottleneck:
    """Tests for bottleneck distance."""

    @pytest.mark.slow
    def test_self_distance_zero(self, dgm_a: np.ndarray) -> None:
        """d_B(D, D) = 0."""
        from topo_llm.topology.distances import DiagramDistances

        d = DiagramDistances.bottleneck(dgm_a, dgm_a)
        assert abs(d) < 1e-6

    @pytest.mark.slow
    def test_nonnegative(self, dgm_a: np.ndarray, dgm_b: np.ndarray) -> None:
        """Bottleneck distance is non-negative."""
        from topo_llm.topology.distances import DiagramDistances

        d = DiagramDistances.bottleneck(dgm_a, dgm_b)
        assert d >= 0


class TestDistanceMatrix:
    """Tests for distance_matrix()."""

    @pytest.mark.slow
    def test_symmetric(self, dgm_a: np.ndarray, dgm_b: np.ndarray) -> None:
        """Distance matrix should be symmetric."""
        from topo_llm.topology.distances import DiagramDistances

        dgms = [dgm_a, dgm_b, np.array([[0.0, 0.5]])]
        D = DiagramDistances.distance_matrix(dgms, show_progress=False)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    @pytest.mark.slow
    def test_diagonal_zero(self, dgm_a: np.ndarray, dgm_b: np.ndarray) -> None:
        """Diagonal should be zero."""
        from topo_llm.topology.distances import DiagramDistances

        dgms = [dgm_a, dgm_b]
        D = DiagramDistances.distance_matrix(dgms, show_progress=False)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-6)
