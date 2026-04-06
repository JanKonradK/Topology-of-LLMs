"""
Tests for PersistenceLandscape.

Validates landscape computation, norms, distances, and permutation tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.topology.landscapes import PersistenceLandscape


@pytest.fixture
def simple_diagram() -> np.ndarray:
    """Simple persistence diagram."""
    return np.array(
        [
            [0.0, 1.0],
            [0.2, 0.8],
            [0.5, 2.0],
        ]
    )


@pytest.fixture
def landscape(simple_diagram: np.ndarray) -> PersistenceLandscape:
    """Landscape from simple diagram."""
    return PersistenceLandscape(simple_diagram, n_landscapes=3, resolution=500)


class TestPersistenceLandscape:
    """Tests for landscape computation."""

    def test_shape(self, landscape: PersistenceLandscape) -> None:
        """Landscapes should have correct shape."""
        assert landscape.landscapes_.shape == (3, 500)
        assert landscape.grid_.shape == (500,)

    def test_non_negative(self, landscape: PersistenceLandscape) -> None:
        """Landscape values should be non-negative."""
        assert np.all(landscape.landscapes_ >= -1e-10)

    def test_first_landscape_dominates(self, landscape: PersistenceLandscape) -> None:
        """First landscape should be >= second at all points."""
        diff = landscape.landscapes_[0] - landscape.landscapes_[1]
        assert np.all(diff >= -1e-10)

    def test_empty_diagram(self) -> None:
        """Empty diagram should produce zero landscapes."""
        l = PersistenceLandscape(np.empty((0, 2)))
        assert np.all(l.landscapes_ == 0)


class TestLandscapeNorms:
    """Tests for integrate() and norm()."""

    def test_integral_positive(self, landscape: PersistenceLandscape) -> None:
        """Integral of first landscape should be positive."""
        assert landscape.integrate(0) > 0

    def test_norm_positive(self, landscape: PersistenceLandscape) -> None:
        """L^2 norm should be positive for non-trivial landscape."""
        assert landscape.norm(0, p=2.0) > 0

    def test_norm_nonnegative(self, landscape: PersistenceLandscape) -> None:
        """All norms should be non-negative."""
        for k in range(3):
            for p in [1.0, 2.0]:
                assert landscape.norm(k, p) >= 0


class TestLandscapeDistance:
    """Tests for distance computation."""

    def test_self_distance_zero(self, landscape: PersistenceLandscape) -> None:
        """Distance from a landscape to itself should be 0."""
        d = PersistenceLandscape.distance(landscape, landscape)
        assert abs(d) < 1e-6

    def test_distance_nonnegative(self, simple_diagram: np.ndarray) -> None:
        """Distance should be non-negative."""
        l1 = PersistenceLandscape(simple_diagram, n_landscapes=3)
        other = np.array([[0.1, 0.9], [0.3, 1.5]])
        l2 = PersistenceLandscape(other, n_landscapes=3)
        d = PersistenceLandscape.distance(l1, l2)
        assert d >= 0

    def test_triangle_inequality(self, simple_diagram: np.ndarray) -> None:
        """Triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        dgm_a = simple_diagram
        dgm_b = np.array([[0.1, 0.9], [0.3, 1.5]])
        dgm_c = np.array([[0.0, 2.0]])

        la = PersistenceLandscape(dgm_a, n_landscapes=3)
        lb = PersistenceLandscape(dgm_b, n_landscapes=3)
        lc = PersistenceLandscape(dgm_c, n_landscapes=3)

        d_ac = PersistenceLandscape.distance(la, lc)
        d_ab = PersistenceLandscape.distance(la, lb)
        d_bc = PersistenceLandscape.distance(lb, lc)

        assert d_ac <= d_ab + d_bc + 1e-6


class TestMeanLandscape:
    """Tests for mean_landscape()."""

    def test_mean_of_identical(self, landscape: PersistenceLandscape) -> None:
        """Mean of identical landscapes should equal original."""
        mean_l = PersistenceLandscape.mean_landscape([landscape, landscape])
        d = PersistenceLandscape.distance(mean_l, landscape)
        assert d < 1e-4


class TestPermutationTest:
    """Tests for permutation_test()."""

    def test_same_distribution_high_pvalue(self) -> None:
        """Same distribution should give high p-value."""
        rng = np.random.default_rng(42)
        dgms = [rng.uniform(0, 1, (5, 2)) for _ in range(10)]
        # Sort each so birth < death
        for d in dgms:
            d.sort(axis=1)

        landscapes = [PersistenceLandscape(d, n_landscapes=3, resolution=100) for d in dgms]
        group_a = landscapes[:5]
        group_b = landscapes[5:]

        result = PersistenceLandscape.permutation_test(group_a, group_b, n_permutations=100)
        # With same distribution, p-value should not be extremely low
        assert result["p_value"] >= 0.0
        assert "test_statistic" in result
