"""
Tests for PersistentHomologyAnalyzer.

Validates Betti number computation, persistence entropy,
and significance filtering.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.topology.homology import PersistentHomologyAnalyzer


@pytest.fixture
def simple_diagrams() -> list[np.ndarray]:
    """Simple test diagrams with known properties.

    H_0: 3 components, one very persistent
    H_1: 2 loops, one persistent
    """
    h0 = np.array([
        [0.0, 0.5],   # short-lived
        [0.0, 0.3],   # short-lived
        [0.0, 10.0],  # very persistent (connected component)
    ])
    h1 = np.array([
        [0.2, 0.8],   # short loop
        [0.1, 5.0],   # persistent loop
    ])
    return [h0, h1]


class TestBettiNumbers:
    """Tests for betti_numbers()."""

    def test_at_zero(self, simple_diagrams: list[np.ndarray]) -> None:
        """At epsilon=0, all features born at 0 should be alive."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        betti = analyzer.betti_numbers(0.0)
        assert betti[0] == 3  # all three H_0 features alive at 0

    def test_at_large_scale(self, simple_diagrams: list[np.ndarray]) -> None:
        """At very large epsilon, only very persistent features survive."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        betti = analyzer.betti_numbers(6.0)
        assert betti[0] == 1  # only the persistent component
        assert betti[1] == 0  # loops all died by 5.0


class TestBettiCurve:
    """Tests for betti_curve()."""

    def test_returns_all_dimensions(self, simple_diagrams: list[np.ndarray]) -> None:
        """Should return curves for all dimensions."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        curves = analyzer.betti_curve(n_steps=50)
        assert 0 in curves
        assert 1 in curves

    def test_curve_shape(self, simple_diagrams: list[np.ndarray]) -> None:
        """Each curve should have correct shape."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        curves = analyzer.betti_curve(n_steps=50)
        for k, (eps, vals) in curves.items():
            assert len(eps) == 50
            assert len(vals) == 50


class TestPersistenceEntropy:
    """Tests for persistence_entropy()."""

    def test_non_negative(self, simple_diagrams: list[np.ndarray]) -> None:
        """Entropy should be non-negative."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        for k in range(2):
            e = analyzer.persistence_entropy(k)
            assert e >= 0, f"H_{k} entropy = {e}"

    def test_empty_diagram(self) -> None:
        """Empty diagram should have zero entropy."""
        diagrams = [np.empty((0, 2)), np.empty((0, 2))]
        analyzer = PersistentHomologyAnalyzer(diagrams)
        assert analyzer.persistence_entropy(0) == 0.0

    def test_single_feature_zero_entropy(self) -> None:
        """Single feature has entropy 0 (log(1) = 0)."""
        diagrams = [np.array([[0.0, 1.0]])]
        analyzer = PersistentHomologyAnalyzer(diagrams)
        e = analyzer.persistence_entropy(0)
        assert abs(e) < 1e-10


class TestSignificantFeatures:
    """Tests for significant_features()."""

    def test_fewer_than_total(self, simple_diagrams: list[np.ndarray]) -> None:
        """Significant features should be a subset."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        sig = analyzer.significant_features(0, threshold="mean_lifetime")
        assert len(sig) <= len(simple_diagrams[0])

    def test_returns_correct_shape(self, simple_diagrams: list[np.ndarray]) -> None:
        """Should return (n, 2) array."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        sig = analyzer.significant_features(0)
        if len(sig) > 0:
            assert sig.shape[1] == 2


class TestSummaryStatistics:
    """Tests for summary_statistics()."""

    def test_all_keys_present(self, simple_diagrams: list[np.ndarray]) -> None:
        """Summary should contain all expected fields."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        summary = analyzer.summary_statistics()

        assert hasattr(summary, "n_features")
        assert hasattr(summary, "n_significant")
        assert hasattr(summary, "max_persistence")
        assert hasattr(summary, "mean_persistence")
        assert hasattr(summary, "persistence_entropy")
        assert hasattr(summary, "total_persistence")

    def test_max_persistence_values(self, simple_diagrams: list[np.ndarray]) -> None:
        """Max persistence of H_0 should be 10.0."""
        analyzer = PersistentHomologyAnalyzer(simple_diagrams)
        summary = analyzer.summary_statistics()
        assert abs(summary.max_persistence[0] - 10.0) < 0.01
