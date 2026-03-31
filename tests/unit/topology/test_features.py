"""
Tests for TopologicalFeatures.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.topology.features import TopologicalFeatures


@pytest.fixture
def sample_diagrams() -> list[np.ndarray]:
    """Sample persistence diagrams for testing."""
    h0 = np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 3.0]])
    h1 = np.array([[0.2, 0.8], [0.1, 2.0]])
    h2 = np.array([[0.5, 1.5]])
    return [h0, h1, h2]


class TestStatisticsVector:
    """Tests for statistics_vector()."""

    def test_shape(self, sample_diagrams: list[np.ndarray]) -> None:
        """Should return 30-dimensional vector."""
        v = TopologicalFeatures.statistics_vector(sample_diagrams)
        assert v.shape == (30,)

    def test_finite(self, sample_diagrams: list[np.ndarray]) -> None:
        """All values should be finite."""
        v = TopologicalFeatures.statistics_vector(sample_diagrams)
        assert np.all(np.isfinite(v))

    def test_empty_diagrams(self) -> None:
        """Empty diagrams should produce zero vector."""
        v = TopologicalFeatures.statistics_vector([np.empty((0, 2))])
        assert v.shape == (30,)

    def test_feature_count_correct(self, sample_diagrams: list[np.ndarray]) -> None:
        """First feature of H_0 block should be n_features = 3."""
        v = TopologicalFeatures.statistics_vector(sample_diagrams)
        assert v[0] == 3.0  # H_0 has 3 features


class TestPersistenceImage:
    """Tests for persistence_image()."""

    def test_shape(self, sample_diagrams: list[np.ndarray]) -> None:
        """Should return correct resolution."""
        img = TopologicalFeatures.persistence_image(
            sample_diagrams[0], resolution=(15, 15)
        )
        assert img.shape == (15, 15)

    def test_non_negative(self, sample_diagrams: list[np.ndarray]) -> None:
        """Image values should be non-negative (weighted Gaussians)."""
        img = TopologicalFeatures.persistence_image(sample_diagrams[0])
        assert np.all(img >= -1e-10)

    def test_empty_diagram(self) -> None:
        """Empty diagram produces zero image."""
        img = TopologicalFeatures.persistence_image(np.empty((0, 2)))
        assert np.all(img == 0)

    def test_not_all_zero(self, sample_diagrams: list[np.ndarray]) -> None:
        """Non-empty diagram should produce non-zero image."""
        img = TopologicalFeatures.persistence_image(sample_diagrams[0])
        assert img.sum() > 0


class TestCombinedFeatureVector:
    """Tests for combined_feature_vector()."""

    def test_includes_all(self, sample_diagrams: list[np.ndarray]) -> None:
        """With all features, vector should be longer than statistics alone."""
        v_stats = TopologicalFeatures.statistics_vector(sample_diagrams)
        v_combined = TopologicalFeatures.combined_feature_vector(sample_diagrams)
        assert len(v_combined) > len(v_stats)

    def test_finite(self, sample_diagrams: list[np.ndarray]) -> None:
        """All values should be finite."""
        v = TopologicalFeatures.combined_feature_vector(sample_diagrams)
        assert np.all(np.isfinite(v))

    def test_statistics_only(self, sample_diagrams: list[np.ndarray]) -> None:
        """With only statistics, should equal statistics_vector."""
        v = TopologicalFeatures.combined_feature_vector(
            sample_diagrams,
            include_statistics=True,
            include_landscapes=False,
            include_images=False,
        )
        v_stats = TopologicalFeatures.statistics_vector(sample_diagrams)
        np.testing.assert_array_almost_equal(v, v_stats)
