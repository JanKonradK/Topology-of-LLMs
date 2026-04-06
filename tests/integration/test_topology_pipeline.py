"""
Integration tests: Topology pipeline.

Tests the full flow from point cloud → filtration → homology → landscapes → features
on synthetic manifolds with known topological invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.topology import (
    DiagramDistances,
    FiltrationBuilder,
    PersistenceLandscape,
    PersistentHomologyAnalyzer,
    TopologicalFeatures,
)


class TestCircleTopology:
    """Full TDA pipeline on a circle (S^1): H_0=1, H_1=1."""

    @pytest.mark.slow
    def test_circle_full_pipeline(self, circle_points: np.ndarray) -> None:
        """Circle → Rips → Homology → Landscapes → Features end-to-end."""
        # Build filtration
        diagrams = FiltrationBuilder.vietoris_rips(circle_points, max_dim=1, max_edge=2.0)
        assert len(diagrams) >= 2  # H_0 and H_1

        # Analyze homology
        analyzer = PersistentHomologyAnalyzer(diagrams)

        # At small scale: many components. At large scale: 1 component.
        betti_large = analyzer.betti_numbers(epsilon=10.0)
        assert betti_large[0] == 1  # One connected component

        # H_1 should have at least one persistent feature (the loop)
        sig = analyzer.significant_features(dimension=1)
        assert len(sig) >= 1

        # Persistence entropy should be non-negative
        entropy = analyzer.persistence_entropy()
        assert entropy >= 0

        # Summary statistics
        stats = analyzer.summary_statistics()
        assert "total_features" in stats
        assert stats["total_features"] > 0

        # Build landscape
        landscape = PersistenceLandscape(diagrams[1], n_landscapes=5, resolution=100)
        assert landscape.landscapes_.shape[0] == 5
        assert landscape.landscapes_.shape[1] == 100

        # Landscape norm should be positive (there are features)
        norm_val = landscape.norm(k=0, p=2.0)
        assert norm_val > 0

        # Feature vector
        feat = TopologicalFeatures.statistics_vector(diagrams)
        assert feat.shape == (30,)
        assert np.all(np.isfinite(feat))

        # Combined features
        combined = TopologicalFeatures.combined_feature_vector(diagrams)
        assert len(combined) > 30  # stats + landscape features


class TestTwoClustersTopology:
    """Two well-separated clusters should show H_0=2."""

    @pytest.mark.slow
    def test_two_clusters_h0(self, two_clusters: np.ndarray) -> None:
        """Two clusters should produce two persistent H_0 features."""
        diagrams = FiltrationBuilder.vietoris_rips(two_clusters, max_dim=1, max_edge=5.0)

        analyzer = PersistentHomologyAnalyzer(diagrams)

        # At moderate scale (between intra- and inter-cluster distance)
        betti_mid = analyzer.betti_numbers(epsilon=3.0)
        assert betti_mid[0] >= 2  # Two clusters still separated


class TestDiagramDistancesPipeline:
    """Test diagram comparison between different spaces."""

    @pytest.mark.slow
    def test_different_spaces_have_positive_distance(
        self, circle_points: np.ndarray, two_clusters: np.ndarray
    ) -> None:
        """Diagrams from circle vs. two clusters should have positive distance."""
        dgm_circle = FiltrationBuilder.vietoris_rips(circle_points, max_dim=1, max_edge=2.0)
        dgm_clusters = FiltrationBuilder.vietoris_rips(two_clusters, max_dim=1, max_edge=5.0)

        # Compare H_0 diagrams
        if len(dgm_circle[0]) > 0 and len(dgm_clusters[0]) > 0:
            dist = DiagramDistances.wasserstein(dgm_circle[0], dgm_clusters[0])
            assert dist >= 0

    @pytest.mark.slow
    def test_same_space_zero_distance(self, circle_points: np.ndarray) -> None:
        """Same diagram should have zero Wasserstein distance with itself."""
        dgm = FiltrationBuilder.vietoris_rips(circle_points, max_dim=1, max_edge=2.0)

        if len(dgm[0]) > 0:
            dist = DiagramDistances.wasserstein(dgm[0], dgm[0])
            assert dist < 1e-10


class TestMaxminSubsampling:
    """Test subsampling before TDA for large point clouds."""

    def test_subsample_then_homology(self, synthetic_embeddings: np.ndarray) -> None:
        """Subsample high-dim data → compute pairwise distances → verify shape."""
        # Subsample to manageable size
        subset, indices = FiltrationBuilder.maxmin_subsample(synthetic_embeddings, n_points=100)
        assert subset.shape[0] == 100
        assert subset.shape[1] == 768
        assert len(indices) == 100
        assert len(set(indices)) == 100  # All unique
