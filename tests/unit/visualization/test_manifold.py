"""Smoke tests for manifold visualization functions."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.figure import Figure


@pytest.mark.slow
class TestPlotCurvatureField:
    """Test curvature field scatter plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.manifold import plot_curvature_field

        pts = np.random.randn(50, 2)
        curvatures = np.random.randn(50)
        fig = plot_curvature_field(pts, curvatures)
        assert isinstance(fig, Figure)

    def test_custom_cmap(self):
        from topo_llm.visualization.manifold import plot_curvature_field

        pts = np.random.randn(30, 2)
        curvatures = np.random.randn(30)
        fig = plot_curvature_field(pts, curvatures, cmap="viridis")
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotGeodesic:
    """Test geodesic path overlay plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.manifold import plot_geodesic

        pts = np.random.randn(50, 2)
        geodesic = np.random.randn(10, 2)
        fig = plot_geodesic(pts, geodesic)
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotMetricEllipses:
    """Test metric tensor ellipse plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.manifold import plot_metric_ellipses

        pts = np.random.randn(50, 2)
        metrics = [np.eye(2) + 0.1 * np.random.randn(2, 2) for _ in range(50)]
        # Make symmetric positive definite
        metrics = [m @ m.T + 0.1 * np.eye(2) for m in metrics]
        fig = plot_metric_ellipses(pts, metrics, n_ellipses=10)
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotLayerCurvatureProfile:
    """Test layer curvature profile plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.manifold import plot_layer_curvature_profile

        layers = list(range(12))
        means = np.random.randn(12).tolist()
        stds = np.abs(np.random.randn(12)).tolist()
        fig = plot_layer_curvature_profile(layers, means, stds)
        assert isinstance(fig, Figure)
