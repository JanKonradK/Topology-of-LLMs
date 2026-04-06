"""Smoke tests for information geometry visualization functions."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.figure import Figure


@pytest.mark.slow
class TestPlotFisherHeatmap:
    """Test Fisher matrix heatmap plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.information import plot_fisher_heatmap

        matrix = np.random.rand(10, 10)
        matrix = matrix @ matrix.T  # Make symmetric
        fig = plot_fisher_heatmap(matrix)
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotEntropyScatter:
    """Test entropy scatter plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.information import plot_entropy_scatter

        pts = np.random.randn(50, 2)
        entropies = np.random.rand(50) * 5
        fig = plot_entropy_scatter(pts, entropies)
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotKLMatrix:
    """Test KL divergence matrix heatmap."""

    def test_returns_figure(self):
        from topo_llm.visualization.information import plot_kl_matrix

        matrix = np.random.rand(5, 5)
        matrix = (matrix + matrix.T) / 2  # Symmetric for JSD
        fig = plot_kl_matrix(matrix)
        assert isinstance(fig, Figure)

    def test_with_labels(self):
        from topo_llm.visualization.information import plot_kl_matrix

        matrix = np.random.rand(3, 3)
        fig = plot_kl_matrix(matrix, labels=["a", "b", "c"])
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotFisherTraceByLayer:
    """Test Fisher trace by layer plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.information import plot_fisher_trace_by_layer

        layers = list(range(12))
        traces = np.random.rand(12).tolist()
        fig = plot_fisher_trace_by_layer(layers, traces)
        assert isinstance(fig, Figure)
