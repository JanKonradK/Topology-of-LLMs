"""Smoke tests for persistence visualization functions."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.figure import Figure


@pytest.mark.slow
class TestPlotPersistenceDiagram:
    """Test persistence diagram scatter plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.persistence import plot_persistence_diagram

        dgm0 = np.array([[0.0, 0.5], [0.0, 1.0], [0.0, np.inf]])
        dgm1 = np.array([[0.3, 0.8], [0.5, 1.2]])
        fig = plot_persistence_diagram([dgm0, dgm1])
        assert isinstance(fig, Figure)

    def test_empty_diagrams(self):
        from topo_llm.visualization.persistence import plot_persistence_diagram

        dgm0 = np.array([]).reshape(0, 2)
        fig = plot_persistence_diagram([dgm0])
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotBarcode:
    """Test persistence barcode plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.persistence import plot_barcode

        dgm0 = np.array([[0.0, 0.5], [0.0, 1.0]])
        dgm1 = np.array([[0.3, 0.8]])
        fig = plot_barcode([dgm0, dgm1])
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotBettiCurve:
    """Test Betti curve plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.persistence import plot_betti_curve

        grid = np.linspace(0, 2, 100)
        curves = {
            0: (grid, np.ones(100)),
            1: (grid, np.maximum(0, 1 - grid)),
        }
        fig = plot_betti_curve(curves)
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestPlotPersistenceLandscape:
    """Test persistence landscape plot."""

    def test_returns_figure(self):
        from topo_llm.visualization.persistence import plot_persistence_landscape

        grid = np.linspace(0, 2, 100)
        landscapes = np.random.rand(3, 100)  # 3 landscape functions
        fig = plot_persistence_landscape(grid, landscapes)
        assert isinstance(fig, Figure)
