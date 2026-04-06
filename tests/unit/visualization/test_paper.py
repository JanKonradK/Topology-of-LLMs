"""Smoke tests for publication figure generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.figure import Figure


@pytest.mark.slow
class TestSetPaperStyle:
    """Test paper style RC params configuration."""

    def test_runs_without_error(self):
        from topo_llm.visualization.paper import set_paper_style

        set_paper_style()  # Should not raise


@pytest.mark.slow
class TestFigureIntrinsicDimension:
    """Test intrinsic dimension figure generator."""

    def test_returns_figure(self):
        from topo_llm.visualization.paper import figure_intrinsic_dimension

        layers = list(range(12))
        dims = {"gpt2": np.random.rand(12).tolist()}
        fig = figure_intrinsic_dimension(layers, dims)
        assert isinstance(fig, Figure)

    def test_multiple_models(self):
        from topo_llm.visualization.paper import figure_intrinsic_dimension

        layers = list(range(6))
        dims = {
            "gpt2": np.random.rand(6).tolist(),
            "bert": np.random.rand(6).tolist(),
        }
        fig = figure_intrinsic_dimension(layers, dims)
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestFigureCurvatureProfile:
    """Test curvature profile figure generator."""

    def test_returns_figure(self):
        from topo_llm.visualization.paper import figure_curvature_profile

        layers = list(range(12))
        stats = {
            "gpt2": {
                "mean": np.random.randn(12).tolist(),
                "std": np.abs(np.random.randn(12)).tolist(),
            }
        }
        fig = figure_curvature_profile(layers, stats)
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestFigureHallucinationComparison:
    """Test hallucination comparison bar chart."""

    def test_returns_figure(self):
        from topo_llm.visualization.paper import figure_hallucination_comparison

        methods = ["Ours", "Cosine", "Entropy"]
        auroc = [0.85, 0.70, 0.65]
        auprc = [0.80, 0.60, 0.55]
        fig = figure_hallucination_comparison(methods, auroc, auprc)
        assert isinstance(fig, Figure)


@pytest.mark.slow
class TestSaveAllFigures:
    """Test batch figure saving."""

    def test_saves_to_directory(self):
        from topo_llm.visualization.paper import save_all_figures

        with tempfile.TemporaryDirectory() as tmpdir:
            layers = list(range(6))
            intrinsic_dim = {
                "layers": layers,
                "dims_by_model": {"gpt2": np.random.rand(6).tolist()},
            }
            saved = save_all_figures(tmpdir, intrinsic_dim=intrinsic_dim)
            assert len(saved) > 0
            for path in saved:
                assert Path(path).exists()

    def test_empty_data_returns_empty(self):
        from topo_llm.visualization.paper import save_all_figures

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_all_figures(tmpdir)
            assert isinstance(saved, list)
