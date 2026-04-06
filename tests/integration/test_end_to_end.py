"""End-to-end integration tests spanning multiple modules."""

from __future__ import annotations

import numpy as np
import pytest

from topo_llm.config import Config, load_config


class TestConfigToModuleInit:
    """Test that config values can initialize modules."""

    def test_load_config_returns_valid_config(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.riemannian.reduced_dim > 0
        assert cfg.topology.max_dimension >= 1

    def test_config_driven_metric_estimation(self, sphere_points):
        from topo_llm.riemannian.metric import MetricTensorEstimator

        cfg = load_config()
        est = MetricTensorEstimator(
            n_neighbors=min(cfg.riemannian.n_neighbors, len(sphere_points) - 1),
            regularization=cfg.riemannian.regularization,
        )
        est.fit(sphere_points)
        assert len(est.metric_tensors_) == len(sphere_points)


class TestFullGeometryPipeline:
    """Test extraction analysis → metric → curvature → topology pipeline."""

    def test_synthetic_full_pipeline(self, sphere_points):
        """Full pipeline on sphere: PCA → metric → curvature → topology features."""

        from topo_llm.extraction.layers import LayerAnalyzer
        from topo_llm.riemannian.connection import ChristoffelEstimator
        from topo_llm.riemannian.curvature import CurvatureAnalyzer
        from topo_llm.riemannian.metric import MetricTensorEstimator

        # Step 1: Intrinsic dim estimation
        intrinsic_dim = LayerAnalyzer.intrinsic_dimensionality(sphere_points, method="mle")
        assert 1 < intrinsic_dim < 4  # Should be ~2 for S^2

        # Step 2: Metric fitting
        metric = MetricTensorEstimator(n_neighbors=15, intrinsic_dim=2)
        metric.fit(sphere_points)

        # Step 3: Christoffel symbols
        christoffel = ChristoffelEstimator(metric, h=1e-3)

        # Step 4: Curvature
        curvature = CurvatureAnalyzer(metric, christoffel)
        S = curvature.scalar_curvature_at(0)
        assert np.isfinite(S)

    def test_high_dim_pipeline(self, low_dim_subspace):
        """Pipeline on high-dim data: PCA reduction → metric → curvature stats."""
        from sklearn.decomposition import PCA

        from topo_llm.riemannian.connection import ChristoffelEstimator
        from topo_llm.riemannian.curvature import CurvatureAnalyzer
        from topo_llm.riemannian.metric import MetricTensorEstimator

        # Reduce from 100D to 10D
        reduced = PCA(n_components=10).fit_transform(low_dim_subspace[:100])

        metric = MetricTensorEstimator(n_neighbors=15, intrinsic_dim=5)
        metric.fit(reduced)

        christoffel = ChristoffelEstimator(metric, h=1e-3)
        curvature = CurvatureAnalyzer(metric, christoffel)
        stats = curvature.curvature_statistics(show_progress=False)

        assert "mean" in stats
        assert "std" in stats
        assert np.isfinite(stats["mean"])


class TestInputValidation:
    """Test that validation catches bad inputs across modules."""

    def test_metric_rejects_1d_input(self):
        from topo_llm.riemannian.metric import MetricTensorEstimator

        est = MetricTensorEstimator()
        with pytest.raises(ValueError, match="2D"):
            est.fit(np.array([1, 2, 3]))
