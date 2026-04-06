"""Tests for configuration loading and validation."""

from __future__ import annotations

import tempfile

import pytest
import yaml

from topo_llm.config import (
    Config,
    ExtractionConfig,
    HallucinationConfig,
    InformationConfig,
    RiemannianConfig,
    TopologyConfig,
    load_config,
)


class TestConfigDefaults:
    """Test that all config models have correct defaults."""

    def test_extraction_defaults(self):
        cfg = ExtractionConfig()
        assert cfg.model_name == "gpt2"
        assert cfg.pooling == "mean"
        assert cfg.precision == "float32"
        assert cfg.batch_size == 32

    def test_riemannian_defaults(self):
        cfg = RiemannianConfig()
        assert cfg.reduced_dim == 50
        assert cfg.n_neighbors == 50
        assert cfg.regularization == 1e-6

    def test_topology_defaults(self):
        cfg = TopologyConfig()
        assert cfg.max_dimension == 2
        assert cfg.max_edge_length is None
        assert cfg.backend == "auto"

    def test_information_defaults(self):
        cfg = InformationConfig()
        assert cfg.n_mc_samples == 100
        assert cfg.top_k_tokens == 100

    def test_hallucination_defaults(self):
        cfg = HallucinationConfig()
        assert cfg.reference_layer == -2
        assert cfg.score_weights["curvature"] == 0.25

    def test_top_level_defaults(self):
        cfg = Config()
        assert cfg.device == "auto"
        assert cfg.seed == 42
        assert cfg.log_level == "INFO"
        assert isinstance(cfg.extraction, ExtractionConfig)
        assert isinstance(cfg.riemannian, RiemannianConfig)


class TestConfigValidation:
    """Test Pydantic validation catches invalid values."""

    def test_invalid_pooling_rejected(self):
        with pytest.raises(Exception):  # ValidationError
            ExtractionConfig(pooling="invalid")

    def test_invalid_backend_rejected(self):
        with pytest.raises(Exception):
            TopologyConfig(backend="tensorflow")

    def test_invalid_precision_rejected(self):
        with pytest.raises(Exception):
            ExtractionConfig(precision="float8")

    def test_valid_custom_values(self):
        cfg = ExtractionConfig(model_name="gpt2-medium", batch_size=16)
        assert cfg.model_name == "gpt2-medium"
        assert cfg.batch_size == 16


class TestLoadConfig:
    """Test config loading from YAML files."""

    def test_load_nonexistent_returns_defaults(self):
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, Config)
        assert cfg.device == "auto"
        assert cfg.seed == 42

    def test_load_custom_yaml(self):
        custom = {"device": "cpu", "seed": 123}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom, f)
            f.flush()
            cfg = load_config(f.name)

        assert cfg.device == "cpu"
        assert cfg.seed == 123

    def test_load_nested_yaml(self):
        custom = {
            "extraction": {"model_name": "bert-base-uncased", "batch_size": 8},
            "riemannian": {"reduced_dim": 30},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom, f)
            f.flush()
            cfg = load_config(f.name)

        assert cfg.extraction.model_name == "bert-base-uncased"
        assert cfg.extraction.batch_size == 8
        assert cfg.riemannian.reduced_dim == 30

    def test_load_empty_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            cfg = load_config(f.name)

        assert isinstance(cfg, Config)
