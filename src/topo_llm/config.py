"""
Configuration management for topo-llm.

Loads configuration from YAML files and environment variables.
All hyperparameters are centralized here as Pydantic models.

Usage
-----
    from topo_llm.config import load_config

    config = load_config()                          # default config
    config = load_config("config/experiment.yaml")  # custom config
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Project root (two levels up from this file: src/topo_llm/config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class ExtractionConfig(BaseModel):
    """Configuration for embedding extraction."""

    model_name: str = "gpt2"
    layers: Literal["all"] | list[int] = "all"
    pooling: Literal["mean", "cls", "last", "max"] = "mean"
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    batch_size: int = 32
    max_seq_length: int = 512
    cache_dir: str = "data/embeddings"


class RiemannianConfig(BaseModel):
    """Configuration for Riemannian geometry computations."""

    reduced_dim: int = 50
    n_neighbors: int = 50
    regularization: float = 1e-6
    finite_diff_step: float = 1e-3
    geodesic_dt: float = 0.01
    geodesic_max_steps: int = 1000


class TopologyConfig(BaseModel):
    """Configuration for topological data analysis."""

    max_dimension: int = 2
    max_edge_length: float | None = None
    max_points: int = 2000
    backend: Literal["auto", "ripser", "gudhi", "giotto"] = "auto"
    significance_threshold: Literal["otsu", "percentile_90", "mean_lifetime"] = "otsu"


class InformationConfig(BaseModel):
    """Configuration for information geometry."""

    reduced_dim: int = 50
    n_mc_samples: int = 100
    top_k_tokens: int = 100
    perturbation_eps: float = 1e-3


class HallucinationConfig(BaseModel):
    """Configuration for hallucination detection."""

    reference_layer: int = -2
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "curvature": 0.25,
            "topological": 0.25,
            "information": 0.25,
            "density": 0.25,
        }
    )


class Config(BaseModel):
    """Top-level configuration for topo-llm.

    Parameters
    ----------
    extraction : ExtractionConfig
        Embedding extraction settings.
    riemannian : RiemannianConfig
        Riemannian geometry settings.
    topology : TopologyConfig
        TDA settings.
    information : InformationConfig
        Information geometry settings.
    hallucination : HallucinationConfig
        Hallucination detection settings.
    device : str
        Compute device: "auto", "cpu", "cuda", "mps".
    seed : int
        Random seed for reproducibility.
    log_level : str
        Logging level.
    """

    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    riemannian: RiemannianConfig = Field(default_factory=RiemannianConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    information: InformationConfig = Field(default_factory=InformationConfig)
    hallucination: HallucinationConfig = Field(default_factory=HallucinationConfig)
    device: str = "auto"
    seed: int = 42
    log_level: str = "INFO"


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str | Path | None
        Path to YAML config file. If None, loads the default config
        from ``config/default.yaml``. If the file doesn't exist,
        returns default values.

    Returns
    -------
    Config
        Validated configuration object.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if path.exists():
        logger.info("Loading config from %s", path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return Config(**raw)

    logger.info("Config file not found at %s, using defaults", path)
    return Config()
