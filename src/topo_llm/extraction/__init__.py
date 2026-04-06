"""
Embedding Extraction
====================

Extract hidden-layer embeddings from HuggingFace transformer models.

This is the only subpackage that imports PyTorch. All outputs are
returned as NumPy arrays for downstream consumption by the geometry,
topology, and information modules.

Classes
-------
EmbeddingExtractor
    Load a model and extract per-layer embeddings for text inputs.
DatasetGenerator
    Generate curated text datasets for geometric/topological experiments.
LayerAnalyzer
    Intrinsic dimensionality, anisotropy, and inter-layer similarity.
"""

from __future__ import annotations

from topo_llm.extraction.datasets import DatasetGenerator
from topo_llm.extraction.extractor import EmbeddingExtractor
from topo_llm.extraction.layers import LayerAnalyzer

__all__ = ["EmbeddingExtractor", "DatasetGenerator", "LayerAnalyzer"]
