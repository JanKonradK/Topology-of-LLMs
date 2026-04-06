"""
Topological Data Analysis
=========================

Compute persistent homology of embedding point clouds, extract
topological feature vectors, and measure distances between
persistence diagrams.

Supports multiple backends (ripser, gudhi, giotto-tda) via a
unified interface. The backend is auto-selected based on what
is installed.

Classes
-------
FiltrationBuilder
    Build Vietoris-Rips and Alpha complex filtrations.
PersistentHomologyAnalyzer
    Betti numbers, persistence entropy, and significance filtering.
PersistenceLandscape
    Functional summaries of persistence diagrams for statistics.
DiagramDistances
    Wasserstein and bottleneck distances between persistence diagrams.
TopologicalFeatures
    Fixed-size feature vectors from persistence diagrams for ML.
"""

from __future__ import annotations

from topo_llm.topology.distances import DiagramDistances
from topo_llm.topology.features import TopologicalFeatures
from topo_llm.topology.filtration import FiltrationBuilder
from topo_llm.topology.homology import PersistentHomologyAnalyzer
from topo_llm.topology.landscapes import PersistenceLandscape

__all__ = [
    "FiltrationBuilder",
    "PersistentHomologyAnalyzer",
    "PersistenceLandscape",
    "DiagramDistances",
    "TopologicalFeatures",
]
