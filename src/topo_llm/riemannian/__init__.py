"""
Riemannian Geometry Engine
==========================

Estimate the Riemannian structure of embedding point clouds:
metric tensors, Christoffel symbols, curvature tensors, and geodesics.

Uses JAX for automatic differentiation of the interpolated metric field.
JAX is lazily imported — this package can be imported without JAX installed,
but computation methods will raise ImportError.

Classes
-------
MetricTensorEstimator
    Estimate local Riemannian metric from point cloud neighborhoods.
ChristoffelEstimator
    Compute Christoffel symbols via finite differences of the metric.
CurvatureAnalyzer
    Riemann, Ricci, and scalar curvature from Christoffel symbols.
GeodesicSolver
    Solve geodesic equations via RK4 integration.
RiemannianSearch
    Geodesic-aware nearest neighbor search.
"""

from __future__ import annotations

from topo_llm.riemannian.connection import ChristoffelEstimator
from topo_llm.riemannian.curvature import CurvatureAnalyzer
from topo_llm.riemannian.geodesic import GeodesicSolver
from topo_llm.riemannian.metric import MetricTensorEstimator
from topo_llm.riemannian.search import RiemannianSearch

__all__ = [
    "MetricTensorEstimator",
    "ChristoffelEstimator",
    "CurvatureAnalyzer",
    "GeodesicSolver",
    "RiemannianSearch",
]
