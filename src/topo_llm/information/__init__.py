"""
Information Geometry
====================

Analyze the information-geometric structure of LLM output distributions:
Fisher information, entropy surfaces, and KL divergence geometry.

Uses JAX for automatic differentiation where needed.

Classes
-------
FisherInformationEstimator
    Estimate the Fisher Information Matrix of LLM output distributions.
EntropySurface
    Compute entropy landscapes over embedding space.
KLGeometry
    KL divergence and Jensen-Shannon distance between distributions.
"""

from __future__ import annotations

from topo_llm.information.divergence import KLGeometry
from topo_llm.information.entropy import EntropySurface
from topo_llm.information.fisher import FisherInformationEstimator

__all__ = [
    "FisherInformationEstimator",
    "EntropySurface",
    "KLGeometry",
]
