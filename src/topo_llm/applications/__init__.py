"""
Applications
=============

End-to-end applications built on the geometric and topological analysis:

- **Hallucination Detection**: Score text outputs for hallucination risk
  using curvature anomaly, topological isolation, information divergence,
  and density-based features.
- **Geodesic Retrieval**: Manifold-aware document retrieval using geodesic
  distance instead of cosine similarity.

Classes
-------
HallucinationDetector
    Unified hallucination detection combining all geometric features.
GeodesicRetrieval
    Geodesic-distance-based nearest neighbor retrieval.
"""

from __future__ import annotations

from topo_llm.applications.hallucination import HallucinationDetector
from topo_llm.applications.retrieval import GeodesicRetrieval

__all__ = ["HallucinationDetector", "GeodesicRetrieval"]
