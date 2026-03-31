# applications — Hallucination Detection & Geodesic Retrieval

## Purpose

End-to-end applications that combine all geometric, topological, and information-
geometric features into practical tools.

## Hallucination Detector

The `HallucinationDetector` scores text outputs for hallucination risk by combining
four geometric/topological signals:

| Score | What it Measures | Intuition |
|-------|-----------------|-----------|
| Curvature | Local curvature anomaly vs. reference | Hallucinations land in geometrically unusual regions |
| Topological | Distance from persistent H_0 component centers | Hallucinations are topologically isolated |
| Information | Inverse Fisher trace | Hallucinations have low information content |
| Density | Riemannian-corrected local density | Hallucinations are in sparse regions |

The combined score H(x) is a weighted average, with weights learned from labeled
data (if available) or equal weights (unsupervised).

## Geodesic Retrieval

`GeodesicRetrieval` is a drop-in replacement for cosine-similarity-based retrieval
that uses geodesic distance on the embedding manifold. This accounts for the curved
geometry of the embedding space, potentially retrieving more semantically relevant
documents than flat-space methods.

## Evaluation

Both applications include built-in benchmarking:
- Hallucination: AUROC, AUPRC, F1 against baselines (entropy, density, cosine)
- Retrieval: Recall@K comparison (geodesic vs. cosine vs. Euclidean)
