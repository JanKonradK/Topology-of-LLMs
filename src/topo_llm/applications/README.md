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
| Information | Fisher trace + entropy gradient + JSD | Hallucinations have unstable, low-information distributions |
| Density | Riemannian-corrected local density | Hallucinations are in sparse regions |

The combined score H(x) is a weighted average, with weights learned from labeled
data (if available) or equal weights (unsupervised).

### Scoring Signals in Detail

**1. Curvature Anomaly Score**

Computes the scalar curvature at the query embedding's location on the Riemannian
manifold. High curvature relative to the reference corpus indicates the embedding
lies in a geometrically unusual region — a sign that the model's internal
representation is strained.

Score = percentile rank of |S(x) - S_mean| / S_std against reference curvatures.

**2. Topological Isolation Score**

Computes persistent homology (H_0) of the reference corpus and identifies cluster
centers as the generators of persistent connected components. The query's distance
to the nearest component center, normalized by the component's persistence, measures
how topologically isolated the query is from known-good text distributions.

Score = 1 - exp(-distance / persistence_scale).

**3. Information Geometry Score**

Combines three information-theoretic signals computed via the `information/` subpackage:

- **Fisher Trace Anomaly (weight 0.5)**: The trace of the Fisher Information Matrix
  measures how sensitive the model's output distribution is to embedding perturbations.
  Low Fisher trace = the model is insensitive = uncertain = hallucination risk.
  Score = 1 - Φ((trace - μ_ref) / σ_ref) where Φ is the standard normal CDF.

- **Entropy Gradient Magnitude (weight 0.3)**: The gradient of the entropy surface
  at the query point. Large gradients indicate the model's confidence is locally
  unstable — small perturbations change the entropy significantly.
  Score = Φ((|∇H| - μ_ref) / σ_ref).

- **JSD from Nearest Reference (weight 0.2)**: Jensen-Shannon divergence between
  the query's output distribution and the nearest reference text's distribution.
  Large JSD = the query produces a distribution unlike any reference text.
  Score = min(JSD / JSD_max, 1.0).

**4. Riemannian Density Score**

Estimates local density using k-NN distances corrected by the metric tensor
determinant (the Riemannian volume element). This accounts for the non-Euclidean
geometry: points may be close in Euclidean space but far on the manifold.

Score = 1 - percentile rank of corrected density among reference points.

### Usage Example

```python
from topo_llm.applications import HallucinationDetector

# Fit on reference corpus (known-good texts)
detector = HallucinationDetector("gpt2", device="cpu")
detector.fit(
    reference_texts=["Paris is the capital of France.", ...],
    layer=-1,
    reduced_dim=50,
)

# Score new text
score = detector.score("The Eiffel Tower is located in Berlin.")
print(f"Hallucination risk: {score.combined_score:.3f}")
print(f"  Curvature:    {score.curvature_score:.3f}")
print(f"  Topological:  {score.topological_score:.3f}")
print(f"  Information:  {score.information_score:.3f}")
print(f"  Density:      {score.density_score:.3f}")

# Evaluate against labeled data
results = detector.evaluate(
    test_texts=["Claim A", "Claim B", ...],
    labels=[1, 0, ...],  # 1 = hallucination, 0 = factual
)
print(f"AUROC: {results['auroc']:.3f}, AUPRC: {results['auprc']:.3f}")
```

### Skipping Information Geometry

The information geometry signal requires model forward passes during `fit()` (for
Fisher/entropy/JSD reference statistics). To skip this for faster fitting:

```python
detector.fit(reference_texts, layer=-1, skip_information=True)
```

This falls back to raw entropy for the information signal.

## Geodesic Retrieval

`GeodesicRetrieval` is a drop-in replacement for cosine-similarity-based retrieval
that uses geodesic distance on the embedding manifold. This accounts for the curved
geometry of the embedding space, potentially retrieving more semantically relevant
documents than flat-space methods.

### Usage Example

```python
from topo_llm.applications import GeodesicRetrieval

retriever = GeodesicRetrieval("gpt2", device="cpu")
retriever.fit(corpus_texts, layer=-1, reduced_dim=50)

results = retriever.query("What is the capital of France?", top_k=5)
for r in results:
    print(f"  [{r['rank']}] score={r['geodesic_distance']:.4f}: {r['text'][:80]}")
```

## Evaluation

Both applications include built-in benchmarking:
- Hallucination: AUROC, AUPRC, F1 against baselines (entropy, density, cosine)
- Retrieval: Recall@K comparison (geodesic vs. cosine vs. Euclidean)
