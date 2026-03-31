# Topology of LLMs

**Riemannian Geometry, Persistent Topology, and Information Geometry of LLM Representation Spaces**

## Overview

This library provides tools to analyze the geometric and topological structure of
Large Language Model embedding spaces. It combines three mathematical frameworks:

1. **Riemannian Geometry** — Estimate metric tensors, compute curvature, and trace
   geodesics on the manifold of LLM embeddings
2. **Persistent Topology** — Compute persistent homology to identify topological
   features (connected components, loops, voids) that persist across scales
3. **Information Geometry** — Analyze Fisher information, entropy surfaces, and
   KL divergence geometry of output distributions

The primary application is **hallucination detection**: identifying when an LLM's
output lies in geometrically or topologically anomalous regions of its embedding space.

## Project Structure

```
src/topo_llm/
  extraction/       Embedding extraction from HuggingFace models     (PyTorch)
  riemannian/       Riemannian geometry engine                       (NumPy/SciPy)
  topology/         Persistent homology and TDA                      (ripser/gudhi)
  information/      Information geometry                             (NumPy/SciPy)
  applications/     Hallucination detection, geodesic retrieval
  visualization/    Plotting and paper figures                       (matplotlib)
```

Each subpackage has its own `README.md` with detailed module documentation.

## Installation

```bash
# Core only (NumPy/SciPy/scikit-learn)
pip install -e .

# With PyTorch for embedding extraction
pip install -e ".[torch]"

# With TDA libraries (ripser, gudhi, persim)
pip install -e ".[tda]"

# With visualization (matplotlib, seaborn, plotly)
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"

# Development (everything + pytest + ruff + mypy)
pip install -e ".[dev]"
```

Requires **Python 3.10+**. See `pyproject.toml` for the full dependency matrix.

## Quick Start

### 1. Extract Embeddings

```python
from topo_llm.extraction import EmbeddingExtractor, DatasetGenerator

# Generate a curated dataset
texts, labels = DatasetGenerator.semantic_categories(n_per_category=50)

# Extract embeddings from all layers
extractor = EmbeddingExtractor("gpt2", device="cpu")
embeddings = extractor.extract_dataset(texts, layers=[0, 6, 11])
# → {0: np.ndarray(500, 768), 6: np.ndarray(500, 768), 11: np.ndarray(500, 768)}
```

### 2. Analyze Layer Properties

```python
from topo_llm.extraction import LayerAnalyzer

for layer_idx, emb in embeddings.items():
    dim = LayerAnalyzer.intrinsic_dimensionality(emb, method="mle")
    aniso = LayerAnalyzer.compute_anisotropy(emb)
    print(f"Layer {layer_idx}: intrinsic_dim={dim:.1f}, "
          f"effective_rank={aniso['effective_rank']:.1f}")
```

### 3. Compute Riemannian Geometry

```python
from sklearn.decomposition import PCA
from topo_llm.riemannian import MetricTensorEstimator, CurvatureAnalyzer

# Mandatory dimensionality reduction (curvature is O(d^4))
emb_reduced = PCA(n_components=50).fit_transform(embeddings[11])

# Estimate metric tensors
metric_est = MetricTensorEstimator(n_neighbors=15)
metric_est.fit(emb_reduced)

# Compute curvature at each point
curv = CurvatureAnalyzer(metric_est)
stats = curv.curvature_statistics(emb_reduced)
print(f"Mean scalar curvature: {stats['scalar_mean']:.4f}")
print(f"Curvature std: {stats['scalar_std']:.4f}")
```

### 4. Compute Persistent Homology

```python
from topo_llm.topology import FiltrationBuilder, PersistentHomologyAnalyzer

# Build Rips filtration and compute homology
diagrams = FiltrationBuilder.vietoris_rips(emb_reduced, max_dim=2, max_edge=2.0)
analyzer = PersistentHomologyAnalyzer(diagrams)

# Betti numbers at a given scale
betti = analyzer.betti_numbers(epsilon=1.0)
print(f"H_0={betti[0]}, H_1={betti[1]}, H_2={betti[2]}")

# Persistence entropy (topological complexity)
entropy = analyzer.persistence_entropy()
print(f"Persistence entropy: {entropy:.3f}")
```

### 5. Score Hallucination Risk

```python
from topo_llm.applications import HallucinationDetector

detector = HallucinationDetector("gpt2", device="cpu")
detector.fit(reference_texts, layer=-1, reduced_dim=50)

score = detector.score("The Eiffel Tower is located in Berlin.")
print(f"Hallucination risk: {score.combined_score:.3f}")
print(f"  Curvature: {score.curvature_score:.3f}")
print(f"  Topological: {score.topological_score:.3f}")
print(f"  Information: {score.information_score:.3f}")
print(f"  Density: {score.density_score:.3f}")
```

### 6. Visualize Results

```python
from topo_llm.visualization import (
    plot_persistence_diagram,
    plot_curvature_field,
    set_paper_style,
)

# Exploratory
fig = plot_persistence_diagram(diagrams, title="GPT-2 Layer 11")
fig.savefig("persistence.png", dpi=150)

# Publication-quality
set_paper_style()
fig = plot_curvature_field(embeddings_2d, curvatures, title="Scalar Curvature")
fig.savefig("curvature_field.pdf")
```

## Implementation Status

| Phase | Module | Status | Tests |
|-------|--------|--------|-------|
| 1 | Embedding Extraction | Complete | 32 tests (14 fast, 18 slow) |
| 2 | Riemannian Geometry | Complete | 30 tests (all fast) |
| 3 | Topological Data Analysis | Complete | 31 tests (27 fast, 4 slow) |
| 4 | Information Geometry | Complete | 20 tests (all slow) |
| 5 | Applications | Complete | 8 tests (3 fast, 5 slow) |
| 6 | Visualization | Complete | — (visual output) |

**113 fast tests pass** on `pytest -m "not slow"`. Slow tests require model loading
(`sshleifer/tiny-gpt2`) and optional dependencies (ripser, persim).

## Testing

```bash
# Run all fast tests (no model loading, no optional deps)
pytest -m "not slow"

# Run all tests (requires torch, ripser, persim)
pytest

# Run tests for a specific phase
pytest tests/unit/riemannian/
pytest tests/unit/topology/

# With coverage
pytest --cov=topo_llm --cov-report=html
```

## Framework Boundaries

This project enforces strict separation of heavy dependencies:

| Framework | Where it's imported | Why |
|-----------|-------------------|-----|
| **PyTorch** | `extraction/` only | Model loading, forward pass |
| **JAX** | `riemannian/`, `information/` only | Autodiff for metric derivatives |
| **ripser/gudhi** | `topology/` only | Persistent homology computation |
| **matplotlib** | `visualization/` only | Plotting |

**NumPy** is the interchange format — all public APIs accept and return `np.ndarray`.
Lazy imports ensure that `import topo_llm.topology` never triggers PyTorch or JAX loading.

## Key Design Decisions

1. **Mandatory dimensionality reduction**: Full curvature tensors in d=768 are O(d^4).
   PCA to `reduced_dim=50` is required before any Riemannian or information geometry.
   Full-dim embeddings remain available for TDA (only needs pairwise distances).

2. **Memory-efficient extraction**: Layer-streaming processes one layer at a time,
   converts to NumPy immediately, and discards PyTorch tensors. Default float16 precision.

3. **Protocol-based TDA backends**: Auto-detects installed libraries (ripser, gudhi,
   giotto-tda) and selects the best available. Falls back to landmark complexes for N > 5000.

4. **Hallucination scoring**: Four independent geometric signals (curvature anomaly,
   topological isolation, inverse Fisher trace, Riemannian density) combined with
   configurable weights.

## Research Paper

This library supports a research paper targeting NeurIPS/ICML format.
Publication figures can be generated via `topo_llm.visualization.paper`:

```python
from topo_llm.visualization.paper import save_all_figures

save_all_figures(
    "figures/",
    intrinsic_dim={"layers": layers, "dims_by_model": dims},
    curvature={"layers": layers, "stats": curvature_stats},
    hallucination={"methods": names, "auroc": aurocs, "auprc": auprcs},
)
```

## License

MIT
