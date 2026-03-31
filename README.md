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
  extraction/       Embedding extraction from HuggingFace models
  riemannian/       Riemannian geometry engine
  topology/         Persistent homology and TDA
  information/      Information geometry
  applications/     Hallucination detection, geodesic retrieval
  visualization/    Plotting and paper figures
```

## Installation

```bash
# Core only (NumPy/SciPy)
pip install -e .

# With PyTorch for embedding extraction
pip install -e ".[torch]"

# With JAX for Riemannian geometry
pip install -e ".[jax]"

# With TDA libraries
pip install -e ".[tda]"

# Everything
pip install -e ".[all]"

# Development (everything + testing + linting)
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

```python
from topo_llm.extraction import EmbeddingExtractor, DatasetGenerator
from topo_llm.extraction import LayerAnalyzer

# Extract embeddings
extractor = EmbeddingExtractor("gpt2")
texts, labels = DatasetGenerator.semantic_categories(n_per_category=50)
embeddings = extractor.extract_dataset(texts, layers=[0, 6, 11])

# Analyze layer geometry
for layer_idx, emb in embeddings.items():
    dim = LayerAnalyzer.intrinsic_dimensionality(emb)
    aniso = LayerAnalyzer.compute_anisotropy(emb)
    print(f"Layer {layer_idx}: intrinsic dim={dim:.1f}, "
          f"effective rank={aniso['effective_rank']:.1f}")
```

## Implementation Phases

| Phase | Module | Status |
|-------|--------|--------|
| 1 | Embedding Extraction | In Progress |
| 2 | Riemannian Geometry | Planned |
| 3 | Topological Data Analysis | Planned |
| 4 | Information Geometry | Planned |
| 5 | Hallucination Detection | Planned |
| 6 | Visualization & Paper | Planned |

## Research Paper

This library supports a research paper targeting NeurIPS/ICML format (~10-12 pages).
See the paper outline in the project specification for details on planned experiments
and expected results.

## License

MIT
