# CLAUDE.md — Developer Guide for topo-llm

## Project Overview

**topo-llm** is a Python library that analyzes the Riemannian geometry, persistent
topology, and information geometry of LLM embedding spaces. The primary application
is hallucination detection via geometric and topological features.

## Architecture

```
src/topo_llm/
  __init__.py        — Package root, version string
  types.py           — Shared dataclasses (EmbeddingResult, CurvatureResult, etc.)
  config.py          — Pydantic config hierarchy loaded from config/default.yaml
  device.py          — GPU/CPU/MPS auto-detection

  extraction/        — PyTorch-based embedding extraction from HuggingFace models
  riemannian/        — Riemannian geometry engine (metric tensors, curvature, geodesics)
  topology/          — Persistent homology and topological data analysis
  information/       — Information geometry (Fisher information, entropy, KL divergence)
  applications/      — Hallucination detection and geodesic retrieval
  visualization/     — Plotting and paper figure generation
```

Each subpackage has its own `README.md` with detailed module-level documentation.

## Key Design Rules

### Framework Boundaries

This is the most important architectural constraint. **Violations will cause import
failures on machines that don't have all frameworks installed.**

| Framework | Allowed in | Nowhere else |
|-----------|-----------|-------------|
| **PyTorch** (`torch`, `transformers`) | `extraction/` | Never in riemannian/, topology/, etc. |
| **JAX** (`jax`, `jaxlib`) | `riemannian/`, `information/` | Never in extraction/, topology/, etc. |
| **ripser/gudhi/giotto-tda** | `topology/` | Never elsewhere |
| **matplotlib** | `visualization/` | Never in core computation modules |

- **NumPy** is the interchange format. All public APIs accept and return `np.ndarray`.
- Each subpackage uses **lazy imports** — the heavy framework is only loaded when a
  function is actually called, never at module import time.
- Guard pattern used throughout:
  ```python
  def _require_torch():
      try:
          import torch
          return torch
      except ImportError:
          raise ImportError("PyTorch required. Install with: pip install topo-llm[torch]")
  ```

### Python Version

- Target: **Python 3.10+** (torch, jax, gudhi lack 3.14 wheels as of March 2026)
- Use `from __future__ import annotations` in **all files** for modern type syntax
- Use `X | Y` union syntax in annotations (enabled by `__future__` import)

### Code Standards

- **Type hints** on ALL function signatures (parameters and return types)
- **NumPy-style docstrings** on all public functions and classes
- **No `print()`** in library code — use `logging` module exclusively
- Each module must be **independently testable** without importing other subpackages
- Return concrete types, not `object` — use `np.ndarray`, not `Any`
- All functions that create matplotlib figures return `matplotlib.figure.Figure`

### Dimensionality Reduction

- Full curvature tensors in d=768 are **O(d^4)** — computationally impossible
- **Mandatory** PCA reduction to `reduced_dim` (default 50) before Riemannian/information geometry
- Full-dimensional embeddings remain available for TDA (only needs pairwise distances)
- Rule of thumb: `reduced_dim` should be ~2x the intrinsic dimensionality

### Memory Management

- **Layer-streaming extraction**: one layer at a time, convert to NumPy, discard tensor
- Save embeddings as `.npz`, load with `mmap_mode='r'`
- Default **float16** for extraction to halve memory
- **Sequential GPU**: extract with PyTorch first, `del model; torch.cuda.empty_cache()`, then JAX
- Never hold PyTorch and JAX models in GPU memory simultaneously

## Common Commands

```bash
# Install in development mode (core deps only, fast)
pip install -e .

# Install with specific optional dependencies
pip install -e ".[torch]"       # For embedding extraction
pip install -e ".[tda]"         # For persistent homology
pip install -e ".[viz]"         # For visualization
pip install -e ".[dev]"         # Everything + testing tools

# Run fast tests only (no model loading, no optional deps needed)
pytest -m "not slow"

# Run all tests (requires torch, ripser, persim)
pytest

# Run specific phase tests
pytest tests/unit/extraction/
pytest tests/unit/riemannian/
pytest tests/unit/topology/
pytest tests/unit/information/
pytest tests/unit/applications/

# Run with verbose output and stop on first failure
pytest -m "not slow" -x -v

# Run with coverage
pytest --cov=topo_llm --cov-report=html -m "not slow"

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/topo_llm/
```

## Git Commit Conventions

Commits follow **conventional commit** format:

```
<type>(<scope>): <short description>

<body — what and why, not how>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or modifying tests |
| `docs` | Documentation only |
| `chore` | Build, config, dependency changes |
| `perf` | Performance improvement |

### Scopes

`extraction`, `riemannian`, `topology`, `information`, `applications`, `viz`, `config`

### Examples

```
feat(extraction): add EmbeddingExtractor with batch processing
feat(riemannian): implement metric tensor estimation via local PCA
feat(topology): add persistent homology with ripser/gudhi backends
feat(information): implement Fisher information and entropy surfaces
feat(applications): add hallucination detector with four-signal scoring
feat(viz): add publication-quality figure generation for NeurIPS format
test(topology): add sphere and torus validation tests
fix(topology): replace np.trapz with np.trapezoid for NumPy 2.x
docs: add package READMEs and CLAUDE.md
chore: switch build backend from hatchling to setuptools
```

### Maximizing Commit History

To create a granular, meaningful git history, split work by subpackage:

```bash
# Scaffolding first
git add .gitignore pyproject.toml Makefile config/ CLAUDE.md README.md
git commit -m "chore: initialize project scaffolding with build config and dev guide"

# Core types
git add src/topo_llm/__init__.py src/topo_llm/types.py src/topo_llm/config.py src/topo_llm/device.py
git commit -m "feat(config): add shared types, Pydantic config hierarchy, and device detection"

# One commit per phase source + one per phase tests
git add src/topo_llm/extraction/
git commit -m "feat(extraction): add EmbeddingExtractor, DatasetGenerator, and LayerAnalyzer"

git add tests/unit/extraction/ tests/conftest.py tests/__init__.py tests/unit/__init__.py
git commit -m "test(extraction): add unit tests for extractor, datasets, and layers"

# ... repeat for each phase
```

## Testing Strategy

### Test Organization

```
tests/
  conftest.py           — Shared fixtures (synthetic manifolds, RNG seeds)
  unit/                 — Fast tests, synthetic data only
    extraction/         — Datasets, layers (fast); extractor (slow)
    riemannian/         — Metric, connection, curvature, geodesic, search
    topology/           — Filtration, homology, landscapes, distances, features
    information/        — Fisher, entropy, divergence (all slow)
    applications/       — Hallucination dataclass (fast); integration (slow)
  integration/          — Cross-module pipelines with tiny models
```

### Test Markers

| Marker | Meaning | When to use |
|--------|---------|-------------|
| `@pytest.mark.slow` | Requires model loading or optional deps | Real model tests, ripser/persim tests |
| `@pytest.mark.gpu` | Requires CUDA GPU | GPU-specific behavior tests |
| (no marker) | Fast, runs with core deps only | Synthetic data, pure math tests |

### Synthetic Validation Manifolds

Tests use known-geometry manifolds from `conftest.py`:

| Fixture | Shape | Known Properties |
|---------|-------|-----------------|
| `sphere_points` | 200 pts on S^2 (R=1) | Scalar curvature = 2, H_0=1, H_2=1 |
| `torus_points` | 300 pts on T^2 (R=3, r=1) | Mixed curvature, H_0=1, H_1=2, H_2=1 |
| `circle_points` | 200 pts on S^1 | H_0=1, H_1=1 |
| `flat_plane_points` | 200 pts in R^2 | All curvatures = 0 |
| `two_clusters` | 2×100 pts | H_0=2 |
| `low_dim_subspace` | 500 pts in 5D of R^100 | Intrinsic dim = 5 |

### Writing New Tests

```python
# Fast test (no marker needed)
def test_metric_positive_definite(self, sphere_points):
    metric_est = MetricTensorEstimator(n_neighbors=15)
    metric_est.fit(sphere_points)
    for g in metric_est.metrics_:
        eigenvalues = np.linalg.eigvalsh(g)
        assert np.all(eigenvalues > 0)

# Slow test (requires real model)
@pytest.mark.slow
def test_extract_real_model(self, tiny_model_name):
    extractor = EmbeddingExtractor(tiny_model_name, device="cpu")
    result = extractor.extract("Hello world")
    assert result.pooled_embeddings.shape[0] > 0
```

## Configuration

All hyperparameters live in `config/default.yaml` and are loaded via Pydantic models
in `src/topo_llm/config.py`.

### Config Hierarchy

```python
Config
  ├── ExtractionConfig    # model_name, batch_size, pooling, precision
  ├── RiemannianConfig    # n_neighbors, reduced_dim, geodesic steps
  ├── TopologyConfig      # max_dim, max_edge, n_landscapes, backend
  ├── InformationConfig   # n_perturbations, perturbation_scale
  └── HallucinationConfig # score weights, percentile thresholds
```

### Override with Environment Variables

Prefix with `TOPO_LLM_`:

```bash
TOPO_LLM_EXTRACTION__MODEL_NAME=gpt2-medium pytest -m slow
TOPO_LLM_RIEMANNIAN__REDUCED_DIM=30 python my_script.py
```

## Troubleshooting

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install -e ".[torch]"` |
| `ModuleNotFoundError: ripser` | TDA libs not installed | `pip install -e ".[tda]"` |
| `np.trapz` AttributeError | NumPy 2.x removed `trapz` | Already fixed: uses `np.trapezoid` |
| `CUDA out of memory` during extraction | Model too large | Use `precision="float16"`, reduce `batch_size` |
| Curvature values seem wrong | Input not reduced | Apply PCA to `reduced_dim=50` first |
| Slow geodesic computation | Too many points | Use `RiemannianSearch` with Euclidean pre-filtering |

### Verifying Installation

```python
import topo_llm
print(topo_llm.__version__)  # 0.1.0

# Check which optional deps are available
try:
    from topo_llm.extraction import EmbeddingExtractor
    print("PyTorch: available")
except ImportError:
    print("PyTorch: not installed")

try:
    from topo_llm.topology import FiltrationBuilder
    FiltrationBuilder.vietoris_rips.__module__  # will fail if ripser missing
    print("TDA: available")
except Exception:
    print("TDA: not installed")
```
