# CLAUDE.md — Developer Guide for topo-llm

## Project Overview

**topo-llm** is a Python library that analyzes the Riemannian geometry, persistent
topology, and information geometry of LLM embedding spaces. The primary application
is hallucination detection via geometric and topological features.

## Architecture

```
src/topo_llm/
  extraction/   — PyTorch-based embedding extraction from HuggingFace models
  riemannian/   — Riemannian geometry engine (metric tensors, curvature, geodesics)
  topology/     — Persistent homology and topological data analysis
  information/  — Information geometry (Fisher information, entropy, KL divergence)
  applications/ — Hallucination detection and geodesic retrieval
  visualization/ — Plotting and paper figure generation
```

## Key Design Rules

### Framework Boundaries
- **PyTorch** is ONLY imported inside `extraction/`. Never elsewhere.
- **JAX** is ONLY imported inside `riemannian/` and `information/`. Never elsewhere.
- **NumPy** is the interchange format. All public APIs accept and return `np.ndarray`.
- Each subpackage uses lazy imports to avoid loading heavy frameworks unnecessarily.

### Python Version
- Target: Python 3.10+ (torch, jax, gudhi lack 3.14 wheels as of March 2026)
- Use `from __future__ import annotations` in all files for modern type syntax

### Code Standards
- Type hints on ALL functions
- NumPy-style docstrings on all public functions/classes
- No `print()` in library code — use `logging` module
- Each module must be independently testable

### Dimensionality Reduction
- Full curvature tensors in d=768 are O(d^4) — computationally impossible
- Mandatory PCA reduction to `reduced_dim` (default 50) before Riemannian/information geometry
- Full-dimensional embeddings remain available for TDA (only needs pairwise distances)

### Memory Management
- Layer-streaming extraction: one layer at a time, convert to NumPy, discard tensor
- Save embeddings as `.npz`, load with `mmap_mode='r'`
- Default float16 for extraction to halve memory
- Sequential GPU: extract with PyTorch first, `del model`, then JAX

## Common Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run fast tests only (skip model loading)
pytest -m "not slow"

# Run specific phase tests
pytest tests/unit/extraction/
pytest tests/unit/riemannian/
pytest tests/unit/topology/
pytest tests/unit/information/

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/topo_llm/
```

## Git Commit Conventions

Commits follow conventional commit format:

```
<type>(<scope>): <short description>

<body — what and why, not how>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`
Scopes: `extraction`, `riemannian`, `topology`, `information`, `applications`, `viz`, `config`

Examples:
- `feat(extraction): add EmbeddingExtractor with batch processing`
- `feat(riemannian): implement metric tensor estimation via local PCA`
- `test(topology): add sphere and torus validation tests`
- `docs: add package READMEs and CLAUDE.md`

## Testing Strategy

- **Unit tests** (`tests/unit/`): Fast, synthetic data only. Each runs < 1 second.
- **Integration tests** (`tests/integration/`): Cross-module pipelines with tiny models.
- **Markers**: `@pytest.mark.slow` for real model tests, `@pytest.mark.gpu` for CUDA.
- **Synthetic validation**: Sphere (curvature=2/R^2), torus (mixed curvature), circle (H_1=1).
- Use `sshleifer/tiny-gpt2` for tests requiring a real model.

## Configuration

All hyperparameters live in `config/default.yaml` and are loaded via Pydantic models
in `src/topo_llm/config.py`. Override with environment variables prefixed `TOPO_LLM_`.
