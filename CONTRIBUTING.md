# Contributing to topo-llm

## Development Setup

```bash
# Clone and install
git clone <repo-url>
cd "Topology of LLMs"
pip install -e ".[dev]"

# Verify installation
pytest -m "not slow" -x
```

## Development Workflow

### 1. Branch Naming

```
feat/<scope>-<description>     # New features
fix/<scope>-<description>      # Bug fixes
refactor/<scope>-<description> # Restructuring
test/<scope>-<description>     # Test additions
docs/<description>             # Documentation
```

Examples:
```
feat/riemannian-sectional-curvature
fix/topology-numpy2-trapz
test/information-fisher-validation
docs/api-reference
```

### 2. Making Changes

1. Create a feature branch from `master`
2. Make changes in small, focused commits (see commit conventions below)
3. Run fast tests before every commit: `pytest -m "not slow" -x`
4. Run linting: `ruff check src/ tests/`
5. Push and open a PR

### 3. Commit Conventions

We use **conventional commits**. Every commit message follows this format:

```
<type>(<scope>): <short description in imperative mood>

<optional body — explain what changed and why>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

#### Types

| Type | Use for |
|------|---------|
| `feat` | New functionality (new class, new method, new module) |
| `fix` | Bug fix (correcting incorrect behavior) |
| `refactor` | Code change that doesn't alter behavior |
| `test` | Adding or updating tests |
| `docs` | Documentation changes |
| `chore` | Build system, CI, dependencies, tooling |
| `perf` | Performance improvement |

#### Scopes

| Scope | Subpackage |
|-------|-----------|
| `extraction` | `src/topo_llm/extraction/` |
| `riemannian` | `src/topo_llm/riemannian/` |
| `topology` | `src/topo_llm/topology/` |
| `information` | `src/topo_llm/information/` |
| `applications` | `src/topo_llm/applications/` |
| `viz` | `src/topo_llm/visualization/` |
| `config` | `src/topo_llm/config.py`, `config/default.yaml` |

Omit scope for cross-cutting changes: `docs: update README with installation guide`

#### Good Commit Messages

```
feat(riemannian): add sectional curvature computation

Implements K(v1, v2) = R(v1,v2,v2,v1) / (|v1|^2|v2|^2 - <v1,v2>^2)
using the existing Riemann tensor. Needed for the curvature profile
paper figure.
```

```
fix(topology): replace np.trapz with np.trapezoid for NumPy 2.x

np.trapz was removed in NumPy 2.0. The replacement np.trapezoid has
identical behavior. Affects landscape integration and norm computation.
```

```
test(riemannian): add sphere curvature validation

Verifies scalar curvature ≈ 2.0 on unit sphere (200 sample points,
rtol=0.3). This is the primary correctness check for the curvature
pipeline.
```

#### Bad Commit Messages

```
# Too vague
fix: fixed stuff

# Implementation details instead of intent
feat: add for loop to iterate over layers

# Missing type/scope
updated the metric tensor code
```

### 4. Splitting Work into Commits

**One logical change per commit.** Here's how to split a feature across commits:

```bash
# Source code first
git add src/topo_llm/riemannian/metric.py
git commit -m "feat(riemannian): implement MetricTensorEstimator with local PCA"

# Tests second
git add tests/unit/riemannian/test_metric.py
git commit -m "test(riemannian): add metric tensor validation (positive definite, sphere)"

# Documentation third
git add src/topo_llm/riemannian/README.md
git commit -m "docs(riemannian): document metric estimation pipeline"
```

For a full phase, aim for **3-5 commits**:
1. Core source module(s)
2. Unit tests
3. Documentation / README
4. Config changes (if any)

## Code Style

### Module Template

Every new module should follow this structure:

```python
"""
Short module description.

Longer explanation of what this module does and how it fits
into the overall pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Heavy imports only for type checking, never at runtime
    pass

logger = logging.getLogger(__name__)


class MyAnalyzer:
    """One-line summary.

    Extended description of the class, its purpose, and how it
    relates to the mathematical concepts.

    Parameters
    ----------
    param1 : int
        Description.
    param2 : float
        Description with default.

    Examples
    --------
    >>> analyzer = MyAnalyzer(param1=10)
    >>> analyzer.fit(data)
    >>> result = analyzer.compute(point)
    """

    def __init__(self, param1: int, param2: float = 1.0) -> None:
        self.param1 = param1
        self.param2 = param2

    def fit(self, data: np.ndarray) -> None:
        """Fit the analyzer to a point cloud.

        Parameters
        ----------
        data : np.ndarray
            Input data, shape ``(N, D)``.
        """
        ...

    def compute(self, point: np.ndarray) -> np.ndarray:
        """Compute the thing at a single point.

        Parameters
        ----------
        point : np.ndarray
            Query point, shape ``(D,)``.

        Returns
        -------
        np.ndarray
            Result array, shape ``(d, d)``.
        """
        ...
```

### Lazy Import Pattern

For optional heavy dependencies:

```python
def _require_torch():
    """Import torch or raise helpful error."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch required for this module. "
            "Install with: pip install topo-llm[torch]"
        )
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Processing point %d/%d", i, n)      # Noisy progress
logger.info("Fitted metric on %d points", n)       # Key milestones
logger.warning("Matrix near-singular, regularizing")  # Recoverable issues
logger.error("Failed to converge after %d iterations", max_iter)  # Failures
```

## Testing Guide

### Running Tests

```bash
# Fast tests (always run before committing)
pytest -m "not slow" -x

# Verbose with test names
pytest -m "not slow" -x -v

# Specific module
pytest tests/unit/riemannian/test_curvature.py -v

# Single test
pytest tests/unit/riemannian/test_curvature.py::TestScalarCurvature::test_sphere_positive -v

# With coverage
pytest --cov=topo_llm --cov-report=term-missing -m "not slow"
```

### Test Structure

```python
"""Tests for SomeModule."""

from __future__ import annotations

import numpy as np
import pytest


class TestFeatureName:
    """Group related tests in a class."""

    def test_basic_behavior(self, fixture_name):
        """Test the happy path."""
        result = do_thing(fixture_name)
        assert result.shape == (10, 3)

    def test_edge_case(self):
        """Test boundary conditions."""
        result = do_thing(np.array([]))
        assert len(result) == 0

    def test_known_answer(self, sphere_points):
        """Validate against analytical result."""
        curvature = compute_curvature(sphere_points)
        # Sphere of radius R has scalar curvature = 2/R^2
        np.testing.assert_allclose(curvature, 2.0, rtol=0.3)

    @pytest.mark.slow
    def test_with_real_model(self, tiny_model_name):
        """Test requiring model loading."""
        extractor = EmbeddingExtractor(tiny_model_name)
        result = extractor.extract("Hello")
        assert result.pooled_embeddings.ndim == 1
```

### Fixture Guidelines

- Define shared fixtures in `tests/conftest.py`
- Use `@pytest.fixture` with descriptive names
- Set random seeds for reproducibility: `np.random.default_rng(42)`
- Keep synthetic data small (100-500 points) for speed
- Use known-geometry manifolds (sphere, torus, plane) for validation

## Architecture Rules

### Adding a New Module

1. Create the module in the appropriate subpackage
2. Add exports to the subpackage `__init__.py`
3. Write tests in the corresponding `tests/unit/<subpackage>/` directory
4. Update the subpackage `README.md`
5. If it adds a new dependency, update `pyproject.toml` optional deps

### Cross-Package Communication

Subpackages communicate **only through NumPy arrays and dataclasses from `types.py`**:

```
extraction/ ──(np.ndarray)──> riemannian/
extraction/ ──(np.ndarray)──> topology/
extraction/ ──(np.ndarray)──> information/

riemannian/  ──(CurvatureResult)──> applications/
topology/    ──(np.ndarray)──────> applications/
information/ ──(FisherResult)────> applications/
```

Never import directly between sibling subpackages. If `riemannian/` needs
something from `topology/`, refactor the shared logic into `types.py`.

### Performance Considerations

- Profile before optimizing — `python -m cProfile -o prof.out script.py`
- Curvature computation is the bottleneck; reduce `n_neighbors` or `reduced_dim`
- For large point clouds (N > 5000), use `FiltrationBuilder.maxmin_subsample()` before TDA
- Geodesic distance matrices are O(N^2) — use `RiemannianSearch` for k-NN queries instead
