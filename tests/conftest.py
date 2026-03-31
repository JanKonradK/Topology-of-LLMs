"""
Shared test fixtures for topo-llm.

Provides synthetic data fixtures for fast testing without model loading,
plus markers for slow (model-loading) and GPU-requiring tests.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Synthetic data fixtures ───────────────────────────────────

@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_embeddings(rng: np.random.Generator) -> np.ndarray:
    """Synthetic embeddings: 200 points in R^768 (like GPT-2)."""
    return rng.standard_normal((200, 768)).astype(np.float32)


@pytest.fixture
def small_embeddings(rng: np.random.Generator) -> np.ndarray:
    """Small synthetic embeddings: 50 points in R^64."""
    return rng.standard_normal((50, 64)).astype(np.float32)


@pytest.fixture
def sphere_points() -> np.ndarray:
    """200 points uniformly sampled on a unit 2-sphere in R^3.

    Known properties:
    - Intrinsic dimension ≈ 2
    - Scalar curvature = 2/R² = 2
    - H_0 = 1, H_1 = 0, H_2 = 1 (persistent)
    """
    rng = np.random.default_rng(42)
    n = 200
    # Sample from standard normal and normalize to unit sphere
    points = rng.standard_normal((n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return (points / norms).astype(np.float32)


@pytest.fixture
def sphere_points_large() -> np.ndarray:
    """500 points on a unit 2-sphere for topology tests."""
    rng = np.random.default_rng(42)
    n = 500
    points = rng.standard_normal((n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return (points / norms).astype(np.float32)


@pytest.fixture
def torus_points() -> np.ndarray:
    """300 points on a torus (R=3, r=1) in R^3.

    Parametrization: ((R+r*cos θ)cos φ, (R+r*cos θ)sin φ, r*sin θ)

    Known properties:
    - Intrinsic dimension ≈ 2
    - Gaussian curvature K = cos θ / (r(R + r·cos θ))
    - H_0 = 1, H_1 = 2, H_2 = 1
    """
    rng = np.random.default_rng(42)
    n = 300
    R, r = 3.0, 1.0
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.column_stack([x, y, z]).astype(np.float32)


@pytest.fixture
def circle_points() -> np.ndarray:
    """200 points on a unit circle in R^2.

    Known properties:
    - Intrinsic dimension ≈ 1
    - H_0 = 1, H_1 = 1
    """
    n = 200
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.column_stack([x, y]).astype(np.float32)


@pytest.fixture
def flat_plane_points(rng: np.random.Generator) -> np.ndarray:
    """200 points on a flat plane in R^3 (z=0).

    Known properties:
    - All curvatures ≈ 0
    - Geodesic distance ≈ Euclidean distance
    """
    n = 200
    xy = rng.uniform(-5, 5, (n, 2))
    z = np.zeros((n, 1))
    return np.hstack([xy, z]).astype(np.float32)


@pytest.fixture
def two_clusters(rng: np.random.Generator) -> np.ndarray:
    """Two well-separated Gaussian clusters in R^3.

    Known properties:
    - H_0 should show 2 persistent components
    """
    n = 100
    cluster_1 = rng.standard_normal((n, 3)) + np.array([5, 0, 0])
    cluster_2 = rng.standard_normal((n, 3)) + np.array([-5, 0, 0])
    return np.vstack([cluster_1, cluster_2]).astype(np.float32)


@pytest.fixture
def low_dim_subspace(rng: np.random.Generator) -> np.ndarray:
    """Data from a known 5-dimensional subspace of R^100.

    For testing intrinsic dimensionality estimation.
    """
    n = 500
    intrinsic_dim = 5
    ambient_dim = 100
    # Random 5D data projected to 100D
    low_dim = rng.standard_normal((n, intrinsic_dim))
    projection = rng.standard_normal((intrinsic_dim, ambient_dim))
    return (low_dim @ projection).astype(np.float32)


# ── Model name fixtures ───────────────────────────────────────

@pytest.fixture
def tiny_model_name() -> str:
    """Tiny GPT-2 model for fast tests."""
    return "sshleifer/tiny-gpt2"
