# riemannian — Riemannian Geometry Engine

## Purpose

Estimate the Riemannian structure of embedding point clouds: local metric tensors,
Christoffel symbols, curvature tensors, and geodesics. This reveals the intrinsic
geometry of how LLMs organize representations.

## Mathematical Pipeline

```
Point cloud (N, D)
    → PCA reduction to (N, d) where d ≈ 50
    → Local PCA at each point → tangent bases T_i
    → Metric tensor: g_ij = T_i^T @ T_i
    → Christoffel symbols: Γ^k_{ij} via finite differences of g
    → Riemann tensor: R^l_{ijk} via derivatives of Γ
    → Ricci tensor: Ric_{ij} = R^k_{ikj}
    → Scalar curvature: S = g^{ij} Ric_{ij}
    → Geodesics: solve d²γ/dt² + Γ·(dγ/dt)² = 0
```

## Modules

### `metric.py` — MetricTensorEstimator

Estimates the local Riemannian metric at each point using k-nearest-neighbor
covariance analysis. The metric is interpolated between points using inverse-distance
weighting to create a smooth field.

### `connection.py` — ChristoffelEstimator

Computes Christoffel symbols Γ^k_{ij} via central finite differences of the
interpolated metric field. These encode how the coordinate system "twists" —
they're needed for geodesic computation and curvature.

### `curvature.py` — CurvatureAnalyzer

Full curvature pipeline:
- **Riemann tensor** R^l_{ijk}: The complete curvature information
- **Ricci tensor** Ric_{ij}: Contraction measuring volume distortion
- **Scalar curvature** S: Single number summarizing curvature at a point
- **Sectional curvature** K(v1, v2): Curvature of a 2-plane

### `geodesic.py` — GeodesicSolver

Solves the geodesic equation (shortest paths on the manifold) using 4th-order
Runge-Kutta integration. Also provides:
- Exponential map (tangent vector → point on manifold)
- Logarithmic map (inverse of exponential)
- Geodesic distance matrix
- Shooting method for boundary-value geodesics

### `search.py` — RiemannianSearch

Geodesic-aware nearest neighbor search. Pre-filters candidates with Euclidean
distance, then refines with geodesic distance. Compares Euclidean, cosine, and
geodesic neighborhoods.

## Validation

Correctness is verified on manifolds with known geometry:
- **2-Sphere** (R=1): scalar curvature = 2, geodesics = great circles
- **Flat plane**: all curvatures = 0, geodesic = Euclidean distance
- **Torus**: positive curvature outside, negative inside
