# topology — Topological Data Analysis Pipeline

## Purpose

Compute persistent homology of embedding point clouds to identify topological
features (connected components, loops, voids) that persist across scales. These
features reveal qualitative structural properties invisible to standard distance metrics.

## Key Concepts

- **H_0 (connected components)**: How many clusters? When do they merge?
- **H_1 (loops/cycles)**: Are there circular relationships in the embeddings?
- **H_2 (voids)**: Are there enclosed cavities in the point cloud?
- **Persistence**: Features that persist across many scales are "real"; short-lived ones are noise.

## Modules

### `filtration.py` — FiltrationBuilder

Builds simplicial complex filtrations from point clouds:
- **Vietoris-Rips**: Standard choice, works in any dimension
- **Alpha complex**: More geometrically meaningful, efficient for low intrinsic dim
- **Maxmin subsampling**: Furthest point sampling when N is too large

### `homology.py` — PersistentHomologyAnalyzer

Core analysis of persistence diagrams:
- Betti numbers at any scale
- Betti curves across all scales
- Persistence entropy (complexity measure)
- Significance filtering (Otsu, percentile, mean-lifetime)
- Summary statistics

### `landscapes.py` — PersistenceLandscape

Functional representation of persistence diagrams that lives in a Banach space,
enabling proper statistical analysis:
- Means, norms, and distances between landscapes
- Permutation tests for comparing groups

### `distances.py` — DiagramDistances

Metrics between persistence diagrams:
- **Wasserstein distance**: Optimal transport between diagrams
- **Bottleneck distance**: Worst-case matching cost

### `features.py` — TopologicalFeatures

Fixed-size feature vectors from persistence diagrams for ML:
- 30-dim statistics vector (10 features x 3 homology dimensions)
- Persistence images (2D rasterization)
- Combined feature vectors

## Validation

Known topological spaces verify correctness:
- **Circle** S^1: H_0=1, H_1=1
- **Sphere** S^2: H_0=1, H_1=0, H_2=1
- **Torus** T^2: H_0=1, H_1=2, H_2=1
