# visualization — Plotting & Paper Figure Generation

## Purpose

Visualization tools for all geometric, topological, and information-geometric
analysis results. Includes both exploratory plotting functions and
publication-quality figure generators targeting NeurIPS/ICML format.

## Modules

### `manifold.py` — Riemannian Manifold Plots

Visualize the Riemannian structure of embedding spaces:

| Function | What it Shows |
|----------|--------------|
| `plot_curvature_field()` | Scalar curvature as colored scatter on 2D embeddings (RdBu_r diverging colormap, centered at 0) |
| `plot_geodesic()` | Geodesic path overlaid on the embedding point cloud, with start/end markers |
| `plot_metric_ellipses()` | Local metric tensor visualized as ellipses — elongated where the metric stretches, contracted where it compresses |
| `plot_layer_curvature_profile()` | Mean scalar curvature ± 1 std as a function of layer depth |

### `persistence.py` — Persistent Homology Plots

Standard TDA visualization toolkit:

| Function | What it Shows |
|----------|--------------|
| `plot_persistence_diagram()` | Birth-vs-death scatter plot with diagonal reference line, colored by homology dimension (H_0, H_1, H_2) |
| `plot_barcode()` | Horizontal bar plot where each bar represents a topological feature's lifetime, sorted by persistence |
| `plot_betti_curve()` | Betti numbers (β_0, β_1, β_2) as functions of the filtration scale ε |
| `plot_persistence_landscape()` | First k landscape functions λ_k(t) with filled area and line overlay |

### `information.py` — Information Geometry Plots

Visualize Fisher information, entropy, and divergence structures:

| Function | What it Shows |
|----------|--------------|
| `plot_fisher_heatmap()` | Fisher Information Matrix as a viridis heatmap |
| `plot_entropy_scatter()` | Shannon entropy of next-token distributions as colored scatter on 2D embeddings |
| `plot_kl_matrix()` | Pairwise Jensen-Shannon divergence as a symmetric heatmap with optional labels |
| `plot_fisher_trace_by_layer()` | Fisher trace as a function of layer depth |

### `paper.py` — Publication-Quality Figures

NeurIPS/ICML-ready composite figures with standardized formatting:

- **`set_paper_style()`** — Configures matplotlib RC params: 300 DPI, 9pt fonts, LaTeX serif rendering (Computer Modern), grid with 0.3 alpha
- **Column widths**: `SINGLE_COL = 3.25"`, `DOUBLE_COL = 6.75"` (standard two-column format)
- **`figure_intrinsic_dimension()`** — Intrinsic dimensionality across layers, multi-model comparison
- **`figure_curvature_profile()`** — Curvature mean ± std across layers with fill_between
- **`figure_hallucination_comparison()`** — Side-by-side AUROC/AUPRC bar charts across detection methods
- **`save_all_figures()`** — Batch-generate all paper figures from a data dictionary, save as PDF

## Design Notes

- **Lazy matplotlib import**: All functions call `_require_matplotlib()` at runtime, so importing `topo_llm.visualization` never triggers matplotlib loading unless a plot is actually created.
- **Return Figure objects**: Every function returns `matplotlib.figure.Figure`, never calls `plt.show()`. The caller decides whether to display, save, or compose into subplots.
- **No side effects**: Functions create new figures, never modify global state (except `set_paper_style()` which intentionally sets RC params).

## Usage

```python
from topo_llm.visualization.persistence import plot_persistence_diagram
from topo_llm.visualization.paper import set_paper_style, figure_curvature_profile

# Exploratory plot
fig = plot_persistence_diagram(diagrams, title="GPT-2 Layer 6")
fig.savefig("diagram.png")

# Publication figure
set_paper_style()
fig = figure_curvature_profile(layers, curvature_stats)
fig.savefig("fig_curvature.pdf")
```
