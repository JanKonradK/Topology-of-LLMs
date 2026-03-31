"""
Visualization
=============

Plotting tools for geometric and topological analysis results,
plus publication-quality figure generation.

All plot functions require matplotlib (install with ``pip install topo-llm[viz]``).
Each function returns a ``matplotlib.figure.Figure`` object without calling
``plt.show()``, giving the caller full control over display and saving.

Modules
-------
manifold
    Curvature field plots, geodesic paths, metric ellipses.
persistence
    Persistence diagrams, barcodes, landscapes, Betti curves.
information
    Fisher information heatmaps, entropy surfaces, KL matrices.
paper
    Publication-quality figure generation with journal RC params.
"""

from __future__ import annotations

__all__ = [
    # manifold
    "plot_curvature_field",
    "plot_geodesic",
    "plot_metric_ellipses",
    "plot_layer_curvature_profile",
    # persistence
    "plot_persistence_diagram",
    "plot_barcode",
    "plot_betti_curve",
    "plot_persistence_landscape",
    # information
    "plot_fisher_heatmap",
    "plot_entropy_scatter",
    "plot_kl_matrix",
    "plot_fisher_trace_by_layer",
    # paper
    "set_paper_style",
    "figure_intrinsic_dimension",
    "figure_curvature_profile",
    "figure_hallucination_comparison",
    "save_all_figures",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading matplotlib at package import time."""
    _manifold = {
        "plot_curvature_field", "plot_geodesic",
        "plot_metric_ellipses", "plot_layer_curvature_profile",
    }
    _persistence = {
        "plot_persistence_diagram", "plot_barcode",
        "plot_betti_curve", "plot_persistence_landscape",
    }
    _information = {
        "plot_fisher_heatmap", "plot_entropy_scatter",
        "plot_kl_matrix", "plot_fisher_trace_by_layer",
    }
    _paper = {
        "set_paper_style", "figure_intrinsic_dimension",
        "figure_curvature_profile", "figure_hallucination_comparison",
        "save_all_figures",
    }

    if name in _manifold:
        from topo_llm.visualization.manifold import (
            plot_curvature_field, plot_geodesic,
            plot_metric_ellipses, plot_layer_curvature_profile,
        )
        return locals()[name]
    if name in _persistence:
        from topo_llm.visualization.persistence import (
            plot_persistence_diagram, plot_barcode,
            plot_betti_curve, plot_persistence_landscape,
        )
        return locals()[name]
    if name in _information:
        from topo_llm.visualization.information import (
            plot_fisher_heatmap, plot_entropy_scatter,
            plot_kl_matrix, plot_fisher_trace_by_layer,
        )
        return locals()[name]
    if name in _paper:
        from topo_llm.visualization.paper import (
            set_paper_style, figure_intrinsic_dimension,
            figure_curvature_profile, figure_hallucination_comparison,
            save_all_figures,
        )
        return locals()[name]

    raise AttributeError(f"module 'topo_llm.visualization' has no attribute {name!r}")
