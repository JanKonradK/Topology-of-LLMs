"""
Command-line interface for topo-llm.

Provides subcommands for common workflows:
- extract: Extract embeddings from a HuggingFace model
- analyze: Run geometric/topological analysis on saved embeddings
- detect: Score texts for hallucination risk
- figures: Generate publication-quality figures

Usage
-----
    topo-llm extract --model gpt2 --texts data/texts.txt --output data/embeddings/
    topo-llm analyze --embeddings data/embeddings/ --output results/
    topo-llm detect --model gpt2 --reference data/reference.txt --query "Some claim"
    topo-llm figures --results results/ --output figures/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from topo_llm import __version__
from topo_llm.config import load_config


def _setup_logging(level: str) -> None:
    """Configure logging for CLI usage."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract embeddings from a HuggingFace model.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments with model, texts, output, layers, pooling, batch_size, device.
    """
    from topo_llm.extraction import EmbeddingExtractor

    logger = logging.getLogger("topo_llm.cli.extract")

    # Load texts
    texts_path = Path(args.texts)
    if not texts_path.exists():
        logger.error("Texts file not found: %s", texts_path)
        sys.exit(1)

    with open(texts_path) as f:
        texts = [line.strip() for line in f if line.strip()]

    logger.info("Loaded %d texts from %s", len(texts), texts_path)

    # Parse layers
    if args.layers == "all":
        layers = None
    else:
        layers = [int(x) for x in args.layers.split(",")]

    # Extract
    extractor = EmbeddingExtractor(
        args.model,
        device=args.device,
        pooling=args.pooling,
    )
    embeddings = extractor.extract_dataset(texts, layers=layers, batch_size=args.batch_size)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model.replace('/', '_')}_embeddings.npz"

    np.savez_compressed(
        output_path,
        **{f"layer_{k}": v for k, v in embeddings.items()},
    )
    logger.info("Saved embeddings to %s", output_path)
    logger.info(
        "Layers: %s, Shape per layer: %s",
        list(embeddings.keys()),
        next(iter(embeddings.values())).shape,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run geometric and topological analysis on saved embeddings.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments with embeddings path, output dir, reduced_dim, n_neighbors.
    """
    from sklearn.decomposition import PCA

    from topo_llm.extraction import LayerAnalyzer
    from topo_llm.riemannian import CurvatureAnalyzer, MetricTensorEstimator

    logger = logging.getLogger("topo_llm.cli.analyze")

    # Load embeddings
    emb_path = Path(args.embeddings)
    if not emb_path.exists():
        logger.error("Embeddings file not found: %s", emb_path)
        sys.exit(1)

    data = np.load(emb_path)
    layer_keys = sorted(data.files)
    logger.info("Loaded %d layers from %s", len(layer_keys), emb_path)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for key in layer_keys:
        emb = data[key]
        logger.info("Analyzing %s: shape %s", key, emb.shape)

        # Intrinsic dimensionality
        intrinsic_dim = LayerAnalyzer.intrinsic_dimensionality(emb, method="mle")
        anisotropy = LayerAnalyzer.compute_anisotropy(emb)

        # PCA reduction for Riemannian analysis
        reduced_dim = min(args.reduced_dim, emb.shape[1], emb.shape[0] - 1)
        emb_reduced = PCA(n_components=reduced_dim).fit_transform(emb)

        # Metric and curvature
        metric_est = MetricTensorEstimator(n_neighbors=args.n_neighbors)
        metric_est.fit(emb_reduced)
        curv = CurvatureAnalyzer(metric_est)
        stats = curv.curvature_statistics(emb_reduced)

        results[key] = {
            "intrinsic_dim": intrinsic_dim,
            "effective_rank": anisotropy["effective_rank"],
            "isotropy": anisotropy["isotropy_score"],
            "scalar_curvature_mean": stats["scalar_mean"],
            "scalar_curvature_std": stats["scalar_std"],
        }

        logger.info(
            "  intrinsic_dim=%.1f, curvature_mean=%.4f ± %.4f",
            intrinsic_dim,
            stats["scalar_mean"],
            stats["scalar_std"],
        )

    # Save results
    results_path = output_dir / "analysis_results.npz"
    np.savez(
        results_path,
        **{f"{k}_{metric}": v for k, metrics in results.items() for metric, v in metrics.items()},
    )
    logger.info("Saved analysis results to %s", results_path)


def cmd_detect(args: argparse.Namespace) -> None:
    """Score texts for hallucination risk.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments with model, reference texts, query text, device.
    """
    from topo_llm.applications import HallucinationDetector

    logger = logging.getLogger("topo_llm.cli.detect")

    # Load reference texts
    ref_path = Path(args.reference)
    if not ref_path.exists():
        logger.error("Reference file not found: %s", ref_path)
        sys.exit(1)

    with open(ref_path) as f:
        reference_texts = [line.strip() for line in f if line.strip()]

    logger.info("Loaded %d reference texts", len(reference_texts))

    # Build detector
    detector = HallucinationDetector(args.model, device=args.device)
    detector.fit(
        reference_texts,
        layer=args.layer,
        reduced_dim=args.reduced_dim,
    )

    # Score query
    queries = args.query if isinstance(args.query, list) else [args.query]
    for query in queries:
        score = detector.score(query)
        print(f"\nQuery: {query}")
        print(f"  Combined score:  {score.combined_score:.4f}")
        print(f"  Curvature:       {score.curvature_score:.4f}")
        print(f"  Topological:     {score.topological_score:.4f}")
        print(f"  Information:     {score.information_score:.4f}")
        print(f"  Density:         {score.density_score:.4f}")


def cmd_figures(args: argparse.Namespace) -> None:
    """Generate publication-quality figures.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments with results dir, output dir.
    """
    from topo_llm.visualization.paper import save_all_figures

    logger = logging.getLogger("topo_llm.cli.figures")

    results_dir = Path(args.results)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load available results and generate figures
    figure_data = {}

    # Check for intrinsic dimension data
    intrinsic_path = results_dir / "intrinsic_dimensions.npz"
    if intrinsic_path.exists():
        data = np.load(intrinsic_path, allow_pickle=True)
        figure_data["intrinsic_dim"] = {
            "layers": data["layers"].tolist(),
            "dims_by_model": data["dims_by_model"].item(),
        }

    # Check for curvature data
    curvature_path = results_dir / "curvature_profiles.npz"
    if curvature_path.exists():
        data = np.load(curvature_path, allow_pickle=True)
        figure_data["curvature"] = {
            "layers": data["layers"].tolist(),
            "stats": data["stats"].item(),
        }

    if not figure_data:
        logger.warning("No result files found in %s. Run 'topo-llm analyze' first.", results_dir)
        sys.exit(1)

    saved = save_all_figures(output_dir, **figure_data)
    logger.info("Generated %d figures in %s", len(saved), output_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="topo-llm",
        description="Geometry and Topology of LLM Representation Spaces",
    )
    parser.add_argument("--version", action="version", version=f"topo-llm {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--config", default=None, help="Path to YAML config file (default: config/default.yaml)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── extract ──────────────────────────────────────────────
    p_extract = subparsers.add_parser("extract", help="Extract embeddings from a model")
    p_extract.add_argument("--model", default="gpt2", help="HuggingFace model name (default: gpt2)")
    p_extract.add_argument("--texts", required=True, help="Path to text file (one text per line)")
    p_extract.add_argument("--output", default="data/embeddings", help="Output directory")
    p_extract.add_argument("--layers", default="all", help="Comma-separated layer indices or 'all'")
    p_extract.add_argument(
        "--pooling", default="mean", choices=["mean", "cls", "last", "max"], help="Pooling strategy"
    )
    p_extract.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_extract.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")

    # ── analyze ──────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Analyze embeddings (geometry + topology)")
    p_analyze.add_argument("--embeddings", required=True, help="Path to .npz embeddings file")
    p_analyze.add_argument("--output", default="data/results", help="Output directory")
    p_analyze.add_argument(
        "--reduced-dim",
        type=int,
        default=50,
        help="PCA dimensions for Riemannian analysis (default: 50)",
    )
    p_analyze.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="k-NN neighbors for metric estimation (default: 15)",
    )

    # ── detect ───────────────────────────────────────────────
    p_detect = subparsers.add_parser("detect", help="Score texts for hallucination risk")
    p_detect.add_argument("--model", default="gpt2", help="HuggingFace model name")
    p_detect.add_argument("--reference", required=True, help="Path to reference texts file")
    p_detect.add_argument("--query", required=True, nargs="+", help="Text(s) to score")
    p_detect.add_argument("--layer", type=int, default=-1, help="Layer to analyze (default: -1)")
    p_detect.add_argument("--reduced-dim", type=int, default=50, help="PCA dimensions")
    p_detect.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")

    # ── figures ──────────────────────────────────────────────
    p_figures = subparsers.add_parser("figures", help="Generate paper figures")
    p_figures.add_argument("--results", required=True, help="Path to results directory")
    p_figures.add_argument("--output", default="figures", help="Output directory for figures")

    return parser


def main() -> None:
    """CLI entry point for topo-llm."""
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)

    if args.config:
        load_config(args.config)

    commands = {
        "extract": cmd_extract,
        "analyze": cmd_analyze,
        "detect": cmd_detect,
        "figures": cmd_figures,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands[args.command](args)


if __name__ == "__main__":
    main()
