"""Tests for the CLI module."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from topo_llm.cli import _setup_logging, build_parser, main


class TestBuildParser:
    """Test CLI argument parser construction."""

    def test_returns_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_has_version_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_has_log_level(self):
        parser = build_parser()
        args = parser.parse_args(["--log-level", "DEBUG", "extract", "--texts", "t.txt"])
        assert args.log_level == "DEBUG"

    def test_extract_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["extract", "--texts", "texts.txt", "--model", "gpt2"])
        assert args.command == "extract"
        assert args.texts == "texts.txt"
        assert args.model == "gpt2"

    def test_extract_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["extract", "--texts", "t.txt"])
        assert args.model == "gpt2"
        assert args.pooling == "mean"
        assert args.batch_size == 32
        assert args.device == "auto"
        assert args.layers == "all"

    def test_analyze_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "--embeddings", "emb.npz"])
        assert args.command == "analyze"
        assert args.reduced_dim == 50
        assert args.n_neighbors == 15

    def test_detect_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["detect", "--reference", "ref.txt", "--query", "test query"])
        assert args.command == "detect"
        assert args.query == ["test query"]

    def test_detect_multiple_queries(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "detect",
                "--reference",
                "ref.txt",
                "--query",
                "query1",
                "query2",
            ]
        )
        assert len(args.query) == 2

    def test_figures_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["figures", "--results", "results/"])
        assert args.command == "figures"
        assert args.output == "figures"

    def test_no_command_sets_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestSetupLogging:
    """Test logging configuration."""

    def test_runs_without_error(self):
        _setup_logging("INFO")
        # basicConfig only works on first call; just verify no exception

    def test_case_insensitive(self):
        _setup_logging("warning")
        # Should not raise


class TestMainEntryPoint:
    """Test the main() function dispatch."""

    def test_no_command_prints_help(self):
        with patch("sys.argv", ["topo-llm"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_extract_missing_file_exits(self):
        with patch(
            "sys.argv",
            [
                "topo-llm",
                "extract",
                "--texts",
                "/nonexistent/file.txt",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_analyze_missing_file_exits(self):
        with patch(
            "sys.argv",
            [
                "topo-llm",
                "analyze",
                "--embeddings",
                "/nonexistent/embeddings.npz",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_detect_missing_reference_exits(self):
        with patch(
            "sys.argv",
            [
                "topo-llm",
                "detect",
                "--reference",
                "/nonexistent/ref.txt",
                "--query",
                "test",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_figures_missing_dir_exits(self):
        with patch(
            "sys.argv",
            [
                "topo-llm",
                "figures",
                "--results",
                "/nonexistent/results/",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
