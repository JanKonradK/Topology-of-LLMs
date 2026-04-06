"""Tests for device detection and management."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from topo_llm.device import _cuda_available, _mps_available, device_info, get_device


class TestGetDevice:
    """Test device selection logic."""

    def test_cpu_always_returns_cpu(self):
        assert get_device("cpu") == "cpu"

    def test_auto_returns_valid_string(self):
        result = get_device("auto")
        assert result in ("cpu", "cuda", "mps")

    def test_invalid_preference_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown device"):
            get_device("tpu")

    def test_cuda_unavailable_raises_runtime_error(self):
        with patch("topo_llm.device._cuda_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA requested"):
                get_device("cuda")

    def test_mps_unavailable_raises_runtime_error(self):
        with patch("topo_llm.device._mps_available", return_value=False):
            with pytest.raises(RuntimeError, match="MPS requested"):
                get_device("mps")

    def test_auto_prefers_cuda_over_mps(self):
        with patch("topo_llm.device._cuda_available", return_value=True):
            assert get_device("auto") == "cuda"

    def test_auto_falls_back_to_mps(self):
        with patch("topo_llm.device._cuda_available", return_value=False):
            with patch("topo_llm.device._mps_available", return_value=True):
                assert get_device("auto") == "mps"

    def test_auto_falls_back_to_cpu(self):
        with patch("topo_llm.device._cuda_available", return_value=False):
            with patch("topo_llm.device._mps_available", return_value=False):
                assert get_device("auto") == "cpu"


class TestDeviceInfo:
    """Test device info dictionary."""

    def test_returns_dict_with_expected_keys(self):
        info = device_info()
        assert "cuda_available" in info
        assert "cuda_device_count" in info
        assert "cuda_device_name" in info
        assert "mps_available" in info
        assert "selected" in info

    def test_cuda_available_is_bool(self):
        info = device_info()
        assert isinstance(info["cuda_available"], bool)

    def test_selected_is_valid_device(self):
        info = device_info()
        assert info["selected"] in ("cpu", "cuda", "mps")


class TestHelpers:
    """Test private helper functions."""

    def test_cuda_available_returns_bool(self):
        assert isinstance(_cuda_available(), bool)

    def test_mps_available_returns_bool(self):
        assert isinstance(_mps_available(), bool)
