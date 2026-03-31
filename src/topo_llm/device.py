"""
Device detection and management for topo-llm.

Provides a unified interface for detecting available compute devices
across both PyTorch and JAX backends.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> str:
    """Detect and return the best available compute device.

    Parameters
    ----------
    preference : str
        Device preference. Options:
        - ``"auto"``: CUDA > MPS > CPU (tries in order)
        - ``"cpu"``: Force CPU
        - ``"cuda"``: Force CUDA (raises if unavailable)
        - ``"mps"``: Force MPS (raises if unavailable)

    Returns
    -------
    str
        Device string compatible with PyTorch (e.g., "cpu", "cuda", "mps").

    Raises
    ------
    RuntimeError
        If the requested device is not available.
    """
    if preference == "cpu":
        return "cpu"

    if preference == "auto":
        # Try CUDA first
        if _cuda_available():
            logger.info("Auto-detected CUDA device")
            return "cuda"
        # Then MPS (Apple Silicon)
        if _mps_available():
            logger.info("Auto-detected MPS device")
            return "mps"
        # Fall back to CPU
        logger.info("No GPU detected, using CPU")
        return "cpu"

    if preference == "cuda":
        if not _cuda_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"

    if preference == "mps":
        if not _mps_available():
            raise RuntimeError("MPS requested but not available")
        return "mps"

    raise ValueError(f"Unknown device preference: {preference!r}")


def device_info() -> dict[str, object]:
    """Return information about available compute devices.

    Returns
    -------
    dict[str, object]
        Dictionary with keys:
        - ``"cuda_available"``: bool
        - ``"cuda_device_count"``: int
        - ``"cuda_device_name"``: str or None
        - ``"mps_available"``: bool
        - ``"selected"``: str (result of auto-detection)
    """
    info: dict[str, object] = {
        "cuda_available": _cuda_available(),
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "mps_available": _mps_available(),
        "selected": get_device("auto"),
    }

    if info["cuda_available"]:
        try:
            import torch
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    return info


def _cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available via PyTorch."""
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False
