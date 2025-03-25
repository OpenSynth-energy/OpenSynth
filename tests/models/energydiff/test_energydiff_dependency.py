import pytest


# flake8: noqa: F401
def test_basic_imports():
    """Test if core dependencies are met."""
    try:
        import einops
        import ema_pytorch
        import numpy
        import torch
    except ImportError as e:
        pytest.fail(f"Failed to import required library: {e}")


def test_energydiff_dependency():
    """Test if energydiff module is importable."""
    try:
        from opensynth.models.energydiff.diffusion import PLDiffusion1D
    except ImportError as e:
        pytest.fail(f"Failed to import required library: {e}")


def test_calibrate_dependency():
    """Test if calibrate module is importable."""
    try:
        from opensynth.models.energydiff.calibrate import calibrate
    except ImportError as e:
        pytest.fail(f"Failed to import required library: {e}")
