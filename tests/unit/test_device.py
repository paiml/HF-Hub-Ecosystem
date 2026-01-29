"""Tests for device management."""

import pytest

from hf_ecosystem.inference.device import (
    get_device,
    get_device_map,
    get_gpu_memory_info,
    clear_gpu_memory,
)


def test_get_device_returns_string():
    """get_device should return a device string."""
    device = get_device()
    assert device in ["cuda", "mps", "cpu"]


def test_get_device_map_returns_valid():
    """get_device_map should return valid device map."""
    device_map = get_device_map(model_size_gb=0.5)
    assert device_map in ["cuda:0", "cpu", "auto"]


def test_get_gpu_memory_info_returns_dict():
    """get_gpu_memory_info should return dict with keys."""
    info = get_gpu_memory_info()
    assert "total" in info
    assert "allocated" in info
    assert "free" in info
    assert all(isinstance(v, float) for v in info.values())


def test_clear_gpu_memory_no_error():
    """clear_gpu_memory should not raise errors."""
    clear_gpu_memory()  # Should not raise
