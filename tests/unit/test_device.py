"""Tests for device management."""

from unittest.mock import MagicMock, patch

from hf_ecosystem.inference.device import (
    clear_gpu_memory,
    get_device,
    get_device_map,
    get_gpu_memory_info,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_returns_string(self):
        """get_device should return a device string."""
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]

    @patch("hf_ecosystem.inference.device.torch")
    def test_get_device_returns_cuda_when_available(self, mock_torch):
        """get_device should return 'cuda' when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True

        device = get_device()

        assert device == "cuda"

    @patch("hf_ecosystem.inference.device.torch")
    def test_get_device_returns_mps_when_available(self, mock_torch):
        """get_device should return 'mps' when MPS is available (no CUDA)."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        device = get_device()

        assert device == "mps"

    @patch("hf_ecosystem.inference.device.torch")
    def test_get_device_returns_cpu_as_fallback(self, mock_torch):
        """get_device should return 'cpu' when no GPU available."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        device = get_device()

        assert device == "cpu"


class TestGetDeviceMap:
    """Tests for get_device_map function."""

    def test_get_device_map_returns_valid(self):
        """get_device_map should return valid device map."""
        device_map = get_device_map(model_size_gb=0.5)
        assert device_map in ["cuda:0", "cpu", "auto"]

    @patch("hf_ecosystem.inference.device.torch")
    def test_get_device_map_returns_cpu_without_cuda(self, mock_torch):
        """get_device_map should return 'cpu' when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False

        device_map = get_device_map(model_size_gb=1.0)

        assert device_map == "cpu"

    @patch("hf_ecosystem.inference.device.torch")
    def test_get_device_map_returns_cuda_for_small_model(self, mock_torch):
        """get_device_map should return 'cuda:0' for small models."""
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)  # 16 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props

        device_map = get_device_map(model_size_gb=1.0)  # 1 GB model, 16 GB GPU

        assert device_map == "cuda:0"

    @patch("hf_ecosystem.inference.device.torch")
    def test_get_device_map_returns_auto_for_large_model(self, mock_torch):
        """get_device_map should return 'auto' for large models."""
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024**3)  # 8 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props

        device_map = get_device_map(model_size_gb=10.0)  # 10 GB model, 8 GB GPU

        assert device_map == "auto"


class TestGetGpuMemoryInfo:
    """Tests for get_gpu_memory_info function."""

    def test_get_gpu_memory_info_returns_dict(self):
        """get_gpu_memory_info should return dict with keys."""
        info = get_gpu_memory_info()
        assert "total" in info
        assert "allocated" in info
        assert "free" in info
        assert all(isinstance(v, float) for v in info.values())

    @patch("hf_ecosystem.inference.device.torch")
    def test_get_gpu_memory_info_returns_zeros_without_cuda(self, mock_torch):
        """get_gpu_memory_info should return zeros when no CUDA."""
        mock_torch.cuda.is_available.return_value = False

        info = get_gpu_memory_info()

        assert info == {"total": 0.0, "allocated": 0.0, "free": 0.0}

    @patch("hf_ecosystem.inference.device.torch")
    def test_get_gpu_memory_info_returns_gpu_stats(self, mock_torch):
        """get_gpu_memory_info should return actual GPU stats."""
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)  # 16 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 2 * (1024**3)  # 2 GB

        info = get_gpu_memory_info()

        assert info["total"] == 16.0
        assert info["allocated"] == 2.0
        assert info["free"] == 14.0


class TestClearGpuMemory:
    """Tests for clear_gpu_memory function."""

    def test_clear_gpu_memory_no_error(self):
        """clear_gpu_memory should not raise errors."""
        clear_gpu_memory()  # Should not raise

    @patch("hf_ecosystem.inference.device.torch")
    def test_clear_gpu_memory_calls_empty_cache(self, mock_torch):
        """clear_gpu_memory should call empty_cache when CUDA available."""
        mock_torch.cuda.is_available.return_value = True

        clear_gpu_memory()

        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("hf_ecosystem.inference.device.torch")
    def test_clear_gpu_memory_skips_without_cuda(self, mock_torch):
        """clear_gpu_memory should not call empty_cache without CUDA."""
        mock_torch.cuda.is_available.return_value = False

        clear_gpu_memory()

        mock_torch.cuda.empty_cache.assert_not_called()
