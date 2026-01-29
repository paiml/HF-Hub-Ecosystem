"""Device management utilities."""

import torch


def get_device() -> str:
    """Get the best available device.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_map(model_size_gb: float = 1.0) -> str | dict[str, int | str]:
    """Get device map for model loading.

    Args:
        model_size_gb: Estimated model size in GB

    Returns:
        Device map string or dict for accelerate
    """
    if not torch.cuda.is_available():
        return "cpu"

    # Check available GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024**3)

    if model_size_gb <= total_memory_gb * 0.8:
        return "cuda:0"

    # Use auto device map for large models
    return "auto"


def get_gpu_memory_info() -> dict[str, float]:
    """Get GPU memory information.

    Returns:
        Dict with total, allocated, and free memory in GB
    """
    if not torch.cuda.is_available():
        return {"total": 0.0, "allocated": 0.0, "free": 0.0}

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    free = total - allocated

    return {
        "total": round(total, 2),
        "allocated": round(allocated, 2),
        "free": round(free, 2),
    }


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
