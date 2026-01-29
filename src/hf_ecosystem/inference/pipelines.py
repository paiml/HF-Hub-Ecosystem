"""Pipeline creation utilities."""

from typing import Any

from transformers import Pipeline, pipeline

from hf_ecosystem.inference.device import get_device


def create_pipeline(
    task: str,
    model: str | None = None,
    device: str | int | None = None,
    **kwargs: Any,
) -> Pipeline:
    """Create a transformers pipeline with automatic device selection.

    Args:
        task: Pipeline task (e.g., "text-classification", "text-generation")
        model: Model identifier or path
        device: Device to use (auto-detected if None)
        **kwargs: Additional arguments passed to pipeline()

    Returns:
        Configured Pipeline object
    """
    if device is None:
        device = get_device()

    return pipeline(
        task=task,
        model=model,
        device=device,
        **kwargs,
    )


def list_supported_tasks() -> list[str]:
    """List all supported pipeline tasks.

    Returns:
        List of task names
    """
    return [
        "audio-classification",
        "automatic-speech-recognition",
        "depth-estimation",
        "document-question-answering",
        "feature-extraction",
        "fill-mask",
        "image-classification",
        "image-feature-extraction",
        "image-segmentation",
        "image-to-image",
        "image-to-text",
        "mask-generation",
        "object-detection",
        "question-answering",
        "sentiment-analysis",
        "summarization",
        "table-question-answering",
        "text-classification",
        "text-generation",
        "text-to-audio",
        "text-to-speech",
        "text2text-generation",
        "token-classification",
        "translation",
        "video-classification",
        "visual-question-answering",
        "zero-shot-classification",
        "zero-shot-image-classification",
        "zero-shot-object-detection",
    ]
