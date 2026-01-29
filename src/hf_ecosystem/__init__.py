"""Hugging Face Ecosystem utilities for Courses 1-2."""

__version__ = "1.0.0"

from hf_ecosystem.data.preprocessing import preprocess_text, tokenize_batch
from hf_ecosystem.data.streaming import stream_dataset
from hf_ecosystem.hub.cards import parse_model_card
from hf_ecosystem.hub.search import search_datasets, search_models
from hf_ecosystem.inference.device import get_device, get_device_map
from hf_ecosystem.inference.pipelines import create_pipeline
from hf_ecosystem.training.metrics import compute_metrics
from hf_ecosystem.training.trainer import create_trainer

__all__ = [
    "__version__",
    "search_models",
    "search_datasets",
    "parse_model_card",
    "create_pipeline",
    "get_device",
    "get_device_map",
    "preprocess_text",
    "tokenize_batch",
    "stream_dataset",
    "create_trainer",
    "compute_metrics",
]
