"""Shared pytest fixtures."""

import pytest
from transformers import AutoTokenizer


@pytest.fixture
def tokenizer():
    """Load a small tokenizer for testing."""
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a positive review.",
        "This movie was terrible.",
        "The food was okay, nothing special.",
    ]
