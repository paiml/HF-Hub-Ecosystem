"""Tests for pipeline utilities."""

import pytest

from hf_ecosystem.inference.pipelines import (
    create_pipeline,
    list_supported_tasks,
)


def test_list_supported_tasks_returns_list():
    """list_supported_tasks should return non-empty list."""
    tasks = list_supported_tasks()
    assert isinstance(tasks, list)
    assert len(tasks) > 0
    assert "text-classification" in tasks
    assert "text-generation" in tasks


def test_list_supported_tasks_contains_common_tasks():
    """list_supported_tasks should include common tasks."""
    tasks = list_supported_tasks()
    common = [
        "sentiment-analysis",
        "summarization",
        "translation",
        "question-answering",
    ]
    for task in common:
        assert task in tasks


@pytest.mark.slow
def test_create_pipeline_sentiment():
    """create_pipeline should create working sentiment pipeline."""
    pipe = create_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device="cpu",
    )
    result = pipe("I love this!")
    assert len(result) > 0
    assert "label" in result[0]
    assert "score" in result[0]
