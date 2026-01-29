"""Tests for pipeline utilities."""

from unittest.mock import MagicMock, patch

import pytest

from hf_ecosystem.inference.pipelines import (
    create_pipeline,
    list_supported_tasks,
)


class TestListSupportedTasks:
    """Tests for list_supported_tasks function."""

    def test_list_supported_tasks_returns_list(self):
        """list_supported_tasks should return non-empty list."""
        tasks = list_supported_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert "text-classification" in tasks
        assert "text-generation" in tasks

    def test_list_supported_tasks_contains_common_tasks(self):
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


class TestCreatePipeline:
    """Tests for create_pipeline function."""

    @pytest.mark.slow
    def test_create_pipeline_sentiment(self):
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

    @patch("hf_ecosystem.inference.pipelines.pipeline")
    @patch("hf_ecosystem.inference.pipelines.get_device")
    def test_create_pipeline_auto_device(self, mock_get_device, mock_pipeline):
        """create_pipeline should auto-detect device when not specified."""
        mock_get_device.return_value = "cpu"
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        result = create_pipeline("text-classification", model="test-model")

        mock_get_device.assert_called_once()
        mock_pipeline.assert_called_once_with(
            task="text-classification",
            model="test-model",
            device="cpu",
        )
        assert result == mock_pipe

    @patch("hf_ecosystem.inference.pipelines.pipeline")
    def test_create_pipeline_explicit_device(self, mock_pipeline):
        """create_pipeline should use explicit device when specified."""
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        create_pipeline(
            "text-classification",
            model="test-model",
            device="cuda:1",
        )

        mock_pipeline.assert_called_once_with(
            task="text-classification",
            model="test-model",
            device="cuda:1",
        )

    @patch("hf_ecosystem.inference.pipelines.pipeline")
    def test_create_pipeline_passes_kwargs(self, mock_pipeline):
        """create_pipeline should pass additional kwargs."""
        mock_pipeline.return_value = MagicMock()

        create_pipeline(
            "text-generation",
            model="gpt2",
            device="cpu",
            max_length=100,
            temperature=0.7,
        )

        mock_pipeline.assert_called_once_with(
            task="text-generation",
            model="gpt2",
            device="cpu",
            max_length=100,
            temperature=0.7,
        )
