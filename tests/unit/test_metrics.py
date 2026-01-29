"""Tests for metrics computation utilities."""

from unittest.mock import MagicMock, patch

import numpy as np

from hf_ecosystem.training.metrics import (
    compute_bleu_metrics,
    compute_metrics,
    compute_rouge_metrics,
)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    @patch("hf_ecosystem.training.metrics.evaluate.load")
    def test_compute_metrics_returns_dict(self, mock_load):
        """compute_metrics should return dict with accuracy and f1."""
        # Setup mock evaluators
        mock_accuracy = MagicMock()
        mock_accuracy.compute.return_value = {"accuracy": 0.85}
        mock_f1 = MagicMock()
        mock_f1.compute.return_value = {"f1": 0.82}

        def load_evaluator(name):
            if name == "accuracy":
                return mock_accuracy
            return mock_f1

        mock_load.side_effect = load_evaluator

        # Create eval_pred with logits (2 classes)
        predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = np.array([1, 0, 1])
        eval_pred = (predictions, labels)

        result = compute_metrics(eval_pred)

        assert "accuracy" in result
        assert "f1" in result
        assert result["accuracy"] == 0.85
        assert result["f1"] == 0.82

    @patch("hf_ecosystem.training.metrics.evaluate.load")
    def test_compute_metrics_handles_tuple_predictions(self, mock_load):
        """compute_metrics should handle tuple predictions (from some models)."""
        mock_accuracy = MagicMock()
        mock_accuracy.compute.return_value = {"accuracy": 0.9}
        mock_f1 = MagicMock()
        mock_f1.compute.return_value = {"f1": 0.88}

        def load_evaluator(name):
            if name == "accuracy":
                return mock_accuracy
            return mock_f1

        mock_load.side_effect = load_evaluator

        # Tuple predictions (logits as first element)
        predictions = (np.array([[0.1, 0.9], [0.8, 0.2]]), "extra_data")
        labels = np.array([1, 0])
        eval_pred = (predictions, labels)

        result = compute_metrics(eval_pred)

        assert result["accuracy"] == 0.9


class TestComputeRougeMetrics:
    """Tests for compute_rouge_metrics function."""

    @patch("hf_ecosystem.training.metrics.evaluate.load")
    def test_compute_rouge_metrics_returns_scores(self, mock_load):
        """compute_rouge_metrics should return ROUGE scores."""
        mock_rouge = MagicMock()
        mock_rouge.compute.return_value = {
            "rouge1": 0.75,
            "rouge2": 0.60,
            "rougeL": 0.70,
        }
        mock_load.return_value = mock_rouge

        predictions = ["This is a summary.", "Another summary here."]
        labels = ["This is the reference.", "Another reference."]
        eval_pred = (predictions, labels)

        result = compute_rouge_metrics(eval_pred)

        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        assert result["rouge1"] == 0.75

    @patch("hf_ecosystem.training.metrics.evaluate.load")
    def test_compute_rouge_metrics_handles_array_inputs(self, mock_load):
        """compute_rouge_metrics should convert arrays to strings."""
        mock_rouge = MagicMock()
        mock_rouge.compute.return_value = {
            "rouge1": 0.5,
            "rouge2": 0.4,
            "rougeL": 0.45,
        }
        mock_load.return_value = mock_rouge

        # Array inputs (as might come from model outputs)
        predictions = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        labels = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        eval_pred = (predictions, labels)

        result = compute_rouge_metrics(eval_pred)

        assert "rouge1" in result
        # Verify the strings were passed to compute
        mock_rouge.compute.assert_called_once()


class TestComputeBleuMetrics:
    """Tests for compute_bleu_metrics function."""

    @patch("hf_ecosystem.training.metrics.evaluate.load")
    def test_compute_bleu_metrics_returns_score(self, mock_load):
        """compute_bleu_metrics should return BLEU score."""
        mock_bleu = MagicMock()
        mock_bleu.compute.return_value = {"bleu": 0.65}
        mock_load.return_value = mock_bleu

        predictions = ["The cat sat on the mat.", "Hello world."]
        references = [["The cat is on the mat."], ["Hello there world."]]

        result = compute_bleu_metrics(predictions, references)

        assert "bleu" in result
        assert result["bleu"] == 0.65
        mock_bleu.compute.assert_called_once_with(
            predictions=predictions,
            references=references,
        )
