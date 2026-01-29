"""Metrics computation utilities."""

from typing import Any

import evaluate
import numpy as np


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        eval_pred: EvalPrediction with predictions and labels

    Returns:
        Dict of metric names to values
    """
    # Handle both tuple unpacking and attribute access
    if hasattr(eval_pred, "predictions"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=-1)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    acc_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    return {
        "accuracy": acc_result["accuracy"],
        "f1": f1_result["f1"],
    }


def compute_rouge_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute ROUGE metrics for summarization.

    Args:
        eval_pred: EvalPrediction with predictions and labels

    Returns:
        Dict of ROUGE scores
    """
    # Handle both tuple unpacking and attribute access
    if hasattr(eval_pred, "predictions"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    rouge = evaluate.load("rouge")

    # Decode if needed (assumes string inputs)
    if isinstance(predictions[0], list | np.ndarray):
        predictions = [str(p) for p in predictions]
    if isinstance(labels[0], list | np.ndarray):
        labels = [str(label) for label in labels]

    result = rouge.compute(predictions=predictions, references=labels)
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }


def compute_bleu_metrics(
    predictions: list[str],
    references: list[list[str]],
) -> dict[str, float]:
    """Compute BLEU score for translation.

    Args:
        predictions: Model predictions
        references: Reference translations (list of lists)

    Returns:
        Dict with BLEU score
    """
    bleu = evaluate.load("bleu")
    result = bleu.compute(predictions=predictions, references=references)
    return {"bleu": result["bleu"]}
