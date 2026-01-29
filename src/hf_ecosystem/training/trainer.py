"""Trainer creation utilities."""

from collections.abc import Callable
from typing import Any

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    output_dir: str = "./results",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    compute_metrics: Callable[..., dict[str, float]] | None = None,
    **kwargs: Any,
) -> Trainer:
    """Create a configured Trainer instance.

    Args:
        model: Model to train
        tokenizer: Tokenizer for data collation
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size for training and evaluation
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        compute_metrics: Metrics computation function
        **kwargs: Additional TrainingArguments

    Returns:
        Configured Trainer instance
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(eval_dataset),
        logging_steps=100,
        **kwargs,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


def get_default_training_args(
    output_dir: str = "./results",
    num_epochs: int = 3,
) -> TrainingArguments:
    """Get default training arguments.

    Args:
        output_dir: Output directory
        num_epochs: Number of epochs

    Returns:
        TrainingArguments with sensible defaults
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
