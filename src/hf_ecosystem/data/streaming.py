"""Dataset streaming utilities."""

from typing import Iterator, Any

from datasets import load_dataset, IterableDataset, Dataset


def stream_dataset(
    path: str,
    name: str | None = None,
    split: str = "train",
    streaming: bool = True,
) -> IterableDataset | Dataset:
    """Load a dataset with optional streaming.

    Args:
        path: Dataset path on Hub or local
        name: Dataset configuration name
        split: Dataset split
        streaming: Whether to use streaming mode

    Returns:
        Dataset or IterableDataset
    """
    return load_dataset(
        path,
        name=name,
        split=split,
        streaming=streaming,
    )


def take_samples(
    dataset: IterableDataset,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Take n samples from a streaming dataset.

    Args:
        dataset: Streaming dataset
        n: Number of samples to take

    Returns:
        List of sample dictionaries
    """
    samples = []
    for i, sample in enumerate(dataset):
        if i >= n:
            break
        samples.append(sample)
    return samples


def filter_by_length(
    dataset: IterableDataset | Dataset,
    text_column: str = "text",
    min_length: int = 10,
    max_length: int = 10000,
) -> IterableDataset | Dataset:
    """Filter dataset by text length.

    Args:
        dataset: Input dataset
        text_column: Name of text column
        min_length: Minimum text length
        max_length: Maximum text length

    Returns:
        Filtered dataset
    """
    return dataset.filter(
        lambda x: min_length <= len(x[text_column]) <= max_length
    )
