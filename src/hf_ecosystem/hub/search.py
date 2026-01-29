"""Search utilities for Hugging Face Hub."""

from collections.abc import Iterator

from huggingface_hub import DatasetInfo, HfApi, ModelInfo


def search_models(
    query: str | None = None,
    task: str | None = None,
    library: str | None = None,
    limit: int = 10,
) -> list[ModelInfo]:
    """Search for models on Hugging Face Hub.

    Args:
        query: Search query string
        task: Filter by task (e.g., "text-classification")
        library: Filter by library (e.g., "transformers")
        limit: Maximum number of results

    Returns:
        List of ModelInfo objects
    """
    api = HfApi()
    # Build filter list for library
    filter_list = [library] if library else None
    models = api.list_models(
        search=query,
        pipeline_tag=task,
        filter=filter_list,
        limit=limit,
        sort="downloads",
    )
    return list(models)


def search_datasets(
    query: str | None = None,
    task: str | None = None,
    limit: int = 10,
) -> list[DatasetInfo]:
    """Search for datasets on Hugging Face Hub.

    Args:
        query: Search query string
        task: Filter by task
        limit: Maximum number of results

    Returns:
        List of DatasetInfo objects
    """
    api = HfApi()
    datasets = api.list_datasets(
        search=query,
        task_categories=task,
        limit=limit,
        sort="downloads",
    )
    return list(datasets)


def iter_models(
    task: str | None = None,
    library: str | None = None,
) -> Iterator[ModelInfo]:
    """Iterate over all models matching criteria.

    Args:
        task: Filter by task
        library: Filter by library

    Yields:
        ModelInfo objects
    """
    api = HfApi()
    filter_list = [library] if library else None
    yield from api.list_models(pipeline_tag=task, filter=filter_list)
