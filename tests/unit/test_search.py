"""Tests for hub search utilities."""

from unittest.mock import MagicMock, patch

from hf_ecosystem.hub.search import iter_models, search_datasets, search_models


class TestSearchModels:
    """Tests for search_models function."""

    @patch("hf_ecosystem.hub.search.HfApi")
    def test_search_models_returns_list(self, mock_api_class):
        """search_models should return a list of ModelInfo objects."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_model = MagicMock()
        mock_model.modelId = "test-model"
        mock_api.list_models.return_value = [mock_model]

        result = search_models(query="bert", limit=5)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_api.list_models.assert_called_once_with(
            search="bert",
            pipeline_tag=None,
            filter=None,
            limit=5,
            sort="downloads",
        )

    @patch("hf_ecosystem.hub.search.HfApi")
    def test_search_models_with_task_filter(self, mock_api_class):
        """search_models should filter by task."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.list_models.return_value = []

        search_models(task="text-classification")

        mock_api.list_models.assert_called_once_with(
            search=None,
            pipeline_tag="text-classification",
            filter=None,
            limit=10,
            sort="downloads",
        )

    @patch("hf_ecosystem.hub.search.HfApi")
    def test_search_models_with_library_filter(self, mock_api_class):
        """search_models should filter by library."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.list_models.return_value = []

        search_models(library="transformers")

        mock_api.list_models.assert_called_once_with(
            search=None,
            pipeline_tag=None,
            filter=["transformers"],
            limit=10,
            sort="downloads",
        )


class TestSearchDatasets:
    """Tests for search_datasets function."""

    @patch("hf_ecosystem.hub.search.HfApi")
    def test_search_datasets_returns_list(self, mock_api_class):
        """search_datasets should return a list of DatasetInfo objects."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_dataset = MagicMock()
        mock_dataset.id = "test-dataset"
        mock_api.list_datasets.return_value = [mock_dataset]

        result = search_datasets(query="imdb", limit=5)

        assert isinstance(result, list)
        assert len(result) == 1
        mock_api.list_datasets.assert_called_once_with(
            search="imdb",
            task_categories=None,
            limit=5,
            sort="downloads",
        )

    @patch("hf_ecosystem.hub.search.HfApi")
    def test_search_datasets_with_task_filter(self, mock_api_class):
        """search_datasets should filter by task."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.list_datasets.return_value = []

        search_datasets(task="text-classification")

        mock_api.list_datasets.assert_called_once_with(
            search=None,
            task_categories="text-classification",
            limit=10,
            sort="downloads",
        )


class TestIterModels:
    """Tests for iter_models function."""

    @patch("hf_ecosystem.hub.search.HfApi")
    def test_iter_models_yields_models(self, mock_api_class):
        """iter_models should yield ModelInfo objects."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_models = [MagicMock(), MagicMock()]
        mock_api.list_models.return_value = iter(mock_models)

        result = list(iter_models(task="text-classification"))

        assert len(result) == 2
        mock_api.list_models.assert_called_once_with(
            pipeline_tag="text-classification",
            filter=None,
        )

    @patch("hf_ecosystem.hub.search.HfApi")
    def test_iter_models_with_library(self, mock_api_class):
        """iter_models should filter by library."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.list_models.return_value = iter([])

        list(iter_models(library="transformers"))

        mock_api.list_models.assert_called_once_with(
            pipeline_tag=None,
            filter=["transformers"],
        )
