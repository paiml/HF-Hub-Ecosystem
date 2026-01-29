"""Tests for dataset streaming utilities."""

from unittest.mock import MagicMock, patch

from hf_ecosystem.data.streaming import (
    filter_by_length,
    stream_dataset,
    take_samples,
)


class TestStreamDataset:
    """Tests for stream_dataset function."""

    @patch("hf_ecosystem.data.streaming.load_dataset")
    def test_stream_dataset_calls_load_dataset(self, mock_load):
        """stream_dataset should call load_dataset with correct args."""
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        result = stream_dataset("test-dataset", split="train", streaming=True)

        mock_load.assert_called_once_with(
            "test-dataset",
            name=None,
            split="train",
            streaming=True,
        )
        assert result == mock_dataset

    @patch("hf_ecosystem.data.streaming.load_dataset")
    def test_stream_dataset_with_config_name(self, mock_load):
        """stream_dataset should pass config name."""
        mock_load.return_value = MagicMock()

        stream_dataset("glue", name="sst2", split="validation")

        mock_load.assert_called_once_with(
            "glue",
            name="sst2",
            split="validation",
            streaming=True,
        )

    @patch("hf_ecosystem.data.streaming.load_dataset")
    def test_stream_dataset_non_streaming(self, mock_load):
        """stream_dataset should support non-streaming mode."""
        mock_load.return_value = MagicMock()

        stream_dataset("test-dataset", streaming=False)

        mock_load.assert_called_once_with(
            "test-dataset",
            name=None,
            split="train",
            streaming=False,
        )


class TestTakeSamples:
    """Tests for take_samples function."""

    def test_take_samples_returns_n_samples(self):
        """take_samples should return exactly n samples."""
        mock_dataset = [
            {"text": "sample 1"},
            {"text": "sample 2"},
            {"text": "sample 3"},
            {"text": "sample 4"},
            {"text": "sample 5"},
        ]

        result = take_samples(iter(mock_dataset), n=3)

        assert len(result) == 3
        assert result[0]["text"] == "sample 1"
        assert result[2]["text"] == "sample 3"

    def test_take_samples_returns_all_when_fewer_than_n(self):
        """take_samples should return all samples when fewer than n available."""
        mock_dataset = [{"text": "sample 1"}, {"text": "sample 2"}]

        result = take_samples(iter(mock_dataset), n=10)

        assert len(result) == 2

    def test_take_samples_returns_empty_for_empty_dataset(self):
        """take_samples should return empty list for empty dataset."""
        result = take_samples(iter([]), n=5)

        assert result == []


class TestFilterByLength:
    """Tests for filter_by_length function."""

    def test_filter_by_length_filters_short_texts(self):
        """filter_by_length should filter out short texts."""
        mock_dataset = MagicMock()
        mock_filtered = MagicMock()
        mock_dataset.filter.return_value = mock_filtered

        result = filter_by_length(
            mock_dataset,
            text_column="text",
            min_length=10,
            max_length=100,
        )

        assert result == mock_filtered
        mock_dataset.filter.assert_called_once()

        # Get the filter function and test it
        filter_func = mock_dataset.filter.call_args[0][0]
        assert filter_func({"text": "short"}) is False  # len=5 < min_length=10
        assert filter_func({"text": "this is long enough"}) is True  # len=19
        assert filter_func({"text": "x" * 150}) is False  # len=150 > max_length=100

    def test_filter_by_length_uses_custom_column(self):
        """filter_by_length should use custom text column."""
        mock_dataset = MagicMock()
        mock_dataset.filter.return_value = MagicMock()

        filter_by_length(mock_dataset, text_column="content")

        filter_func = mock_dataset.filter.call_args[0][0]
        assert filter_func({"content": "x" * 50}) is True
