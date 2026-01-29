"""Tests for model card parsing utilities."""

from unittest.mock import MagicMock, patch

from hf_ecosystem.hub.cards import (
    ParsedModelCard,
    get_model_license,
    parse_model_card,
)


class TestParsedModelCard:
    """Tests for ParsedModelCard dataclass."""

    def test_parsed_model_card_creation(self):
        """ParsedModelCard should be creatable with all fields."""
        card = ParsedModelCard(
            model_id="test-model",
            pipeline_tag="text-classification",
            library_name="transformers",
            license="mit",
            tags=["pytorch", "bert"],
            downloads=1000,
            likes=50,
            text="# Model Card",
            metadata={"language": "en"},
        )

        assert card.model_id == "test-model"
        assert card.pipeline_tag == "text-classification"
        assert card.library_name == "transformers"
        assert card.license == "mit"
        assert card.tags == ["pytorch", "bert"]
        assert card.downloads == 1000
        assert card.likes == 50
        assert card.text == "# Model Card"
        assert card.metadata == {"language": "en"}


class TestParseModelCard:
    """Tests for parse_model_card function."""

    @patch("hf_ecosystem.hub.cards.ModelCard")
    @patch("hf_ecosystem.hub.cards.model_info")
    def test_parse_model_card_returns_parsed_card(self, mock_model_info, mock_card):
        """parse_model_card should return ParsedModelCard."""
        # Setup mock model info
        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-classification"
        mock_info.library_name = "transformers"
        mock_info.card_data = MagicMock()
        mock_info.card_data.license = "apache-2.0"
        mock_info.tags = ["pytorch", "bert"]
        mock_info.downloads = 5000
        mock_info.likes = 100
        mock_model_info.return_value = mock_info

        # Setup mock model card
        mock_card_instance = MagicMock()
        mock_card_instance.text = "# Test Model"
        mock_card_instance.data = {"language": "en"}
        mock_card.load.return_value = mock_card_instance

        result = parse_model_card("test-model")

        assert isinstance(result, ParsedModelCard)
        assert result.model_id == "test-model"
        assert result.pipeline_tag == "text-classification"
        assert result.library_name == "transformers"
        assert result.license == "apache-2.0"
        assert result.tags == ["pytorch", "bert"]
        assert result.downloads == 5000
        assert result.likes == 100
        assert result.text == "# Test Model"
        assert result.metadata == {"language": "en"}

    @patch("hf_ecosystem.hub.cards.ModelCard")
    @patch("hf_ecosystem.hub.cards.model_info")
    def test_parse_model_card_handles_none_values(self, mock_model_info, mock_card):
        """parse_model_card should handle None values gracefully."""
        mock_info = MagicMock()
        mock_info.pipeline_tag = None
        mock_info.library_name = None
        mock_info.card_data = None
        mock_info.tags = None
        mock_info.downloads = None
        mock_info.likes = None
        mock_model_info.return_value = mock_info

        mock_card_instance = MagicMock()
        mock_card_instance.text = None
        mock_card_instance.data = None
        mock_card.load.return_value = mock_card_instance

        result = parse_model_card("test-model")

        assert result.pipeline_tag is None
        assert result.library_name is None
        assert result.license is None
        assert result.tags == []
        assert result.downloads == 0
        assert result.likes == 0
        assert result.text == ""
        assert result.metadata == {}


class TestGetModelLicense:
    """Tests for get_model_license function."""

    @patch("hf_ecosystem.hub.cards.model_info")
    def test_get_model_license_returns_license(self, mock_model_info):
        """get_model_license should return license string."""
        mock_info = MagicMock()
        mock_info.card_data = MagicMock()
        mock_info.card_data.license = "mit"
        mock_model_info.return_value = mock_info

        result = get_model_license("test-model")

        assert result == "mit"
        mock_model_info.assert_called_once_with("test-model")

    @patch("hf_ecosystem.hub.cards.model_info")
    def test_get_model_license_returns_none_when_no_card_data(self, mock_model_info):
        """get_model_license should return None when no card_data."""
        mock_info = MagicMock()
        mock_info.card_data = None
        mock_model_info.return_value = mock_info

        result = get_model_license("test-model")

        assert result is None
