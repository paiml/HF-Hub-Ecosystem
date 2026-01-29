"""Model card parsing utilities."""

from dataclasses import dataclass
from typing import Any

from huggingface_hub import ModelCard, model_info


@dataclass
class ParsedModelCard:
    """Parsed model card with structured fields."""

    model_id: str
    pipeline_tag: str | None
    library_name: str | None
    license: str | None
    tags: list[str]
    downloads: int
    likes: int
    text: str
    metadata: dict[str, Any]


def parse_model_card(model_id: str) -> ParsedModelCard:
    """Parse a model card from Hugging Face Hub.

    Args:
        model_id: Model identifier (e.g., "bert-base-uncased")

    Returns:
        ParsedModelCard with structured information
    """
    info = model_info(model_id)
    card = ModelCard.load(model_id)

    return ParsedModelCard(
        model_id=model_id,
        pipeline_tag=info.pipeline_tag,
        library_name=info.library_name,
        license=info.card_data.license if info.card_data else None,
        tags=list(info.tags) if info.tags else [],
        downloads=info.downloads or 0,
        likes=info.likes or 0,
        text=card.text or "",
        metadata=card.data.to_dict() if card.data else {},
    )


def get_model_license(model_id: str) -> str | None:
    """Get the license for a model.

    Args:
        model_id: Model identifier

    Returns:
        License string or None
    """
    info = model_info(model_id)
    if info.card_data:
        return info.card_data.license
    return None
