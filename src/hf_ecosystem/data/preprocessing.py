"""Text preprocessing utilities."""

from typing import Any

from transformers import PreTrainedTokenizer, BatchEncoding


def preprocess_text(
    text: str,
    lowercase: bool = False,
    strip_whitespace: bool = True,
) -> str:
    """Preprocess text for tokenization.

    Args:
        text: Input text
        lowercase: Convert to lowercase
        strip_whitespace: Strip leading/trailing whitespace

    Returns:
        Preprocessed text
    """
    if strip_whitespace:
        text = text.strip()
    if lowercase:
        text = text.lower()
    return text


def tokenize_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    padding: bool | str = True,
    truncation: bool = True,
) -> BatchEncoding:
    """Tokenize a batch of texts.

    Args:
        texts: List of input texts
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate

    Returns:
        BatchEncoding with input_ids, attention_mask, etc.
    """
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )


def create_preprocessing_function(
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    label_column: str | None = "label",
    max_length: int = 512,
) -> Any:
    """Create a preprocessing function for dataset.map().

    Args:
        tokenizer: HuggingFace tokenizer
        text_column: Name of text column
        label_column: Name of label column (optional)
        max_length: Maximum sequence length

    Returns:
        Function suitable for dataset.map()
    """

    def preprocess(examples: dict[str, Any]) -> dict[str, Any]:
        result = tokenizer(
            examples[text_column],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        if label_column and label_column in examples:
            result["labels"] = examples[label_column]
        return result

    return preprocess
