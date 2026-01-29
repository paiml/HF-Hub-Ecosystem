"""Tests for preprocessing utilities."""

from hf_ecosystem.data.preprocessing import (
    create_preprocessing_function,
    preprocess_text,
    tokenize_batch,
)


def test_preprocess_text_strips_whitespace():
    """preprocess_text should strip whitespace by default."""
    result = preprocess_text("  hello world  ")
    assert result == "hello world"


def test_preprocess_text_lowercase():
    """preprocess_text should lowercase when requested."""
    result = preprocess_text("Hello World", lowercase=True)
    assert result == "hello world"


def test_preprocess_text_no_strip():
    """preprocess_text should preserve whitespace when requested."""
    result = preprocess_text("  hello  ", strip_whitespace=False)
    assert result == "  hello  "


def test_tokenize_batch_returns_tensors(tokenizer, sample_texts):
    """tokenize_batch should return BatchEncoding with tensors."""
    result = tokenize_batch(sample_texts, tokenizer, max_length=32)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == len(sample_texts)


def test_tokenize_batch_respects_max_length(tokenizer):
    """tokenize_batch should truncate to max_length."""
    texts = ["a " * 1000]  # Very long text
    result = tokenize_batch(texts, tokenizer, max_length=32)
    assert result["input_ids"].shape[1] <= 32


def test_create_preprocessing_function(tokenizer):
    """create_preprocessing_function should return callable."""
    func = create_preprocessing_function(tokenizer, text_column="text")
    assert callable(func)

    # Test the function
    examples = {"text": ["Hello world", "Test text"]}
    result = func(examples)
    assert "input_ids" in result


def test_create_preprocessing_function_with_labels(tokenizer):
    """create_preprocessing_function should include labels when present."""
    func = create_preprocessing_function(
        tokenizer, text_column="text", label_column="label"
    )

    examples = {"text": ["Hello world", "Test text"], "label": [0, 1]}
    result = func(examples)

    assert "input_ids" in result
    assert "labels" in result
    assert result["labels"] == [0, 1]


def test_create_preprocessing_function_without_label_column(tokenizer):
    """create_preprocessing_function should skip labels when column missing."""
    func = create_preprocessing_function(
        tokenizer, text_column="text", label_column="label"
    )

    # Examples without label column
    examples = {"text": ["Hello world", "Test text"]}
    result = func(examples)

    assert "input_ids" in result
    assert "labels" not in result


def test_create_preprocessing_function_with_none_label_column(tokenizer):
    """create_preprocessing_function should skip labels when label_column is None."""
    func = create_preprocessing_function(
        tokenizer, text_column="text", label_column=None
    )

    examples = {"text": ["Hello world"], "label": [0]}
    result = func(examples)

    assert "input_ids" in result
    assert "labels" not in result
