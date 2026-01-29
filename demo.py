#!/usr/bin/env python3
"""HF Hub Ecosystem Demo - Interactive showcase of library capabilities."""

import sys


def print_header(text: str) -> None:
    """Print a styled header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n>>> {text}")
    print("-" * 50)


def print_success(text: str) -> None:
    """Print success message."""
    print(f"  [OK] {text}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"  [..] {text}")


def demo_device_detection() -> None:
    """Demo device detection."""
    print_section("Device Detection")

    from hf_ecosystem import get_device, get_device_map

    device = get_device()
    print_success(f"Detected device: {device}")

    device_map = get_device_map(model_size_gb=1.0)
    print_success(f"Recommended device map: {device_map}")


def demo_hub_search() -> None:
    """Demo Hub search capabilities."""
    print_section("Hub Search")

    from hf_ecosystem import search_datasets, search_models

    print_info("Searching for sentiment analysis models...")
    models = search_models(task="text-classification", limit=3)
    for model in models:
        downloads = getattr(model, "downloads", "N/A")
        print_success(f"  {model.modelId} ({downloads:,} downloads)")

    print_info("Searching for text datasets...")
    datasets = search_datasets(task="text-classification", limit=3)
    for ds in datasets:
        downloads = getattr(ds, "downloads", "N/A")
        print_success(f"  {ds.id} ({downloads:,} downloads)")


def demo_preprocessing() -> None:
    """Demo text preprocessing."""
    print_section("Text Preprocessing")

    from hf_ecosystem import preprocess_text

    text = "  Hello, World!  "
    processed = preprocess_text(text, lowercase=True)
    print_success(f"Original: '{text}'")
    print_success(f"Processed: '{processed}'")


def demo_pipeline() -> None:
    """Demo pipeline creation."""
    print_section("Inference Pipeline")

    from hf_ecosystem import create_pipeline

    print_info("Loading sentiment analysis pipeline...")
    classifier = create_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    test_texts = [
        "I love this library!",
        "This is terrible.",
        "The weather is okay.",
    ]

    print_info("Running inference...")
    for text in test_texts:
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]
        emoji = "+" if label == "POSITIVE" else "-"
        print_success(f"  [{emoji}] {text[:30]:<30} -> {label} ({score:.2%})")


def main() -> int:
    """Run the demo."""
    print_header("HF Hub Ecosystem Demo")

    print("This demo showcases the hf_ecosystem library capabilities.")
    print("It will demonstrate device detection, Hub search, and inference.\n")

    try:
        demo_device_detection()
        demo_preprocessing()
        demo_hub_search()
        demo_pipeline()

        print_header("Demo Complete!")
        print("For more examples, see the notebooks/ directory.")
        print("Run 'make notebook' to launch Jupyter Lab.\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
