#!/usr/bin/env python3
"""Quick start example for hf_ecosystem library."""

from hf_ecosystem import (
    create_pipeline,
    get_device,
    preprocess_text,
    search_models,
)


def main():
    # 1. Check device
    device = get_device()
    print(f"Using device: {device}")

    # 2. Search for models
    models = search_models(task="text-classification", limit=3)
    print(f"\nFound {len(models)} models:")
    for m in models:
        print(f"  - {m.modelId}")

    # 3. Preprocess text
    text = "  Hello, World!  "
    clean = preprocess_text(text, lowercase=True)
    print(f"\nPreprocessed: '{text}' -> '{clean}'")

    # 4. Run inference
    classifier = create_pipeline("sentiment-analysis")
    result = classifier("I love machine learning!")
    print(f"\nSentiment: {result[0]['label']} ({result[0]['score']:.2%})")


if __name__ == "__main__":
    main()
