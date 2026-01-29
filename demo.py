#!/usr/bin/env python3
"""HF Hub Ecosystem Demo - Interactive showcase with rich terminal output."""

import sys

# ANSI color codes for rich terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a styled header with box drawing."""
    width = 60
    print(f"\n{CYAN}{'═' * width}{RESET}")
    print(f"{CYAN}║{RESET} {BOLD}{text}{RESET}")
    print(f"{CYAN}{'═' * width}{RESET}\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{BLUE}▶ {BOLD}{text}{RESET}")
    print(f"{DIM}{'─' * 50}{RESET}")


def print_success(text: str) -> None:
    """Print success message with green checkmark."""
    print(f"  {GREEN}✓{RESET} {text}")


def print_info(text: str) -> None:
    """Print info message with blue indicator."""
    print(f"  {BLUE}●{RESET} {text}")


def demo_device_detection() -> None:
    """Demo device detection."""
    print_section("Device Detection")

    from hf_ecosystem import get_device, get_device_map

    device = get_device()
    device_color = GREEN if device in ("cuda", "mps") else YELLOW
    print_success(f"Detected device: {device_color}{device}{RESET}")

    device_map = get_device_map(model_size_gb=1.0)
    print_success(f"Recommended device map: {CYAN}{device_map}{RESET}")


def demo_hub_search() -> None:
    """Demo Hub search capabilities."""
    print_section("Hub Model Search")

    from hf_ecosystem import search_models

    print_info("Searching for sentiment analysis models...")
    models = search_models(task="text-classification", limit=3)

    print(f"\n  {DIM}{'─' * 48}{RESET}")
    for model in models:
        downloads = getattr(model, "downloads", 0)
        name = model.modelId[:40]
        print(
            f"  {DIM}│{RESET} {CYAN}{name:<40}{RESET} {DIM}({downloads:>10,} ↓){RESET}"
        )
    print(f"  {DIM}{'─' * 48}{RESET}")


def demo_preprocessing() -> None:
    """Demo text preprocessing."""
    print_section("Text Preprocessing")

    from hf_ecosystem import preprocess_text

    text = "  Hello, World!  "
    processed = preprocess_text(text, lowercase=True)
    print(f"  {DIM}│{RESET} Input:  '{text}'")
    print(f"  {DIM}│{RESET} Output: {GREEN}'{processed}'{RESET}")


def demo_pipeline() -> None:
    """Demo pipeline creation and inference."""
    print_section("Sentiment Analysis Pipeline")

    from hf_ecosystem import create_pipeline

    print_info("Loading model: distilbert-base-uncased-finetuned-sst-2-english")

    classifier = create_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    test_texts = [
        "I absolutely love this library!",
        "This is the worst experience ever.",
        "The weather today is okay.",
    ]

    print_info("Running inference...\n")
    print(f"  {DIM}{'─' * 48}{RESET}")

    for text in test_texts:
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]

        if label == "POSITIVE":
            emoji = f"{GREEN}↑{RESET}"
            label_color = GREEN
        else:
            emoji = f"{RED}↓{RESET}"
            label_color = RED

        short_text = text[:32] + "..." if len(text) > 32 else text
        print(
            f"  {DIM}│{RESET} {emoji} {short_text:<35} "
            f"{label_color}{label:>8}{RESET} {DIM}({score:.1%}){RESET}"
        )

    print(f"  {DIM}{'─' * 48}{RESET}")


def print_banner() -> None:
    """Print the demo banner."""
    banner = f"""
{CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   {BOLD}🤗 HF Hub Ecosystem Demo{RESET}{CYAN}                              ║
║                                                          ║
║   {DIM}Model search, preprocessing, and inference{RESET}{CYAN}            ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def main() -> int:
    """Run the demo."""
    print_banner()

    print(f"{DIM}This demo showcases the hf_ecosystem library capabilities.{RESET}\n")

    try:
        demo_device_detection()
        demo_preprocessing()
        demo_hub_search()
        demo_pipeline()

        print_header("Demo Complete!")
        print(f"  {GREEN}✓{RESET} All demonstrations completed successfully\n")
        print(f"  {DIM}For more examples, see:{RESET}")
        print(f"    • {CYAN}examples/quickstart.py{RESET}")
        print(f"    • {CYAN}notebooks/{RESET}\n")
        print(f"  Run {BOLD}make notebook{RESET} to launch Jupyter Lab.\n")

        return 0

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Demo interrupted by user.{RESET}")
        return 130
    except Exception as e:
        print(f"\n\n{RED}✗ Error:{RESET} {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
