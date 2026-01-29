#!/usr/bin/env python3
"""HF Hub Ecosystem Demo - Interactive showcase with rich terminal output."""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()


def print_header(text: str) -> None:
    """Print a styled header."""
    console.print()
    console.rule(f"[bold cyan]{text}[/]", style="cyan")
    console.print()


def print_section(text: str) -> None:
    """Print a section header."""
    console.print()
    console.print(f"[blue]▶[/] [bold]{text}[/]")
    console.print("[dim]" + "─" * 50 + "[/]")


def print_success(text: str) -> None:
    """Print success message with green checkmark."""
    console.print(f"  [green]✓[/] {text}")


def print_info(text: str) -> None:
    """Print info message with blue indicator."""
    console.print(f"  [blue]●[/] {text}")


def demo_device_detection() -> None:
    """Demo device detection."""
    print_section("Device Detection")

    from hf_ecosystem import get_device, get_device_map

    device = get_device()
    device_style = "green" if device in ("cuda", "mps") else "yellow"
    print_success(f"Detected device: [{device_style}]{device}[/]")

    device_map = get_device_map(model_size_gb=1.0)
    print_success(f"Recommended device map: [cyan]{device_map}[/]")


def demo_hub_search() -> None:
    """Demo Hub search capabilities."""
    print_section("Hub Model Search")

    from hf_ecosystem import search_models

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Searching for sentiment analysis models...", total=None)
        models = search_models(task="text-classification", limit=3)

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Model", style="cyan", width=45)
    table.add_column("Downloads", justify="right", style="dim")

    for model in models:
        downloads = getattr(model, "downloads", 0)
        name = model.modelId[:42] + "..." if len(model.modelId) > 45 else model.modelId
        table.add_row(name, f"{downloads:,} ↓")

    console.print(table)


def demo_preprocessing() -> None:
    """Demo text preprocessing."""
    print_section("Text Preprocessing")

    from hf_ecosystem import preprocess_text

    text = "  Hello, World!  "
    processed = preprocess_text(text, lowercase=True)
    console.print(f"  [dim]│[/] Input:  '{text}'")
    console.print(f"  [dim]│[/] Output: [green]'{processed}'[/]")


def demo_pipeline() -> None:
    """Demo pipeline creation and inference."""
    print_section("Sentiment Analysis Pipeline")

    from hf_ecosystem import create_pipeline

    print_info("Loading model: distilbert-base-uncased-finetuned-sst-2-english")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading model...", total=None)
        classifier = create_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    test_texts = [
        "I absolutely love this library!",
        "This is the worst experience ever.",
        "The weather today is okay.",
    ]

    print_info("Running inference...")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None)
    table.add_column("", width=3)
    table.add_column("Text", width=35)
    table.add_column("Sentiment", justify="right")
    table.add_column("Score", justify="right", style="dim")

    for text in test_texts:
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]

        if label == "POSITIVE":
            emoji = "[green]↑[/]"
            label_styled = f"[green]{label}[/]"
        else:
            emoji = "[red]↓[/]"
            label_styled = f"[red]{label}[/]"

        short_text = text[:32] + "..." if len(text) > 32 else text
        table.add_row(emoji, short_text, label_styled, f"{score:.1%}")

    console.print(table)


def print_banner() -> None:
    """Print the demo banner."""
    banner = Text()
    banner.append("🤗 HF Hub Ecosystem Demo\n", style="bold cyan")
    banner.append("Model search, preprocessing, and inference", style="dim")

    console.print()
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))


def main() -> int:
    """Run the demo."""
    print_banner()

    console.print("[dim]This demo showcases the hf_ecosystem library.[/]\n")

    try:
        demo_device_detection()
        demo_preprocessing()
        demo_hub_search()
        demo_pipeline()

        print_header("Demo Complete!")
        print_success("All demonstrations completed successfully")
        console.print()
        console.print("  [dim]For more examples, see:[/]")
        console.print("    • [cyan]examples/quickstart.py[/]")
        console.print("    • [cyan]notebooks/[/]")
        console.print()
        console.print("  Run [bold]make notebook[/] to launch Jupyter Lab.")
        console.print()

        return 0

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Demo interrupted by user.[/]")
        return 130
    except Exception as e:
        console.print(f"\n\n[red]✗ Error:[/] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
