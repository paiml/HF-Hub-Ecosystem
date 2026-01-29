# Hugging Face Hub and Ecosystem Fundamentals

<p align="center">
  <img src="docs/diagrams/hero-image.svg" alt="HF Hub Ecosystem" width="600">
</p>

**Courses 1-2** of the Hugging Face ML Specialization

[![CI](https://github.com/paiml/HF-Hub-Ecosystem/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/HF-Hub-Ecosystem/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/paiml/HF-Hub-Ecosystem/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-green.svg)](https://github.com/astral-sh/uv)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![TDG Score](https://img.shields.io/badge/TDG-A%20(90.7%25)-brightgreen.svg)](https://github.com/paiml/paiml-mcp-agent-toolkit)

## Overview

This repository contains interactive Jupyter notebooks for the first two courses of the Hugging Face ML specialization:

| Course | Title | Focus | Duration |
|--------|-------|-------|----------|
| **1** | Hub and Ecosystem Fundamentals | Navigation, inference, multi-modal | 3 weeks |
| **2** | Fine-Tuning Transformers | Trainer API, datasets, evaluation | 3 weeks |

**Total:** 6 weeks, ~24 hours of hands-on labs

## Quick Start

```bash
# Clone the repository
git clone https://github.com/paiml/HF-Hub-Ecosystem.git
cd HF-Hub-Ecosystem

# Setup environment (requires uv)
make setup

# Run linting and tests
make check

# Launch Jupyter Lab
make lab
```

## Usage

### Using the Library

```python
from hf_ecosystem import search_models, create_pipeline, get_device

# Search for models on the Hub
models = search_models(task="text-classification", limit=5)
for model in models:
    print(f"{model.modelId}: {model.downloads} downloads")

# Create an inference pipeline with automatic device detection
classifier = create_pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Run inference
result = classifier("I love this course!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Check available device
device = get_device()  # Returns 'cuda', 'mps', or 'cpu'
```

### Running Notebooks

```bash
# Launch Jupyter Lab
make notebook

# Navigate to notebooks/course1/week1/ to begin
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Run a specific test file
uv run pytest tests/unit/test_device.py -v

# Generate HTML coverage report
make coverage
```

## Course Structure

### Course 1: Hub and Ecosystem

| Week | Focus | Notebooks |
|------|-------|-----------|
| 1 | Hub Navigation | `1.3-exploring-hub`, `1.7-dataset-exploration` |
| 2 | Transformers Fundamentals | `2.2-text-pipelines`, `2.5-custom-inference`, `2.8-multi-gpu` |
| 3 | Multi-Modal Models | `3.2-image-classification`, `3.4-whisper-transcription`, `3.6-image-captioning` |

### Course 2: Fine-Tuning

| Week | Focus | Notebooks |
|------|-------|-----------|
| 1 | Data Preparation | `1.2-loading-datasets`, `1.7-custom-dataset` |
| 2 | Trainer API | `2.3-first-finetuning`, `2.5-custom-metrics`, `2.7-experiment-tracking` |
| 3 | Evaluation & Sharing | `3.2-comprehensive-eval`, `3.4-nlg-evaluation`, `3.7-publish-model` |

## Requirements

- **Python**: 3.11+
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (required)
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: Optional (CPU works for most labs)

## Installation

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup Project

```bash
# Install dependencies
make setup

# Verify installation
make check
```

## Development

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run all quality checks
make check
```

## Project Structure

```
HF-Hub-Ecosystem/
├── notebooks/
│   ├── course1/         # Hub and Ecosystem notebooks
│   │   ├── week1/
│   │   ├── week2/
│   │   └── week3/
│   └── course2/         # Fine-Tuning notebooks
│       ├── week1/
│       ├── week2/
│       └── week3/
├── src/
│   └── hf_ecosystem/    # Shared utilities
├── tests/               # Unit and integration tests
├── docs/                # Specifications and readings
└── Makefile             # Build targets
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | Model loading and inference |
| `datasets` | Dataset loading and processing |
| `evaluate` | Evaluation metrics |
| `accelerate` | Multi-GPU support |
| `huggingface-hub` | Hub API access |

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run `make check` before committing
4. Submit a pull request

## Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [Course Specification](docs/specifications/demo-repo.md)
