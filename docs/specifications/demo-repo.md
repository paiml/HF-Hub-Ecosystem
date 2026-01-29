# Hugging Face Hub and Ecosystem Fundamentals

## Courses 1-2: Demonstration Repository Specification

**Version**: 1.0.0
**Status**: Draft
**Last Updated**: 2026-01-29
**Author**: PAIML Research Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Development Philosophy](#2-development-philosophy)
3. [Repository Architecture](#3-repository-architecture)
4. [Toolchain Specification](#4-toolchain-specification)
5. [Notebook Standards](#5-notebook-standards)
6. [Quality Gates](#6-quality-gates)
7. [Popperian Falsification Criteria](#7-popperian-falsification-criteria)
8. [Lab Specifications](#8-lab-specifications)
9. [Peer-Reviewed Citations](#9-peer-reviewed-citations)
10. [Appendices](#appendices)

---

## 1. Executive Summary

### 1.1 Course Context

This specification defines the demonstration repository for **Courses 1-2** of the 5-course Hugging Face specialization:

| Course | Title | Focus | Duration |
|--------|-------|-------|----------|
| **1** | Hub and Ecosystem Fundamentals | Navigation, inference, multi-modal | 3 weeks |
| **2** | Fine-Tuning Transformers | Trainer API, datasets, evaluation | 3 weeks |
| 3 | Advanced NLP | Token classification, QA, generation | 3 weeks |
| 4 | Advanced Fine-Tuning | LoRA, QLoRA, PEFT, DPO | 3 weeks |
| 5 | Production ML | Deployment, optimization, edge | 3 weeks |

**Total Specialization:** 15 weeks, ~60 hours

### 1.2 Primary Conjecture

**Conjecture**: A Python-only implementation using `uv` for dependency management and Jupyter notebooks for interactive demonstrations can achieve:

1. **100% reproducibility** via locked dependencies and deterministic execution
2. **Sub-5-second notebook startup** with cached environments
3. **Zero dependency conflicts** through strict isolation
4. **95%+ test coverage** on all utility code
5. **Complete Hub API coverage** for model, dataset, and space operations

This conjecture is falsifiable through the 100-point validation matrix defined in Section 7.

### 1.3 Language and Runtime Constraints

| Constraint | Specification | Rationale |
|------------|---------------|-----------|
| **Language** | Python 3.11+ | Latest stable, required for transformers |
| **Package Manager** | `uv` only | Fast, reproducible, no pip/conda |
| **Notebook Runtime** | Jupyter Lab 4.x | Industry standard, nbformat 4.5 |
| **Testing** | pytest + nbval | Notebook validation as tests |
| **Linting** | ruff | Fast, replaces flake8/isort/black |
| **Type Checking** | pyright (strict) | Full type coverage |

### 1.4 Scope

| Week | Course | Focus | Labs | Notebooks |
|------|--------|-------|------|-----------|
| 1 | C1 | Hub Navigation | 2 | 4 |
| 2 | C1 | Transformers Fundamentals | 3 | 6 |
| 3 | C1 | Multi-Modal Models | 3 | 6 |
| 4 | C2 | Data Preparation | 2 | 4 |
| 5 | C2 | Trainer API | 3 | 6 |
| 6 | C2 | Evaluation & Sharing | 3 | 6 |

**Total:** 16 labs, 32 notebooks

### 1.5 Non-Goals

- Rust or compiled language components
- Custom CUDA kernels or GPU optimization
- Production deployment (covered in Course 5)
- Model training from scratch
- Distributed training across nodes

---

## 2. Development Philosophy

### 2.1 Test-Driven Development (TDD)

All code follows strict TDD methodology:

```
RED → GREEN → REFACTOR
```

1. **RED**: Write failing test first
2. **GREEN**: Implement minimum code to pass
3. **REFACTOR**: Clean up while tests pass

#### Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_hub_utils.py
│   ├── test_tokenization.py
│   └── test_preprocessing.py
├── integration/             # Cross-component tests
│   ├── test_pipeline_flow.py
│   └── test_trainer_integration.py
├── notebooks/               # Notebook execution tests
│   ├── test_week1_notebooks.py
│   └── test_week2_notebooks.py
└── conftest.py              # Shared fixtures
```

### 2.2 Toyota Way Principles

#### Genchi Genbutsu (現地現物) — Go and See

> "Go to the source to find the facts."

**Application**: All notebooks execute real inference on real models. No mocked outputs in demonstrations. Each lab includes verification cells that assert expected behavior.

```python
# Every notebook ends with verification
def verify_notebook_outputs():
    """Verify all notebook outputs are reproducible."""
    assert model is not None, "Model must be loaded"
    assert len(outputs) > 0, "Must produce outputs"
    assert outputs[0]["score"] > 0.5, "Confidence threshold"
```

#### Jidoka (自働化) — Build Quality In

> "Stop and fix problems when they occur."

**Application**: Notebooks fail fast with clear errors. No silent failures or swallowed exceptions.

```python
# Fail fast pattern
from transformers import AutoModel

try:
    model = AutoModel.from_pretrained(model_id)
except OSError as e:
    raise RuntimeError(f"Model {model_id} not found on Hub: {e}") from e
```

#### Muda (無駄) — Eliminate Waste

> "Eliminate non-value-adding activities."

**Application**:
- No redundant imports across notebooks
- Shared utility modules for common operations
- Cached model downloads via HF_HOME
- Pre-computed embeddings where applicable

### 2.3 Makefile-Driven Workflow

All operations flow through `make` targets:

```makefile
.PHONY: setup lint test check clean

# Setup environment
setup:
	uv sync
	uv run pre-commit install

# Linting
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

# Testing
test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Full check (CI equivalent)
check: lint test
	uv run pmat comply check

# Run specific notebook
notebook-%:
	uv run jupyter lab notebooks/$*.ipynb
```

---

## 3. Repository Architecture

### 3.1 Directory Structure

```
HF-Hub-Ecosystem/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI
├── docs/
│   ├── readings/                  # Course readings (markdown)
│   │   ├── 0.0-course-introduction.md
│   │   ├── 1.2-model-cards.md
│   │   ├── 1.5-licensing.md
│   │   └── ...
│   ├── specifications/
│   │   └── demo-repo.md           # This document
│   └── diagrams/
│       └── hero-image.svg
├── notebooks/
│   ├── course1/
│   │   ├── week1/
│   │   │   ├── 1.3-exploring-hub.ipynb
│   │   │   └── 1.7-dataset-exploration.ipynb
│   │   ├── week2/
│   │   │   ├── 2.2-text-pipelines.ipynb
│   │   │   ├── 2.5-custom-inference.ipynb
│   │   │   └── 2.8-multi-gpu.ipynb
│   │   └── week3/
│   │       ├── 3.2-image-classification.ipynb
│   │       ├── 3.4-whisper-transcription.ipynb
│   │       └── 3.6-image-captioning.ipynb
│   └── course2/
│       ├── week1/
│       │   ├── 1.2-loading-datasets.ipynb
│       │   └── 1.7-custom-dataset.ipynb
│       ├── week2/
│       │   ├── 2.3-first-finetuning.ipynb
│       │   ├── 2.5-custom-metrics.ipynb
│       │   └── 2.7-experiment-tracking.ipynb
│       └── week3/
│           ├── 3.2-comprehensive-eval.ipynb
│           ├── 3.4-nlg-evaluation.ipynb
│           └── 3.7-publish-model.ipynb
├── src/
│   └── hf_ecosystem/
│       ├── __init__.py
│       ├── hub/
│       │   ├── __init__.py
│       │   ├── search.py          # Model/dataset search utilities
│       │   └── cards.py           # Model card parsing
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── pipelines.py       # Pipeline wrappers
│       │   └── device.py          # Device management
│       ├── data/
│       │   ├── __init__.py
│       │   ├── preprocessing.py   # Tokenization utilities
│       │   └── streaming.py       # Dataset streaming
│       └── training/
│           ├── __init__.py
│           ├── trainer.py         # Trainer utilities
│           └── metrics.py         # Custom metrics
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── notebooks/
├── pyproject.toml                 # uv/Python config
├── uv.lock                        # Locked dependencies
├── Makefile                       # Build targets
├── .pmat-metrics.toml             # Quality thresholds
├── .pre-commit-config.yaml        # Pre-commit hooks
├── .gitignore
├── LICENSE                        # Apache-2.0
├── README.md
└── CLAUDE.md                      # AI assistant instructions
```

### 3.2 Source Module Organization

```python
# src/hf_ecosystem/__init__.py
"""Hugging Face Ecosystem utilities for Courses 1-2."""

__version__ = "1.0.0"

from hf_ecosystem.hub import search_models, parse_model_card
from hf_ecosystem.inference import create_pipeline, get_device
from hf_ecosystem.data import preprocess_dataset, stream_dataset
from hf_ecosystem.training import create_trainer, compute_metrics
```

---

## 4. Toolchain Specification

### 4.1 uv Configuration

```toml
# pyproject.toml
[project]
name = "hf-hub-ecosystem"
version = "1.0.0"
description = "Hugging Face Hub and Ecosystem Fundamentals - Courses 1-2"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.11"
authors = [
    { name = "PAIML Research Team" }
]

dependencies = [
    # Core HuggingFace
    "transformers>=4.40.0",
    "datasets>=2.18.0",
    "evaluate>=0.4.0",
    "accelerate>=0.28.0",
    "huggingface-hub>=0.22.0",

    # Tokenization
    "tokenizers>=0.15.0",
    "sentencepiece>=0.2.0",

    # Multi-modal
    "Pillow>=10.0.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",

    # Notebooks
    "jupyterlab>=4.1.0",
    "ipywidgets>=8.1.0",

    # Experiment tracking
    "tensorboard>=2.16.0",
    "wandb>=0.16.0",

    # Utilities
    "tqdm>=4.66.0",
    "pandas>=2.2.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.5.0",
    "nbval>=0.11.0",
    "ruff>=0.3.0",
    "pyright>=1.1.350",
    "pre-commit>=3.6.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "strict"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src --cov-report=term-missing"
```

### 4.2 Dependency Locking

```bash
# Lock all dependencies (reproducible installs)
uv lock

# Sync environment from lock file
uv sync

# Add new dependency
uv add transformers

# Add dev dependency
uv add --dev pytest
```

### 4.3 Environment Variables

```bash
# .env (not committed)
HF_HOME=/path/to/cache           # Model cache directory
HF_TOKEN=hf_xxxxx                # Hub authentication
WANDB_API_KEY=xxxxx              # Weights & Biases
CUDA_VISIBLE_DEVICES=0           # GPU selection
```

---

## 5. Notebook Standards

### 5.1 Notebook Structure

Every notebook follows this structure:

```python
# Cell 1: Metadata (markdown)
"""
# Lab X.Y: Title

**Objective**: Clear statement of what student will learn
**Duration**: XX minutes
**Prerequisites**: List of prior labs

## Learning Outcomes
- Outcome 1
- Outcome 2
"""

# Cell 2: Environment Setup
import sys
sys.path.insert(0, "../../src")

from hf_ecosystem import __version__
print(f"hf-ecosystem version: {__version__}")

# Cell 3: Imports
from transformers import pipeline, AutoModel, AutoTokenizer
from datasets import load_dataset
import torch

# Cell 4-N: Content cells with markdown headers

# Cell N+1: Verification
def verify_lab():
    """Verify lab completion criteria."""
    assert "model" in dir(), "Model not defined"
    assert "results" in dir(), "Results not computed"
    print("✅ Lab completed successfully!")

verify_lab()

# Cell N+2: Cleanup
del model  # Free GPU memory
torch.cuda.empty_cache()
```

### 5.2 Naming Convention

```
{lesson_number}-{kebab-case-title}.ipynb

Examples:
1.3-exploring-hub.ipynb
2.5-custom-inference.ipynb
3.4-whisper-transcription.ipynb
```

### 5.3 Cell Tags

Notebooks use cell tags for execution control:

| Tag | Purpose |
|-----|---------|
| `skip-ci` | Skip in CI (long-running or GPU-only) |
| `raises-exception` | Expected to raise error (for teaching) |
| `solution` | Hidden in student version |
| `exercise` | Student completes this cell |

### 5.4 Output Requirements

- All cells must have executed outputs committed
- No error outputs (except `raises-exception` tagged cells)
- Plots must be rendered inline
- Large outputs truncated to 50 lines max

---

## 6. Quality Gates

### 6.1 Makefile Targets

```makefile
# Makefile
.PHONY: all setup lint format test check notebooks clean

PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
PYRIGHT := uv run pyright
JUPYTER := uv run jupyter

# Default target
all: check

# Setup development environment
setup:
	uv sync --all-extras
	uv run pre-commit install
	@echo "✅ Environment ready"

# Linting (no fixes)
lint:
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYRIGHT)
	@echo "✅ Lint passed"

# Auto-format code
format:
	$(RUFF) check --fix .
	$(RUFF) format .
	@echo "✅ Formatted"

# Run all tests
test:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=95
	@echo "✅ Tests passed"

# Test notebooks execute without error
test-notebooks:
	$(PYTEST) --nbval-lax notebooks/ -v
	@echo "✅ Notebooks validated"

# Full quality check (CI equivalent)
check: lint test test-notebooks
	uv run pmat comply check
	@echo "✅ All checks passed"

# PMAT compliance
comply:
	uv run pmat comply check

# Run specific notebook
run-notebook:
	$(JUPYTER) lab $(NOTEBOOK)

# Clean generated files
clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	rm -rf **/__pycache__ **/*.pyc
	find notebooks -name "*.ipynb" -exec jupyter nbconvert --clear-output {} \;
	@echo "✅ Cleaned"
```

### 6.2 Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: local
    hooks:
      - id: pyright
        name: pyright
        entry: uv run pyright
        language: system
        types: [python]
        pass_filenames: false

      - id: pmat-comply
        name: pmat comply
        entry: uv run pmat comply check
        language: system
        pass_filenames: false
        stages: [commit]
```

### 6.3 PMAT Configuration

```toml
# .pmat-metrics.toml
[project]
name = "hf-hub-ecosystem"
type = "python-notebook"

[quality]
test_coverage_min = 95
complexity_max = 10
doc_coverage_min = 80

[notebooks]
max_cells = 50
max_output_lines = 100
require_verification_cell = true

[dependencies]
allow_floating = false  # All deps must be locked
max_direct_deps = 30

[ci]
require_passing = true
timeout_minutes = 30
```

### 6.4 GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  UV_CACHE_DIR: /tmp/.uv-cache
  HF_HOME: /tmp/.hf-cache

jobs:
  check:
    name: Quality Gates
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v1
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Cache dependencies
        uses: actions/cache@v4
        with:
          path: |
            /tmp/.uv-cache
            /tmp/.hf-cache
          key: ${{ runner.os }}-uv-${{ hashFiles('uv.lock') }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Lint
        run: |
          uv run ruff check .
          uv run ruff format --check .
          uv run pyright

      - name: Test
        run: uv run pytest tests/ -v --cov=src --cov-report=xml --cov-fail-under=95

      - name: Test Notebooks
        run: uv run pytest --nbval-lax notebooks/ -v --ignore=notebooks/**/*gpu*.ipynb

      - name: PMAT Comply
        run: uv run pmat comply check

      - name: Upload Coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
```

---

## 7. Popperian Falsification Criteria

### 7.1 Validation Matrix (100 Points)

| Section | Points | Criterion | Falsification Threshold |
|---------|--------|-----------|------------------------|
| **7.2 Environment** | 15 | Reproducible setup | Any dependency conflict = FAIL |
| **7.3 Test Coverage** | 20 | Code coverage ≥95% | <95% = FAIL |
| **7.4 Notebook Execution** | 25 | All notebooks run | Any execution error = FAIL |
| **7.5 Type Safety** | 10 | Pyright strict mode | Any type error = FAIL |
| **7.6 Lint Compliance** | 10 | Zero ruff violations | Any violation = FAIL |
| **7.7 Hub API Coverage** | 10 | All Hub operations | Missing operation = FAIL |
| **7.8 Documentation** | 10 | All public APIs documented | Missing docstring = FAIL |

**Pass Threshold:** ≥90 points with no critical failures

### 7.2 Environment Reproducibility (15 points)

```bash
# Falsification test
rm -rf .venv uv.lock
uv lock
uv sync
uv run pytest tests/ -v

# PASS: All tests pass with fresh lock
# FAIL: Any test failure or dependency resolution error
```

### 7.3 Test Coverage (20 points)

```bash
# Falsification test
uv run pytest tests/ --cov=src --cov-fail-under=95

# PASS: Coverage ≥95%
# FAIL: Coverage <95%
```

### 7.4 Notebook Execution (25 points)

```bash
# Falsification test
uv run pytest --nbval-lax notebooks/ -v

# PASS: All notebooks execute without error
# FAIL: Any notebook raises exception (except tagged cells)
```

### 7.5 Type Safety (10 points)

```bash
# Falsification test
uv run pyright --outputjson | jq '.generalDiagnostics | length'

# PASS: Zero diagnostics
# FAIL: Any type error in strict mode
```

### 7.6 Lint Compliance (10 points)

```bash
# Falsification test
uv run ruff check . --output-format=json | jq 'length'

# PASS: Zero violations
# FAIL: Any lint violation
```

### 7.7 Hub API Coverage (10 points)

Required operations in notebooks:

| Operation | Notebook | Status |
|-----------|----------|--------|
| `list_models()` | 1.3 | Required |
| `model_info()` | 1.3 | Required |
| `list_datasets()` | 1.7 | Required |
| `load_dataset(streaming=True)` | 1.7 | Required |
| `push_to_hub()` | 3.7 | Required |
| `create_repo()` | 3.7 | Required |

### 7.8 Documentation (10 points)

```bash
# Falsification test
uv run pmat doc-coverage src/

# PASS: ≥80% public API documented
# FAIL: <80% documentation coverage
```

---

## 8. Lab Specifications

### 8.1 Course 1: Hub and Ecosystem

#### Week 1: Hub Navigation

| Lab | Title | Duration | Key Concepts |
|-----|-------|----------|--------------|
| 1.3 | Exploring the Hub | 20 min | `list_models()`, filters, model cards |
| 1.7 | Dataset Exploration | 25 min | `load_dataset()`, streaming, splits |

#### Week 2: Transformers Fundamentals

| Lab | Title | Duration | Key Concepts |
|-----|-------|----------|--------------|
| 2.2 | Text Pipelines | 30 min | `pipeline()`, task types, batching |
| 2.5 | Custom Inference | 25 min | `AutoModel`, `AutoTokenizer`, forward pass |
| 2.8 | Multi-GPU Inference | 20 min | `device_map="auto"`, memory estimation |

#### Week 3: Multi-Modal Models

| Lab | Title | Duration | Key Concepts |
|-----|-------|----------|--------------|
| 3.2 | Image Classification | 25 min | ViT, CLIP, preprocessing |
| 3.4 | Speech-to-Text | 25 min | Whisper, chunking, language detection |
| 3.6 | Image Captioning | 30 min | BLIP, LLaVA, VQA |

### 8.2 Course 2: Fine-Tuning

#### Week 1: Data Preparation

| Lab | Title | Duration | Key Concepts |
|-----|-------|----------|--------------|
| 1.2 | Loading Datasets | 25 min | Hub, local files, formats |
| 1.4 | Preprocessing Pipeline | 30 min | `map()`, tokenization, batching |
| 1.7 | Custom Dataset | 25 min | `from_dict()`, `from_generator()` |

#### Week 2: Trainer API

| Lab | Title | Duration | Key Concepts |
|-----|-------|----------|--------------|
| 2.3 | First Fine-Tuning | 35 min | `Trainer`, `TrainingArguments` |
| 2.5 | Custom Metrics | 30 min | `compute_metrics`, F1, weighted loss |
| 2.7 | Experiment Tracking | 25 min | TensorBoard, W&B integration |

#### Week 3: Evaluation & Sharing

| Lab | Title | Duration | Key Concepts |
|-----|-------|----------|--------------|
| 3.2 | Comprehensive Eval | 25 min | `evaluate`, confusion matrix |
| 3.4 | NLG Evaluation | 25 min | ROUGE, BLEU, perplexity |
| 3.7 | Publish Model | 30 min | `push_to_hub()`, model cards |

---

## 9. Peer-Reviewed Citations

### 9.1 Foundational Papers

1. **Transformers Architecture**
   - Vaswani et al. (2017). "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762

2. **BERT**
   - Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL 2019. arXiv:1810.04805

3. **Vision Transformer**
   - Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words." ICLR 2021. arXiv:2010.11929

4. **Whisper**
   - Radford et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." ICML 2023. arXiv:2212.04356

5. **CLIP**
   - Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021. arXiv:2103.00020

### 9.2 Tokenization

6. **BPE**
   - Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units." ACL 2016. arXiv:1508.07909

7. **SentencePiece**
   - Kudo & Richardson (2018). "SentencePiece: A simple and language independent subword tokenizer." EMNLP 2018. arXiv:1808.06226

### 9.3 Training & Evaluation

8. **AdamW**
   - Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization." ICLR 2019. arXiv:1711.05101

9. **ROUGE**
   - Lin (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." ACL Workshop.

10. **BLEU**
    - Papineni et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL 2002.

---

## Appendices

### A. Model Reference Table

| Task | Recommended Model | Size | Hub ID |
|------|-------------------|------|--------|
| Text Classification | DistilBERT | 66M | `distilbert-base-uncased` |
| Sentiment Analysis | DistilBERT SST-2 | 66M | `distilbert-base-uncased-finetuned-sst-2-english` |
| Text Generation | GPT-2 | 124M | `gpt2` |
| Summarization | BART | 140M | `facebook/bart-base` |
| Translation | MarianMT | 74M | `Helsinki-NLP/opus-mt-en-de` |
| Image Classification | ViT | 86M | `google/vit-base-patch16-224` |
| Speech Recognition | Whisper | 39M | `openai/whisper-tiny` |
| Image Captioning | BLIP | 247M | `Salesforce/blip-image-captioning-base` |

### B. Hardware Requirements

| Configuration | RAM | GPU | Notebooks Supported |
|---------------|-----|-----|---------------------|
| **Minimum** | 8 GB | None (CPU) | Week 1-2 (Course 1) |
| **Recommended** | 16 GB | 8 GB VRAM | All except multi-GPU |
| **Full** | 32 GB | 24 GB VRAM | All notebooks |

### C. Glossary

| Term | Definition |
|------|------------|
| **Hub** | Hugging Face model and dataset repository |
| **Pipeline** | High-level inference API |
| **AutoClass** | Automatic model/tokenizer loading |
| **Trainer** | High-level training API |
| **Model Card** | Documentation for ML models |
| **SafeTensors** | Safe tensor serialization format |

---

*Last Updated: 2026-01-29*
*Document Version: 1.0.0*
