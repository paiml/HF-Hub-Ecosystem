# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Courses**: Hugging Face Hub and Ecosystem Fundamentals (Course 1) + Fine-Tuning Transformers (Course 2)
**Stack**: Python-only with uv, Jupyter notebooks
**Specification**: `docs/specifications/demo-repo.md`

## Quick Reference

```bash
# Setup
make setup

# Quality gates
make lint                    # ruff + pyright
make test                    # pytest with 100% coverage
make coverage                # pytest with HTML coverage report
make test-notebooks          # nbval (skips GPU notebooks)
make check                   # lint + test (CI equivalent)

# Run single test file
uv run pytest tests/unit/test_device.py -v

# Run single test function
uv run pytest tests/unit/test_device.py::TestGetDevice::test_get_device_returns_string -v

# Development
make notebook                # Launch Jupyter Lab
make format                  # Auto-fix with ruff
make comply                  # PMAT compliance check
```

## Development Rules

### Python/uv Only
- Use `uv` for all dependency management
- No pip, conda, or poetry
- Lock dependencies with `uv lock`

### TDD Workflow
1. Write failing test first (RED)
2. Implement minimum code to pass (GREEN)
3. Refactor while tests pass (REFACTOR)

### Notebook Standards
- All notebooks must execute without error
- Include verification cell at end
- Use cell tags: `skip-ci`, `raises-exception`, `solution`, `exercise`

### Quality Requirements
- 100% test coverage (enforced by CI)
- Zero ruff violations
- Pyright basic mode (third-party stubs relaxed)
- `pmat comply check` must pass

## Architecture

### Source Modules (`src/hf_ecosystem/`)

| Module | Purpose |
|--------|---------|
| `hub/` | Hub API wrappers: `search_models()`, `search_datasets()`, `parse_model_card()` |
| `inference/` | Pipeline creation and device management: `create_pipeline()`, `get_device()` |
| `data/` | Dataset preprocessing and streaming: `preprocess_text()`, `tokenize_batch()`, `stream_dataset()` |
| `training/` | Trainer utilities and metrics: `create_trainer()`, `compute_metrics()` |

All public APIs exported via `hf_ecosystem.__init__`.

### Notebooks Structure

```
notebooks/
├── course1/          # Hub and Ecosystem (weeks 1-3)
│   └── week{1,2,3}/  # Hub navigation, transformers, multi-modal
└── course2/          # Fine-Tuning (weeks 1-3)
    └── week{1,2,3}/  # Data prep, Trainer API, evaluation
```

Naming: `{lesson_number}-{kebab-case-title}.ipynb` (e.g., `2.5-custom-inference.ipynb`)

## Commit Messages

Must reference PMAT work item (see `.pmat-work/` for active contracts):
```
feat: Add text pipelines notebook (Refs PMAT-004)
```
