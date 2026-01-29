# HF Hub Ecosystem - Courses 1-2

## Project Overview

**Courses**: Hugging Face Hub and Ecosystem Fundamentals (Course 1) + Fine-Tuning Transformers (Course 2)
**Stack**: Python-only with uv, Jupyter notebooks
**Specification**: `docs/specifications/demo-repo.md`

## Quick Reference

```bash
# Setup
make setup

# Quality gates
make lint
make test
make check

# Development
make notebook
make format
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
- 95% test coverage minimum
- Zero ruff violations
- Pyright strict mode (where practical)
- `pmat comply check` must pass

## File Organization

```
src/hf_ecosystem/       # Source modules
notebooks/              # Jupyter notebooks by course/week
tests/                  # pytest tests
docs/readings/          # Course readings (markdown)
docs/specifications/    # Technical specifications
```

## Commit Messages

Must reference work item:
```
feat: Add text pipelines notebook (Refs PMAT-004)
```
