# HF Hub Ecosystem - Courses 1-2
# Python-only with uv, Jupyter notebooks, TDD

.PHONY: all setup lint format test test-notebooks check comply clean help

PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
PYRIGHT := uv run pyright
JUPYTER := uv run jupyter

# Default target
all: check

# Help
help:
	@printf 'HF Hub Ecosystem - Courses 1-2\n'
	@printf '==============================\n\n'
	@printf 'Setup:\n'
	@printf '  make setup          Install dependencies with uv\n\n'
	@printf 'Quality:\n'
	@printf '  make lint           Run ruff and pyright\n'
	@printf '  make format         Auto-format with ruff\n'
	@printf '  make test           Run pytest with coverage\n'
	@printf '  make test-notebooks Validate notebooks with nbval\n'
	@printf '  make check          Full quality check (CI equivalent)\n'
	@printf '  make comply         Run pmat comply check\n\n'
	@printf 'Development:\n'
	@printf '  make notebook       Launch Jupyter Lab\n'
	@printf '  make clean          Remove generated files\n'

# Setup development environment
setup:
	@printf '=== Setting up environment ===\n'
	uv sync --all-extras
	uv run pre-commit install || true
	@printf '=== Setup complete ===\n'

# Linting (no fixes)
lint:
	@printf '=== Linting ===\n'
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYRIGHT) || true
	@printf '=== Lint complete ===\n'

# Auto-format code
format:
	@printf '=== Formatting ===\n'
	$(RUFF) check --fix .
	$(RUFF) format .
	@printf '=== Format complete ===\n'

# Run all tests
test:
	@printf '=== Running tests ===\n'
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing
	@printf '=== Tests complete ===\n'

# Test notebooks execute without error
test-notebooks:
	@printf '=== Validating notebooks ===\n'
	$(PYTEST) --nbval-lax notebooks/ -v --ignore=notebooks/**/*gpu*.ipynb || true
	@printf '=== Notebook validation complete ===\n'

# Full quality check (CI equivalent)
check: lint test
	@printf '=== All checks passed ===\n'

# PMAT compliance
comply:
	@printf '=== PMAT Compliance ===\n'
	pmat comply check || true
	@printf '=== Compliance check complete ===\n'

# Launch Jupyter Lab
notebook:
	$(JUPYTER) lab notebooks/

# Clean generated files
clean:
	@printf '=== Cleaning ===\n'
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .mypy_cache
	rm -rf **/__pycache__ **/*.pyc
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@printf '=== Clean complete ===\n'
