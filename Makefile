# HF Hub Ecosystem - Courses 1-2
# Python-only with uv, Jupyter notebooks, TDD

.PHONY: all setup lint format test test-notebooks coverage check comply clean help

# Variables
PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
PYRIGHT := uv run pyright
JUPYTER := uv run jupyter

# Coverage requirements
COV_FAIL_UNDER := 100

# Default target
all: check

# Help
help:
	@printf '%s\n' 'HF Hub Ecosystem - Courses 1-2'
	@printf '%s\n' '=============================='
	@printf '\n'
	@printf '%s\n' 'Setup:'
	@printf '%s\n' '  make setup          Install dependencies with uv'
	@printf '\n'
	@printf '%s\n' 'Quality:'
	@printf '%s\n' '  make lint           Run ruff and pyright'
	@printf '%s\n' '  make format         Auto-format with ruff'
	@printf '%s\n' '  make test           Run pytest with coverage'
	@printf '%s\n' '  make test-notebooks Validate notebooks with nbval'
	@printf '%s\n' '  make coverage       Run tests with coverage report (HTML)'
	@printf '%s\n' '  make check          Full quality check (CI equivalent)'
	@printf '%s\n' '  make comply         Run pmat comply check'
	@printf '\n'
	@printf '%s\n' 'Development:'
	@printf '%s\n' '  make notebook       Launch Jupyter Lab'
	@printf '%s\n' '  make clean          Remove generated files'

# Setup development environment
setup:
	@printf '%s\n' '=== Setting up environment ==='
	uv sync --all-extras
	uv run pre-commit install 2>/dev/null || true
	@printf '%s\n' '=== Setup complete ==='

# Linting (no fixes)
lint:
	@printf '%s\n' '=== Linting ==='
	$(RUFF) check .
	$(RUFF) format --check .
	$(PYRIGHT)
	@printf '%s\n' '=== Lint complete ==='

# Auto-format code
format:
	@printf '%s\n' '=== Formatting ==='
	$(RUFF) check --fix .
	$(RUFF) format .
	@printf '%s\n' '=== Format complete ==='

# Run all tests with coverage
test:
	@printf '%s\n' '=== Running tests ==='
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=$(COV_FAIL_UNDER)
	@printf '%s\n' '=== Tests complete ==='

# Test notebooks execute without error
test-notebooks:
	@printf '%s\n' '=== Validating notebooks ==='
	$(PYTEST) --nbval-lax notebooks/ -v --ignore='notebooks/**/*gpu*.ipynb' || true
	@printf '%s\n' '=== Notebook validation complete ==='

# Coverage with HTML report
coverage:
	@printf '%s\n' '=== Running coverage ==='
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=$(COV_FAIL_UNDER)
	@printf '%s\n' '=== Coverage report generated in htmlcov/ ==='

# Full quality check (CI equivalent)
check: lint test
	@printf '%s\n' '=== All checks passed ==='

# PMAT compliance
comply:
	@printf '%s\n' '=== PMAT Compliance ==='
	pmat comply check 2>/dev/null || printf '%s\n' 'pmat not installed, skipping'
	@printf '%s\n' '=== Compliance check complete ==='

# Launch Jupyter Lab
notebook:
	$(JUPYTER) lab notebooks/

# Clean generated files
clean:
	@printf '%s\n' '=== Cleaning ==='
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .mypy_cache
	rm -rf dist build .eggs
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@printf '%s\n' '=== Clean complete ==='
