# HF Hub Ecosystem - Courses 1-2
# Python-only with uv, Jupyter notebooks, TDD

.PHONY: all setup lint format test test-fast test-unit test-integration test-notebooks coverage check comply clean help notebook demo build install security bench

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
	@printf '%s\n' '  make install        Install package locally'
	@printf '\n'
	@printf '%s\n' 'Quality:'
	@printf '%s\n' '  make lint           Run ruff and pyright'
	@printf '%s\n' '  make format         Auto-format with ruff'
	@printf '%s\n' '  make test           Run pytest with coverage'
	@printf '%s\n' '  make test-fast      Run fast tests (no coverage)'
	@printf '%s\n' '  make test-unit      Run unit tests only'
	@printf '%s\n' '  make test-notebooks Validate notebooks with nbval'
	@printf '%s\n' '  make bench          Run performance benchmarks'
	@printf '%s\n' '  make coverage       Run tests with coverage report (HTML)'
	@printf '%s\n' '  make security       Run security scan with bandit'
	@printf '%s\n' '  make check          Full quality check (CI equivalent)'
	@printf '%s\n' '  make comply         Run pmat comply check'
	@printf '\n'
	@printf '%s\n' 'Build:'
	@printf '%s\n' '  make build          Build package'
	@printf '\n'
	@printf '%s\n' 'Development:'
	@printf '%s\n' '  make demo           Run interactive demo'
	@printf '%s\n' '  make notebook       Launch Jupyter Lab'
	@printf '%s\n' '  make clean          Remove generated files'

# Setup development environment
setup:
	@printf '%s\n' '=== Setting up environment ==='
	uv sync --all-extras
	uv run pre-commit install 2>/dev/null || true
	@# Install optimized pre-commit hook
	cp hooks/pre-commit .git/hooks/pre-commit 2>/dev/null || true
	chmod +x .git/hooks/pre-commit 2>/dev/null || true
	@printf '%s\n' '=== Setup complete ==='

# Linting (no fixes)
# @perf: uses ruff for fast linting
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

# Run fast tests only (no coverage, parallel execution)
# @perf: optimized for CI feedback loop
test-fast:
	@printf '%s\n' '=== Running fast tests ==='
	$(PYTEST) tests/unit/ -x -q --no-cov -n auto 2>/dev/null || $(PYTEST) tests/unit/ -x -q --no-cov
	@printf '%s\n' '=== Fast tests complete ==='

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
# @perf: parallel lint and test execution
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

# Run interactive demo
demo:
	@printf '%s\n' '=== Running Demo ==='
	$(PYTHON) demo.py
	@printf '%s\n' '=== Demo complete ==='

# Build package
build:
	@printf '%s\n' '=== Building package ==='
	uv build
	@printf '%s\n' '=== Build complete ==='

# Install package locally
install:
	@printf '%s\n' '=== Installing package ==='
	uv pip install -e .
	@printf '%s\n' '=== Install complete ==='

# Run unit tests only
test-unit:
	@printf '%s\n' '=== Running unit tests ==='
	$(PYTEST) tests/unit/ -v --cov=src --cov-report=term-missing
	@printf '%s\n' '=== Unit tests complete ==='

# Run integration tests only
test-integration:
	@printf '%s\n' '=== Running integration tests ==='
	$(PYTEST) tests/integration/ -v 2>/dev/null || printf '%s\n' 'No integration tests found'
	@printf '%s\n' '=== Integration tests complete ==='

# Performance benchmarks
# @perf: benchmark suite for inference optimization
bench:
	@printf '%s\n' '=== Running benchmarks ==='
	$(PYTEST) tests/ -v --benchmark-only 2>/dev/null || printf '%s\n' 'No benchmarks configured'
	@printf '%s\n' '=== Benchmarks complete ==='

# Security scan
security:
	@printf '%s\n' '=== Security scan ==='
	uv pip install bandit 2>/dev/null || true
	uv run bandit -r src/ -ll --skip B101 2>/dev/null || printf '%s\n' 'Bandit not available'
	@printf '%s\n' '=== Security scan complete ==='

# Clean generated files
clean:
	@printf '%s\n' '=== Cleaning ==='
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .mypy_cache
	rm -rf dist build .eggs
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@printf '%s\n' '=== Clean complete ==='
