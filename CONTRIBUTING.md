# Contributing to HF Hub Ecosystem

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/paiml/HF-Hub-Ecosystem.git
cd HF-Hub-Ecosystem

# Install dependencies with uv
make setup
```

## Code Quality

We maintain strict code quality standards:

- **100% test coverage** required
- **ruff** for linting and formatting
- **pyright** for type checking

Run all checks before committing:

```bash
make check
```

## Making Changes

1. Create a feature branch from `main`
2. Make your changes
3. Run `make check` to verify quality gates pass
4. Submit a pull request

## Commit Messages

Follow conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test improvements
- `ci:` CI/CD changes
- `chore:` Maintenance tasks

## Testing

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/unit/test_device.py -v

# Generate coverage report
make coverage
```

## Pre-commit Hooks

Pre-commit hooks run automatically on commit:

```bash
# Install hooks (done automatically by make setup)
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

## Questions?

Open an issue if you have questions or need clarification.
