# Code Quality and Linting

This project uses pre-commit hooks to ensure code quality and consistency. The hooks automatically run:

1. Ruff for linting and code quality checks
2. Ruff formatter for code formatting

## Setup

The pre-commit hooks are automatically installed when you set up the project. If you need to reinstall them, run:

```bash
pre-commit install
```

## Running Linting Manually

You can run the linters manually at any time:

```bash
# Run ruff linter
ruff check .

# Run ruff formatter
ruff format .

# Run both
ruff check . --fix && ruff format .
```

## Configuration

- Ruff configuration is in `pyproject.toml`
- Pre-commit configuration is in `.pre-commit-config.yaml`

## Pre-commit Hooks

The hooks will automatically run on every commit. If they find and fix issues, you'll need to stage the changes and commit again.

To run pre-commit on all files manually:

```bash
pre-commit run --all-files
```
