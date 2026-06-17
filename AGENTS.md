# GCPhotom Agent Guidelines

## Package Management
- Managed with **uv**. All dependency and environment operations go through uv.
- Create/activate environment: `uv sync --all-extras --dev`
- Run scripts: `uv run <command>`

## Code Standards
- **Formatting**: All Python code must be formatted with **black**.
  - Run: `uv run black src/ tests/`
- **Linting**: Code is linted with **pylint** (config in `pyproject.toml`).
  - Run: `uv run pylint src/ tests/`

## Pre-commit Hooks
- pre-commit hooks enforce black formatting and pylint linting on every commit.
- Install hooks: `uv run pre-commit install`
- Run manually: `uv run pre-commit run --all-files`

## Testing
- Tests use **pytest** and must maintain high coverage.
  - Run: `uv run pytest --cov=gcphotom --cov-report=term-missing --cov-fail-under=95`

## Project Structure
- Source code lives under `src/gcphotom/` (src layout).
- Tests live under `tests/`.
- The package is `gcphotom` — import as `import gcphotom as gcp`.

## Dependencies
- Core: `jax`, `jaxlib`, `astropy`, `numpy`
- GPU support requires appropriate JAX backend installation.
