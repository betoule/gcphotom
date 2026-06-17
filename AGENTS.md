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
  - Ask for confirmation before adding a ignore rule to the pyproject.toml

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

## Behavioral guidelines

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
Propose a test strategy for each task and ask for validation.

2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
Avoid meaningless tests. Try to minimize the number of tests while keeping a good coverage.

3. Build incrementally
After each substantial milestone:

Stop building.
Commit making sure all tests passes and linting is happy.
Provide a quick summary, with a snippet of code demonstrating the usage of what was just developed.
Provide a short critical evaluation of what has been done.
Wait for feedback.
Once feedback is provided, work on implementing fixes and suggestions.
Do not jump to the next task until instructed to do so.

