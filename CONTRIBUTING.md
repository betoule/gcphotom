# Contributing

We welcome contributions to `gcphotom`. Please follow the guidelines below to ensure a smooth processing of your pull requests.

## Development Environment Setup

The project is managed with [**uv**](https://docs.astral.sh/uv/), a modern, fast Python package manager.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/betoule/gcphotom.git
   cd gcphotom
   ```

2. **Create and activate the development environment**:
   ```bash
   uv sync --all-extras --dev
   ```

3. **Install pre-commit hooks** (recommended):
   ```bash
   uv run pre-commit install
   ```

## Code Standards

- **Formatting**: All Python code must be formatted with [**Black**](https://black.readthedocs.io/).
  Run formatting with:
   ```bash
   uv run black src/ tests/
   ```

- **Linting**: Code is linted with [**pylint**](https://pylint.readthedocs.io/). The configuration is provided in `pyproject.toml`.
  Run linting with:
   ```bash
   uv run pylint src/ tests/
   ```

- **Testing and Coverage**: The test suite must maintain at least **95% code coverage**.
  Run tests with coverage using:
   ```bash
   uv run pytest --cov=gcphotom --cov-report=term-missing --cov-fail-under=95
   ```

  All new code must be accompanied by appropriate unit tests. Tests are located in the `tests/` directory and use `pytest`.

## Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes, ensuring compliance with formatting, linting, and coverage requirements.
3. Update documentation and changelog if necessary.
4. Submit a pull request with a clear description of the changes and their rationale.
5. Ensure all CI checks pass (formatting, linting, tests, and coverage).

## Additional Notes

- JAX-specific code should prefer explicit `jax.jit`, `jax.vmap`, and `jax.grad` usage where appropriate.
- Prefer vectorized operations and maintain compatibility with CPU, GPU, and TPU execution.
- Major changes or new features should be discussed via an issue before implementation.

By contributing, you agree that your contributions will be licensed under the same GPLv3 License as the project.
