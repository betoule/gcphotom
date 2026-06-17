**gcphotom** – *Differentiable Growth Curve Photometry with Profile Fitting*

`gcphotom` is a Python package dedicated to high-precision aperture photometry of astronomical images. It performs photometry by fitting all growth curves (curves of growth of the aperture flux with radius) using an analytical profile learned on all the stars of an image. Built on **JAX**, it provides fast, differentiable, vectorized computations with native support for GPU and TPU acceleration.

# Features
- **Growth curve fitting** using the analytical Moffat PSF model for accurate total flux, and local background estimation.
- **Differentiable implementation** powered by JAX — ideal for gradient-based optimization, or integration into larger differentiable pipelines.
- High performance through JIT compilation and automatic vectorization.
- Seamless integration with the scientific Python ecosystem (Astropy, NumPy, Photutils).
- Automatic weighting to retrieve accurate fluxes for faint stars.
- Outlier rejection and automatic detection of contaminated apertures.
- Clean, modular API suitable for both interactive analysis and large-scale surveys.

# Installation

```bash
pip install gcphotom
```

**From source** (for development):
```bash
git clone https://github.com/betoule/gcphotom.git
cd gcphotom
pip install -e ".[dev]"
```

**Dependencies** (core): `jax`, `jaxlib`, `astropy`, `numpy`. GPU support requires the appropriate JAX CUDA/Metal/ROCM installation.

# Quick Start

```python
import numpy as np
import gcphotom as gcp

# 1. Simulate a realistic astronomical image with ~1000 stars
image, catalog = gcp.simulate_image()

# 2. Extract growth curves for each source
positions = np.column_stack([catalog["x"], catalog["y"]])
result = gcp.extract_growth_curves(image, positions)

# 3. Fit all growth curves with a common Moffat profile
# (Fitter class — coming soon)

# 4. Compare fitted fluxes to injected ground truth
```

# Why gcphotom?
Traditional aperture photometry often requires manual aperture corrections, suffer from aperture contamination from neighboring objects, and is suboptimal for faint objects due to background noise. On the other hand PSF photometry is typically limited to small radii so that accurate reconstruction of the total flux is difficult due to poor constraints on the PSF tails. `gcphotom` solves this by fitting the observed growth curve to a Moffat analytical profile, providing a robust estimate of total flux while remaining computationally efficient. 

## Related Work
The methodology builds upon foundational techniques in stellar photometry:
- Stetson, P. B. (1990). On the growth-curve method for calibrating stellar photometry with CCDs. *Publications of the Astronomical Society of the Pacific*, 102, 932.
- Moffat, A. F. J. (1969). A Theoretical Investigation of Focal Stellar Images in the Photographic Emulsion and Application to Photographic Photometry. *Astronomy & Astrophysics*, 3, 455.
- Bickerton, S. J. et al. (2013). A fast algorithm for precise aperture photometry of critically sampled images. *Monthly Notices of the Royal Astronomical Society*, 431(2), 1275.

`gcphotom` complements tools like **Photutils** (which provides `CurveOfGrowth` and `MoffatPSF`) by offering a dedicated, end-to-end differentiable fitter optimized for this specific workflow.

# Documentation
Full documentation is available at [Read the Docs](https://gcphotom.readthedocs.io) (link to be updated).

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
   uv venv --activate
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

- **Testing and Coverage**: The test suite must maintain **100% code coverage**.  
  Run tests with coverage using:
  ```bash
  uv run pytest --cov=gcphotom --cov-report=term-missing --cov-fail-under=95
  ```

  All new code must be accompanied by appropriate unit tests. Tests are located in the `tests/` directory and use `pytest`.

## Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes, ensuring compliance with formatting, linting, and 100% coverage requirements.
3. Update documentation and changelog if necessary.
4. Submit a pull request with a clear description of the changes and their rationale.
5. Ensure all CI checks pass (formatting, linting, tests, and coverage).

## Additional Notes

- JAX-specific code should prefer explicit `jax.jit`, `jax.vmap`, and `jax.grad` usage where appropriate.
- Prefer vectorized operations and maintain compatibility with CPU, GPU, and TPU execution.
- Major changes or new features should be discussed via an issue before implementation.

By contributing, you agree that your contributions will be licensed under the same GPLv3 License as the project.

## License

This project is licensed under the GPLv3 License.

