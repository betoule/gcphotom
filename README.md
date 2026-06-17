**gcphotom** – *Differentiable Growth Curve Photometry with Profile Fitting*

`gcphotom` is a Python package dedicated to high-precision aperture photometry of astronomical images. It performs photometry by fitting all growth curves (curves of growth of the aperture flux with radius) using an analytical profile learned on all the stars of an image. Built on **JAX**, it provides fast, differentiable, vectorized computations with native support for GPU and TPU acceleration.

# Features

- **Growth curve fitting** using the analytical Moffat PSF model for accurate total flux, and local background estimation.
- **Aperture contamination estimation** via segmentation-based source detection — identifies flux from neighboring sources in each aperture.
- **Differentiable implementation** powered by JAX — ideal for gradient-based optimization, or integration into larger differentiable pipelines.
- High performance through JIT compilation and automatic vectorization.
- Seamless integration with the scientific Python ecosystem (Astropy, NumPy, Photutils).
- Automatic weighting to retrieve accurate fluxes for faint stars.
- Outlier rejection and automatic detection of contaminated apertures.
- Clean, modular API suitable for both interactive analysis and large-scale surveys.

# Installation

**From source**:
```bash
git clone https://github.com/betoule/gcphotom.git
cd gcphotom
uv sync --all-extras --dev
```

**Dependencies** (core): `jax`, `jaxlib`, `astropy`, `numpy`, `photutils`. GPU support requires the appropriate JAX CUDA/Metal/ROCM installation.

# Quick Start

```python
import numpy as np
import gcphotom as gcp

# 1. Simulate a realistic astronomical image
background = 100
read_noise = 5
image, catalog = gcp.simulate_image(background=background, read_noise=read_noise)

# 2. Detect sources and build segmentation image
seg = gcp.detect_and_segment(image, background=background)

# 3. Compute per-pixel error estimate
error = gcp.estimate_error(image, background=background, read_noise=read_noise)

# 4. Extract growth curves with contamination estimation
result = gcp.extract_growth_curves(
    image - background, seg["positions"],
    error=error,
    segmentation_image=seg["segmentation_image"],
    labels=seg["labels"]
)

# 5. Fit all growth curves with a common Moffat profile
fitter = gcp.Fitter(result)
best_params, extra = fitter.fit()

# 6. Match detected sources back to the input catalog
input_pos = np.column_stack([catalog["x"], catalog["y"]])
match = gcp.cross_match(input_pos, seg["positions"])
fitted = fitter.results(best_params)
matched = match["match_indices"] >= 0
matched_flux = fitted["flux"][match["match_indices"][matched]]

# 7. Inspect results
print(f"PSF: gamma={fitted['gamma']:.2f}, alpha={fitted['alpha']:.2f}")
print(f"Recovered / injected flux: {matched_flux[:5]} / {catalog['flux'][matched][:5]}")
print(f"Contamination: {result['contamination'][:, -1][match['match_indices'][matched][:5]]}")
```

The `segmentation_image` and `labels` parameters enable contamination estimation by masking out neighboring sources. The result includes `flux_clean` (flux with neighbors masked) and `contamination` (absolute contaminating flux).

# Why gcphotom?

Traditional aperture photometry often requires manual aperture corrections, suffer from aperture contamination from neighboring objects, and is suboptimal for faint objects due to background noise. On the other hand PSF photometry is typically limited to small radii so that accurate reconstruction of the total flux is difficult due to poor constraints on the PSF tails. `gcphotom` solves this by fitting the observed growth curve to a Moffat analytical profile, providing a robust estimate of total flux while remaining computationally efficient.

## Related Work

The methodology builds upon foundational techniques in stellar photometry:
- Stetson, P. B. (1990). On the growth-curve method for calibrating stellar photometry with CCDs. *Publications of the Astronomical Society of the Pacific*, 102, 932.
- Moffat, A. F. J. (1969). A Theoretical Investigation of Focal Stellar Images in the Photographic Emulsion and Application to Photographic Photometry. *Astronomy & Astrophysics*, 3, 455.
- Bickerton, S. J. et al. (2013). A fast algorithm for precise aperture photometry of critically sampled images. *Monthly Notices of the Royal Astronomical Society*, 431(2), 1275.

`gcphotom` complements tools like **Photutils** (which provides `CurveOfGrowth` and `MoffatPSF`) by offering a dedicated, end-to-end differentiable fitter optimized for this specific workflow.

# Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

# License

This project is licensed under the GPLv3 License.
