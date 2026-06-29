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
import gcphotom as gcp
import matplotlib.pyplot as plt

# 1. Simulate a realistic astronomical image
image, sim_cat = gcp.simulate_image(n_sources=100, background=100, read_noise=5)

# 2. Detect sources and build segmentation image (now takes care of background estimation)
seg, det_cat = gcp.detect_and_segment(image)

# 3. Extract growth curves with contamination estimation (takes care of error estimation)
cog = gcp.extract_growth_curves(
    image,
    det_cat,
    segmentation_image=seg
)

# 4. Fit all growth curves
fitter = gcp.Fitter(cog)
best_fit, extra = fitter.fit()
fitted = fitter.results(best_fit)

# 5. Match (return the matched reordered version of the sim_cat)
input_cat = gcp.cross_match(det_cat, sim_cat)

# 6. Inspect results
print(f"PSF: gamma={fitted['gamma']:.2f}, alpha={fitted['alpha']:.2f}")
plt.errorbar(input_cat['flux'], fitted['flux'] / input_cat['flux'], fitted['std_errors']['flux']/input_cat['flux'], marker='o', ls='None')
```

The `segmentation_image` enables contamination estimation by masking out neighboring sources. The result always includes `flux_clean` and `contamination`. When no segmentation is provided, `flux_clean` equals `flux` and `contamination` is zero. Cross-matching detected and simulated catalogs returns a matched table of the same length as the detected catalog (NaNs for unmatched). `Fitter.results` always returns per-source arrays aligned to the original input length (NaNs for internally dropped sources).

# CLI

A command-line interface is provided for processing survey forced-photometry catalogs:

```bash
# Match and show statistics
gcphotom snls match "catalog_forced_D1_g_*.npy" \
    --reference avg_cat_D1.npy --band g --min-flux 10000

# Fit growth curves and save results
gcphotom snls process "catalog_forced_D1_g_*.npy" \
    --reference avg_cat_D1.npy --band g --min-flux 10000 \
    --output-dir ./mophot/ --learning-rate 1e-2 --niter 2000
```

The `snls` subcommand loads SNLS forced-photometry catalogs (record arrays with `apfl_*`, `apvar_*`, `apother_*` columns), selects star entries by matching against a reference catalog via the `bindex` field, and fits a Moffat growth-curve model to each source. Fitted columns (`mflux`, `mback`, `mgoods`, `mchi2`) are appended to the catalog and saved as `.npy`.

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
