# Simulation framework

`gcphotom` provides two levels of simulation: **single-image simulation** for
quick experimentation, and **Monte Carlo simulation** for statistical bias
analysis across many random realizations.  Two rendering backends are
available: the original **astropy/photutils** backend and a **GalSim**-based
backend (FFT or photon shooting).

---

## 1. Two simulation modes

### 1.1 Single-image simulation (`gcphotom.simulate_image`)

Generates one synthetic astronomical image with known ground truth:

```python
image, catalog = gcp.simulate_image(
    shape=(1024, 1024), n_sources=1000,
    gamma=3.0, alpha=3.0, background=100.0, read_noise=5.0, seed=42,
)
```

Returns an `ndarray` image and an `astropy.table.Table` catalog with `x`, `y`,
`flux` columns.  Sources are placed at random positions with log-uniform
fluxes in [100, 1e6] ADU.  Each is rendered with a Moffat PSF via
`photutils.datasets.make_model_image`.  Poisson noise (source + background)
and Gaussian read noise are added.

The catalog generator (`make_realistic_source_catalog`) and image simulator
(`simulate_image`) live in `src/gcphotom/simulator.py`.

### 1.2 Single-image simulation with GalSim (`gcphotom.simulate_image_galsim`)

Drop-in alternative to `simulate_image` with the same signature, plus two
keyword-only parameters:

```python
from gcphotom.galsim_simulator import simulate_image_galsim

image, catalog = simulate_image_galsim(
    shape=(1024, 1024), n_sources=1000,
    gamma=3.0, alpha=3.0, background=100.0, read_noise=5.0, seed=42,
    method="auto",              # "auto" (FFT) or "phot" (photon shooting)
    max_phot_sources=100,       # batch size for photon shooting
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"auto"` | `"auto"` for FFT convolution (fast, noiseless before Poisson noise), `"phot"` for photon shooting (intrinsic Poisson noise) |
| `max_phot_sources` | `100` | Maximum sources per batch when `method="phot"`; reduces memory usage for large catalogs |

The module lives at `src/gcphotom/galsim_simulator.py` and is exported as
`gcp.simulate_image_galsim`.

#### Noise model comparison

| Component | `method="auto"` (FFT) | `method="phot"` (photon shooting) |
|---|---|---|
| Sources | Noiseless FFT render | Poisson photons per source |
| Background | Constant added, then `Poisson(total)` | Constant + separate `Poisson(background)` |
| Read noise | `Gaussian(0, read_noise)` | Same |
| Progress bar | None | `tqdm` over batches |

#### Coordinate convention

GalSim places world coordinate `(0, 0)` at the image centre.  The conversion
from 0-indexed FITS pixel `(x, y)` to GalMan shift `(dx, dy)` is:

```python
dx = x - nx/2 + 0.5
dy = y - ny/2 + 0.5
```

This is handled internally by `_fits_to_galsim_coords`.

### 1.3 Monte Carlo simulation (`gcphotom.montecarlo.MonteCarlo`)

Repeats the single-image simulation + photometry pipeline N times with
independent random catalogs:

```python
from gcphotom.montecarlo import SimulationConfig, MonteCarlo

cfg = SimulationConfig(n_sources=1000, shape=(1024, 1024), ...)
mc = MonteCarlo(cfg, n_realizations=100, seed=42)
results = mc.run()          # -> list[dict]
```

Each realisation:
1. Generates a catalog via `catalog_fn(seed)`.
2. Simulates an image via `simulate_fn(...)`.
3. Runs `run_pipeline` with the configured estimators.
4. Collects the result.

Failed realizations are caught, a warning is issued, and the loop continues.
The number of successful realizations is `len(results)`.

### `catalog_fn`

`MonteCarlo.__init__` accepts an optional `catalog_fn` parameter — a callable
`f(seed) -> Table` that produces the source catalog for each realisation.
Defaults to `partial(make_realistic_source_catalog, n_sources=..., shape=...)`.

This makes it easy to use a deterministic grid catalog or a Gaia-based catalog:

```python
from functools import partial

# Grid catalog (same sources every realisation, only noise varies)
catalog_fn = lambda seed: gcp.make_test_source_catalog(
    n_sources_side=7, shape=cfg.shape, fmin=100, fmax=1e6,
)

# Gaia-based catalog
from astropy.wcs import WCS
wcs = WCS(naxis=2)
wcs.wcs.crpix = [512.0, 512.0]
# ... configure WCS ...
catalog_fn = partial(gcp.make_gaia_source_catalog, wcs=wcs, shape=cfg.shape, zeropoint=25.0)

mc = MonteCarlo(cfg, n_realizations=50, seed=42, catalog_fn=catalog_fn)
```

### `simulate_fn`

`MonteCarlo.__init__` also accepts an optional `simulate_fn` parameter — a
callable with the same signature as `simulate_image(...)`.  Defaults to
`gcp.simulate_image`.  Pass a wrapped `simulate_image_galsim` to use the
GalSim backend:

```python
from functools import partial

simulate_fn = partial(gcp.simulate_image_galsim, method="phot", max_phot_sources=100)
mc = MonteCarlo(cfg, n_realizations=100, seed=42, simulate_fn=simulate_fn)
```

---

## 2. SimulationConfig

A lightweight dataclass that captures all parameters needed for one
realization:

| Field         | Type    | Default | Description                              |
|---------------|---------|---------|------------------------------------------|
| `n_sources`   | `int`   | 1000    | Number of sources per realisation        |
| `shape`       | `tuple` | 1024²   | Image shape `(ny, nx)`                   |
| `gamma`       | `float` | 3.0     | Moffat PSF scale parameter (pixels)     |
| `alpha`       | `float` | 3.0     | Moffat PSF shape parameter               |
| `background`  | `float` | 100.0   | Constant background level (ADU)          |
| `read_noise`  | `float` | 5.0     | Gaussian read noise σ (ADU)             |
| `n_pixels`    | `int`   | 5       | Background mesh for `detect_and_segment` |
| `fit_kwargs`  | `dict`  | `{lr=1e-2, niter=2000}` | Passed to `Fitter.fit()` |

Note that `simulate_fn` is passed to `MonteCarlo` directly, not stored in
`SimulationConfig`, to keep the dataclass backend-agnostic.

---

## 3. Estimator API

*(Unchanged — see existing documentation below for full details.)*

An estimator is a function `(image, detections, cog) -> dict` that performs
one photometry measurement.  The API, `@timed_estimator` decorator,
`detections` dict, and built-in estimators are documented in sections 4–5
below.

---

## 4. Built-in estimators

| Function | Key in `default_estimators()` | Description |
|----------|-------------------------------|-------------|
| `gc_estimator` | `"GC"` | Two-step growth-curve fit with free background.  Returns `flux`, `back`, `gamma`, `alpha`, `ngoods`, `chi2` and their standard errors. |
| `gc_fixed_back_estimator` | `"GC (fixed back)"` | Same but background is fixed to the detection background map at each source position. |
| `aperture_estimator` | `"Aperture + AC"` | Aperture photometry with aperture correction from bright isolated stars.  Uses a data-driven PSF correction from the flux ratio at two intermediate radii.  `best_fit` contains only `flux`. |
| `psf_estimator` | `"PSF"` | PSF photometry via `psf_photometry`.  `best_fit` contains only `flux`. |

### `default_estimators(cfg)`

Convenience function that returns the four estimators above with parameters
pre-bound from a `SimulationConfig`:

```python
from gcphotom.montecarlo import default_estimators

estimators = default_estimators(cfg)
# == {
#     "GC":              partial(gc_estimator, fit_kwargs=cfg.fit_kwargs),
#     "GC (fixed back)": partial(gc_fixed_back_estimator, fit_kwargs=cfg.fit_kwargs),
#     "PSF":             psf_estimator,
#     "Aperture + AC":   aperture_estimator,
# }
```

The PSF and aperture estimators use the background map from `detect_and_segment`
directly and do not need any pre-configured value.

---

## 5. Pipeline runner

`run_pipeline(image, sim_cat, cfg, estimators)` ties everything together:

1. Runs source detection (`detect_and_segment`).
2. Extracts growth curves.
3. Cross-matches the detection catalog to the truth catalog.
4. Calls each estimator in sequence.
5. Returns:

```python
{
    "sim_cat": <Table>,        # truth catalog matched to det_cat length
    "det_cat": <Table>,        # detection catalog
    "params":  <SimulationConfig>,
    "<name>":  <estimator result dict>,    # one per estimator
    ...
}
```

`sim_cat` is the truth catalog (`cross_match(det_cat, truth)`) so that
`sim_cat["flux"]` gives the true flux for each detected source (NaN for
unmatched sources).  This is the reference that `compute_flux_bias` uses.

---

## 6. Compute and plot

### `compute_flux_bias(results, estimators=None, nbins=10)`

Collects `best_fit["flux"]` vs `sim_cat["flux"]` across all realisations,
masks non-finite entries, concatenates, and computes per-bin median bias:

```
bias = (estimate / truth - 1) × 100     (percent)
```

Returns `{name: {"xbins": ..., "bias": ..., "bias_err": ...}}`.

When `estimators` is `None`, it auto-detects all estimator keys (every key
except `"sim_cat"`, `"det_cat"`, `"params"`).

### `plot_flux_bias(bias_stats, figsize=(7, 5))`

Single-panel plot of flux bias vs simulated flux for all estimators in
`bias_stats`.  Returns the `Axes` object.

### `plot_scalar_bias(results, params=("gamma", "alpha"), figsize=None)`

Per-realisation estimated values for scalar parameters (gamma, alpha) that some
estimators provide.  One panel per parameter, with a horizontal line at the
truth and error bars from `std_errors` when available.

### `plot_estimation_times(results, estimators=None, bins=20, figsize=(7, 5))`

Histogram of per-realisation wall-clock time for each estimator.

---

## 7. Save / load

Only MC results are persisted (computation of bias from results is fast and
done on the fly).

```python
from gcphotom.montecarlo import save_results, load_results

save_results("my_sim.pkl", results)       # .pkl appended if missing
loaded = load_results("my_sim.pkl")       # -> list[dict]
```

Uses `pickle.dump` / `pickle.load`.  Loaded results work directly with
`compute_flux_bias` and `plot_flux_bias`.

The `det_cat` saved in each realisation is a lightweight `~astropy.table.Table`
rather than the full `SourceCatalog`.

---

## 8. CLI example script

`examples/mc_bias.py` provides a full-featured CLI for running
and analysing Monte Carlo simulations.  Two subcommands:

### `run` — run a new simulation

The `--n-sources` option selects the source catalog type:
- An **integer** (e.g. `--n-sources 1000`) — synthetic catalog with that many sources per realisation.
- A **field name** (`COSMOS`, `Gemini`, `Cygnus`) — Gaia DR3 catalog for that field,
  with predefined pixel scale (1″/pix), zeropoint (25.0), and G<20 magnitude limit.

All other options (`--gamma`, `--simulator`, `--background`, …) work identically in both modes.

By default the `run` command saves results to an auto-named pickle file and does
**not** produce plots.  Pass `--plot` to also generate bias plots, or use the
`show` subcommand later on any saved result file.

GalSim photon shooting (`--simulator galsim-phot`) additionally supports
chromatic PSF rendering and LSST-like sensor effects — see section
[12](#12-chromatic-rendering-and-sensor-effects) for details.

Synthetic examples:

```bash
# Default (astropy, 1024×1024, 1000 sources, 100 realisations)
uv run python examples/mc_bias.py run

# GalSim photon shooting with custom parameters
uv run python examples/mc_bias.py run \
    --simulator galsim-phot \
    --ny 512 --nx 512 \
    --n-sources 500 \
    --n-realizations 50 \
    --gamma 2.5 --alpha 4.0 \
    --background 200 --read-noise 3 \
    --max-phot-sources 100 \
    --learning-rate 5e-3 --niter 3000 \
    --output my_results
```

Gaia examples (the three fields from the demo script):

```bash
# Low stellar density — COSMOS field (b ≈ +42°)
uv run python examples/mc_bias.py run --n-sources COSMOS

# Mid density — Gemini field (b ≈ +15°)
uv run python examples/mc_bias.py run --n-sources Gemini --n-realizations 50

# High density — Cygnus field (Galactic plane, b ≈ 0°) — with plots
uv run python examples/mc_bias.py run --n-sources Cygnus --plot
```

Output files (auto-named from non-default parameters):

| Command | Saved result | With `--plot` also saves |
|---------|-------------|--------------------------|
| `mc_bias.py run` | `mc_astropy.pkl` | `mc_flux_bias_astropy.png`, … |
| `mc_bias.py run --n-sources 500 --n-realizations 50` | `mc_astropy_r50_n500.pkl` | `mc_flux_bias_astropy.png`, … |
| `mc_bias.py run --n-sources COSMOS` | `mc_COSMOS.pkl` | `mc_flux_bias_COSMOS.png`, … |
| `mc_bias.py run --n-sources COSMOS -o my.pkl` | `my.pkl` | `mc_flux_bias_COSMOS.png`, … |

### `show` — re-plot from saved results

```bash
uv run python examples/mc_bias.py show my_results.pkl
```

Uses `gcp.montecarlo.load_results` to load the pickle and produces the same
three plots, tagged with the filename stem.

---

## 9. Adding a new estimator

1. Write a function with the `(image, detections, cog)` signature.
2. Decorate it with `@timed_estimator`.
3. Return `{"best_fit": ..., "uncertainty": ..., "extra": ...}`.
4. Include a `"flux"` array in `best_fit`, NaN-expanded to `det_cat` length.
5. Pass it in the estimators dict:

```python
mc = MonteCarlo(cfg, n_realizations=100, estimators={
    "GC": gc_estimator,
    "MyNew": my_new_estimator,
})
```

No other code changes are needed — `compute_flux_bias` and `plot_flux_bias`
auto-detect new estimator keys.

---

## 10. Data flow

```
SimulationConfig
       │
       ▼
MonteCarlo.run()
       │
       ├── for each realisation ──┐
       │   │                      │
       │   ├── catalog_fn(seed) ──┤  ← defaults to make_realistic_source_catalog
       │   │                      │    can be make_test_source_catalog,
       │   │                      │    make_gaia_source_catalog, or custom
       │   ├── simulate_fn(...) ──┤  ← defaults to simulate_image (astropy),
       │   │                      │    can be simulate_image_galsim (GalSim)
       │   ├── run_pipeline()
       │   │    ├── detect_and_segment()          → detections dict
       │   │    ├── extract_growth_curves()       → cog
       │   │    ├── cross_match(det_cat, truth)   → sim_cat
       │   │    └── estimator(img, detections, cog)  → result per estimator
       │   │                            (×N estimators)
       │   │
       │   └── append result dict (det_cat → lightweight Table)
       │
       └── return list[dict] ──┐
                                ▼
                     compute_flux_bias()  →  plot_flux_bias()
                     plot_scalar_bias()   →  plot_estimation_times()
                                │
                                ▼
                      save_results() / load_results()  (pickle)
```

Key extension points:
- **`catalog_fn`** — custom source catalog generation
- **`simulate_fn`** — custom image rendering (astropy, GalSim, or custom)
- **Estimator dict** — arbitrary number of photometry estimators

---

## 11. GalSim photon-shooting performance

The following table summarises typical wall-clock times for a single
realisation (1024×1024, 1000 sources, Moffat γ=3.0, α=3.0) on a modern
CPU.  GalSim photon-shooting time scales linearly with total flux and
source count; FFT time scales with image size.

| Backend | Method | Time | Notes |
|---------|--------|------|-------|
| astropy | `make_model_image` | ~2 s | Oversampled rendering, single-threaded |
| GalSim | `method="auto"` (FFT) | ~15 s | FFT convolution, suitable for large images |
| GalSim | `method="phot"` | ~30-60 s | Batch size 100; scales with total flux |

For 100-realisation Monte Carlo runs, plan for:
- **astropy**: ~3-4 minutes
- **GalSim FFT**: ~25 minutes
- **GalSim photon shooting**: ~1 hour

---

## 12. Chromatic rendering and sensor effects

GalSim-based photon shooting (`--simulator galsim-phot` with `--chromatic`)
supports wavelength-dependent PSF and detector physics.

### 12.1 Usage

```bash
# Chromatic photon shooting with top-hat r-band
uv run python examples/mc_bias.py run \
    --simulator galsim-phot --chromatic --bandpass r

# Chromatic + LSST-like sensor (brighter-fatter + charge diffusion)
uv run python examples/mc_bias.py run \
    --simulator galsim-phot --chromatic --bandpass r \
    --sensor --bf-strength 1.0 --diffusion-factor 1.0

# Zenith DCR with a realistic zenith angle (default 30°)
uv run python examples/mc_bias.py run \
    --simulator galsim-phot --chromatic --zenith-angle 30.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--chromatic` | `False` | Enable chromatic PSF + SED rendering (forces galsim-phot) |
| `--bandpass` | `"r"` | Top-hat bandpass: `g` (400–550 nm), `r` (550–700), `i` (700–820), `z` (820–920) |
| `--sensor` | `False` | Enable `SiliconSensor` (brighter-fatter + charge diffusion) |
| `--bf-strength` | `0.0` | Brighter-fatter strength (0 = off, 1 = LSST nominal) |
| `--diffusion-factor` | `0.0` | Charge diffusion factor (0 = off, 1 = LSST nominal) |
| `--zenith-angle` | `30.0` | Zenith angle in degrees for atmospheric DCR |

### 12.2 Architecture

The chromatic PSF is the convolution of two components, each independently
toggleable:

1. **Atmospheric PSF** (`--chromatic`, default on) — `ChromaticAtmosphere`
   wrapping a Moffat base at 500 nm.  Handles λ⁻⁰·² seeing scaling and
   differential chromatic refraction (DCR) at the specified zenith angle.
2. **Optical PSF** (`--chromatic`, default on) — `ChromaticOpticalPSF`
   modelling an 8.36 m telescope (LSST-like) with λ-dependent diffraction
   and Zernike aberrations.

The PSF is pre-computed via `interpolate(waves)` on a grid of 20
wavelengths for efficient batch rendering.

Source SEDs are blackbody spectra derived from the `bp_rp` colour column
in the catalog.  The SED is normalised to the catalog flux in the chosen
bandpass via `sed.withFlux(flux, bandpass=bp)`.

The sensor model (`--sensor`) uses GalSim's `SiliconSensor` with the
`lsst_itl_50_8` pixel grid (8 points per pixel edge).  Brighter-fatter
strength and charge diffusion are independently adjustable.

### 12.3 Catalog colour information

The `bp_rp` column is automatically added to all generated catalogs:

- **Synthetic**: drawn uniformly from [-0.3, 3.0] with a weak correlation
  between flux and colour (brighter → bluer).
- **Gaia DR3**: computed from `phot_bp_mean_mag - phot_rp_mean_mag`.
- **Test grid**: linearly spaced across [-0.3, 3.0] (faintest source is
  reddest).

The colour is propagated through `sim_cat` in Monte Carlo results and is
available for downstream analysis (e.g., colour-binned bias plots).

### 12.4 Performance

Chromatic photon shooting with sensor is substantially slower than the
monochromatic path:

| Configuration | Time / realisation (64², 1 source) |
|-------------|--------------------------------------|
| Monochromatic Moffat, `method="auto"` | ~0.01 s |
| Monochromatic Moffat, `method="phot"` | ~0.1 s |
| Chromatic, `method="phot"` | ~4 s |
| Chromatic + sensor, `method="phot"` | ~5 s |

For MC runs, plan for ~50× the cost of monochromatic FFT rendering.

## 13. What remains to be implemented

### 13.1 Detailed PSF calibration

The current chromatic PSF uses a generic LSST-like model.  Future work
could include:

- **Zenith-angle-dependent DCR** for each source based on its field position
  (currently a single zenith angle for the whole image).
- **PhaseScreenPSF** for time-evolving atmospheric turbulence.
- **Real galaxy SEDs** from stellar libraries (Pickles, etc.) instead of
  blackbody approximations.
- **PSF fitting from the image** — use the chromatic PSF model as a
  template for PSF photometry.

### 13.2 Extended sensor features

- **Tree rings** — radial doping variations via `galsim.LookupTable`.
- **Brighter-fatter correction** — invert the sensor model for calibration.
- **Wavelength-dependent QE** — currently the bandpass throughput is a
  top-hat; a real LSST QE curve would improve realism.

### 13.3 Correlated noise

GalSim can generate correlated noise from an image or power spectrum via
`CorrelatedNoise`, and provides `getCOSMOSNoise()` for HST F814W.  This
would replace the current simple Gaussian read noise with realistic
sky-subtraction residuals.

### 12.4 Correlated noise

GalSim can generate correlated noise from an image or power spectrum via
`CorrelatedNoise`, and provides `getCOSMOSNoise()` for HST F814W.  This
would replace the current simple Gaussian read noise with realistic
sky-subtraction residuals.

### 12.5 WCS and astrometric distortions

The current simulation uses a pixel scale of 1.0 (no WCS).  GalSim supports
a full WCS hierarchy (`PixelScale`, `AffineTransform`, `FitsWCS`, `TanWCS`,
`FittedSIPWCS`).  Adding WCS would allow:
- Simulating images with realistic pixel scales and distortions.
- Matching the WCS used for Gaia source catalog queries.

### 12.6 Real galaxy morphologies

GalSim's `RealGalaxy` from the HST COSMOS catalog provides realistic
galaxy shapes and profiles.  This could replace the simple Moffat model
for science validation.

---

## 14. GalSim integration study (reference)

The following sections document the GalSim capabilities surveyed during
initial integration planning and remain relevant as a reference for future
work.

### 14.1 PSF profiles

GalSim provides these PSF profile classes (all in `galsim.*`, all subclasses of
`GSObject`):

| Class | Description |
|-------|-------------|
| `Gaussian(sigma/fwhm/hlr)` | 2D Gaussian |
| `Moffat(beta, scale_radius/fwhm/hlr, trunc)` | Moffat profile with optional truncation radius |
| `Kolmogorov(lam_over_r0/fwhm/hlr)` | Long-exposure atmospheric PSF (pure Kolmogorov turbulence) |
| `VonKarman(lam, r0/r0_500, L0)` | von Karman model with outer scale L0 (~10–100 m); has a delta-function component at origin |
| `Airy(lam_over_diam, lam, diam, obscuration)` | Diffraction-limited PSF for circular aperture |
| `OpticalPSF(lam_over_diam, aberrations, ...)` | Aberrated PSF via Zernike polynomials (Noll/annular), obscuration, struts, custom pupil image |
| `PhaseScreenPSF` | PSF from atmospheric + optical phase screens; time-evolving; the most physically realistic model |
| `InterpolatedImage(image, ...)` | Arbitrary PSF from a data image (e.g., observed star stamp) |

### 14.2 Rendering accuracy

GalSim draws profiles via three methods, each with different tail behaviour:

| Method | Behaviour | Tail handling |
|--------|-----------|---------------|
| `'auto'` | FFT for most profiles, `'real_space'` for hard-edged ones | FFT: analytic k-space, folding controlled by `folding_threshold` |
| `'fft'` | Convolve with pixel via DFT; multiply in k-space, FFT back | Folding concern: periodic boundaries cause aliasing. Mitigate with `folding_threshold=1e-6` and `maximum_fft_size` up to 8192+ |
| `'real_space'` | Direct Gauss–Kronrod–Patterson integration over pixel area | Accurate for truncated profiles; slower; limited to 2-component convolutions |
| `'phot'` | Photon shooting: profile sampled as PDF, photons binned | Naturally handles infinite tails; needs sufficient `n_photons`; not available for deconvolutions |

**Accuracy controls** (`GSParams`):

```python
gsp = galsim.GSParams(
    folding_threshold=1e-6,     # default 0.005; lower = less PSF wing aliasing
    stepk_minimum_hlr=5,        # stepk ≤ π / (5 × half-light-radius)
    maxk_threshold=1e-3,        # high-k cutoff for ringing control
    maximum_fft_size=8192,      # raise for larger images
)
```

### 14.3 Chromatic and sensor capabilities

See sections 12 and 13 above for the status of chromatic and sensor
effect implementation.

### 14.4 Additional capabilities

| Feature | Details |
|---------|---------|
| **Correlated noise** | `CorrelatedNoise` from image / power spectrum; `whitenImage()`, `symmetrizeImage()`; pre-built `getCOSMOSNoise()` for HST F814W |
| **Real galaxies** | `RealGalaxy` from HST COSMOS catalog (56k–87k galaxies); `RealGalaxyCatalog`; supports chromatic extension |
| **WCS** | Full hierarchy: `PixelScale`, `ShearWCS`, `AffineTransform`, `FitsWCS`, `TanWCS`, `FittedSIPWCS`, celestial frames |
| **Roman module** | `galsim.roman`: bandpasses, PSF per SCA+position, WCS, sky background, detector effects, scheduling (bestPA, allowedPos) |
| **Zernike utilities** | `galsim.zernike.Zernike` — evaluate, fit, compose Noll/annular polynomials; used by `OpticalPSF` |
| **Lookup tables** | `galsim.LookupTable` — 1-D interpolation for SEDs, bandpasses, tree ring profiles |
| **Config system** | YAML-based simulation descriptions for non-Python users (not relevant to our API) |

---

## 15. Photon-shooting verification

A dedicated verification script (`examples/check_galsim_photon_shooting.py`)
validates that GalSim photon shooting produces unbiased flux realisations
with correct centering.  Key results for a single Moffat source with
γ=3.0, α=3.0, flux=50000 ADU at the centre of a 129×129 image:

- **Centroid**: matches input position to within ~0.01 pixels (Poisson-noise
  limited: σ_centroid ≈ FWHM / (2.355√N) ≈ 0.006 pix).
- **Growth-curve**: sinc-extracted profile matches the analytic Moffat to
  within <0.3% at most radii (FFT reference); residuals at large radii are
  shot-noise dominated.
- **Total flux**: recovered flux is within 1σ of the input
  (√50000/50000 ≈ 0.45%).
- **Pixelization bias**: the sinc extraction method corrects the 2-10%
  bias visible with standard (`photutils`) aperture photometry.
