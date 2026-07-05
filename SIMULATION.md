# Simulation framework

`gcphotom` provides two levels of simulation: **single-image simulation** for
quick experimentation, and **Monte Carlo simulation** for statistical bias
analysis across many random realizations.  This document describes the design,
data flow, and extension points.

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

### 1.2 Monte Carlo simulation (`gcphotom.montecarlo.MonteCarlo`)

Repeats the single-image simulation + photometry pipeline N times with
independent random catalogs:

```python
from gcphotom.montecarlo import SimulationConfig, MonteCarlo

cfg = SimulationConfig(n_sources=1000, shape=(1024, 1024), ...)
mc = MonteCarlo(cfg, n_realizations=100, seed=42)
results = mc.run()          # -> list[dict]
```

Each realisation:
1. Generates a catalog via `catalog_fn(seed)` (or the default `make_realistic_source_catalog`).
2. Simulates an image.
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

---

## 3. Estimator API

An **estimator** is a function `(image, detections, cog) -> dict` that
performs one photometry measurement on a realisation and returns:

```python
{
    "best_fit": {           # pytree of fitted parameters (must include "flux")
        "flux": ndarray,    # per-source, NaN-expanded to det_cat length
        ...                 # any other parameters (gamma, alpha, back, …)
    },
    "uncertainty": {        # same structure as best_fit, 1σ uncertainties
        "flux": ndarray,    # or None if unavailable
        ...
    } | None,
    "extra": {
        "estimation_time": float,   # wall-clock seconds (added by @timed_estimator)
        ...                         # any estimator-specific metadata
    },
}
```

Key rules:
- **`best_fit["flux"]`** must be a 1-D array with the same length as the
  detection catalog (sources dropped internally are NaN).
- **`uncertainty`** is either a pytree matching `best_fit` or `None` (when
  uncertainties are not available).
- **`extra["estimation_time"]`** is mandatory — the `@timed_estimator`
  decorator injects it automatically.

### `@timed_estimator` decorator

```python
from gcphotom.montecarlo import timed_estimator

@timed_estimator
def my_estimator(image, detections, cog):
    # ... compute ...
    return {"best_fit": ..., "uncertainty": ...}
    # extra["estimation_time"] is added automatically
```

The decorator wraps the function, measures wall time with
`time.perf_counter()`, and inserts `estimation_time` into the returned
`extra` dict.  If the function already returned an `extra` dict the
decorated fields are merged.

### `detections` dict

Passed to every estimator, it contains the outputs of
`detect_and_segment`:

```python
{
    "seg": seg,             # segmentation image (ndarray)
    "det_cat": det_cat,     # detection catalog (SourceCatalog, *not* persisted)
    "bkg_map": bkg_map,     # background map (ndarray)
    "bkg_var_map": ...,     # background variance map (ndarray)
}
```

The `det_cat` in the `detections` dict is the full `photutils.segmentation.SourceCatalog` needed by the estimators.
In the pipeline result, `det_cat` is a lightweight `~astropy.table.Table` (see `det_cat_to_table`).

**Note:** estimators that need initial-guess quality flags compute them
internally from `det_cat` (e.g. `(ellipticity * area) > 6`).

---

## 4. Built-in estimators

| Function | Key in `default_estimators()` | Description |
|----------|-------------------------------|-------------|
| `gc_estimator` | `"GC"` | Two-step growth-curve fit with free background.  Returns `flux`, `back`, `gamma`, `alpha`, `ngoods`, `chi2` and their standard errors. |
| `gc_fixed_back_estimator` | `"GC (fixed back)"` | Same but background is fixed to the mean fitted value — isolates the effect of background misestimation on fluxes. |
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
(five columns: `x`, `y`, `area`, `ellipticity`, `kron_flux`) rather than the
full `SourceCatalog`.  Use `det_cat_to_table(det_cat)` to create such a table
manually from a `SourceCatalog`.

---

## 8. Adding a new estimator

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

## 9. Data flow

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
       │   ├── simulate_image()
       │   ├── run_pipeline()
       │   │    ├── detect_and_segment()          → detections dict
       │   │    ├── extract_growth_curves()       → cog
       │   │    ├── cross_match(det_cat, truth)   → sim_cat
       │   │    └── estimator(img, detections, cog)  → result per estimator
       │   │                            (×N estimators)
       │   │
       │   └── append result dict (det_cat → lightweight Table via det_cat_to_table)
       │
       └── return list[dict] ──┐
                               ▼
                    compute_flux_bias()  →  plot_flux_bias()
                    plot_scalar_bias()   →  plot_estimation_times()
                               │
                               ▼
                     save_results() / load_results()  (pickle)
```

---

## 10. GalSim integration study

[GalSim](https://github.com/GalSim-developers/GalSim) (Rowe et al. 2015, A&C,
10, 121) is the standard open-source library for simulating astronomical images.
This section documents its capabilities relevant to growth-curve photometry
simulations, based on GalSim v2.8.4.

### 10.1 PSF profiles

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

**Relevance:** Our current simulations use only Moffat PSF. GalSim makes it
trivial to compare Moffat, Kolmogorov, von Karman, aberrated optical, and
data-driven PSFs within the same framework.

### 10.2 PSF tails at large radius

GalSim draws profiles via three methods, each with different tail behaviour:

| Method | Behaviour | Tail handling |
|--------|-----------|---------------|
| `'auto'` | FFT for most profiles, `'real_space'` for hard-edged ones | FFT: analytic k-space, folding controlled by `folding_threshold` |
| `'fft'` | Convolve with pixel via DFT; multiply in k-space, FFT back | **Folding concern**: periodic boundaries cause aliasing. Mitigate with `folding_threshold=1e-6` and `maximum_fft_size` up to 8192+ |
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

**Implication for growth-curve photometry:** The default FFT method can
alias ~0.5% of PSF flux into the wings. For bias studies at the 0.1% level,
use `folding_threshold=1e-6` or render with `'real_space'` for analytic
profiles.  Profiles with `is_analytic_x = True` (Gaussian, Moffat, Sersic, …)
can also be evaluated at arbitrary (x, y) via `xValue()` — ideal for
computing exact growth curves without FFT artifacts.

### 10.3 Spatially and colour-varying PSF

**Chromatic (wavelength-dependent) PSF** is a first-class concept:

- `ChromaticObject` — base class wrapping a `GSObject`; transformation args
  can be functions of wavelength λ.
- `ChromaticAtmosphere(base_obj, base_wavelength, zenith_angle, …)` —
  differential chromatic refraction (DCR) + λ∝⁻⁰·² seeing scaling.
- `ChromaticOpticalPSF(lam, diam, aberrations, …)` — λ-dependent diffraction
  and Zernike scaling.
- `ChromaticAiry`, `ChromaticRealGalaxy`, `InterpolatedChromaticObject`.
- `ChromaticConvolution`, `ChromaticSum`, `ChromaticTransformation`.

**Bandpass integration:**

```python
bp = galsim.Bandpass('LSST_r.dat', wave_type='nm')
image = chromatic_obj.drawImage(bp, scale=0.2)
```

The integrator caches SED×Bandpass products for speed.  `Spectrum` objects
(built-in SEDs like `SED('CWW_E_ext.sed', wave_type='nm')`) model galaxy/star
spectral energy distributions.

**Field-varying PSF:**

- `PhaseScreenList.makePSF(theta=(x_angle, y_angle))` — atmospheric PSF at
  a specific field angle (the phase screens compute the wavefront at that
  angle).
- `galsim.roman.getPSF(SCA, bandpass, SCA_pos)` — Roman Space Telescope PSF
  with Zernike aberrations interpolated across each SCA from WebbPSF tables.
- No built-in "multi-position PSF manager" — users loop over positions and
  construct per-position PSFs manually.

**Relevance:** Chromatic effects (DCR + λ-dependent seeing) can be significant
for growth-curve photometry in wide-band surveys.  GalSim allows quantifying
this bias.  Field-varying PSF is relevant for large-format detectors (Roman,
LSST).

### 10.4 Sensor effects

**Brighter-fatter effect, charge diffusion, tree rings:**

```python
sensor = galsim.SiliconSensor(
    name='lsst_itl_50_8',    # also 'lsst_e2v_*' models
    strength=1.0,            # brighter-fatter amplitude
    diffusion_factor=1.0,    # charge diffusion (0 to disable)
    treering_func=...,       # LookupTable for radial tree ring profile
    treering_center=...,     # PositionD
)
```

Applied during photon accumulation: `image = obj.drawImage(..., sensor=sensor,
method='phot')`.

**Roman-specific detector effects** (`galsim.roman`):

| Function | Effect |
|----------|--------|
| `applyNonlinearity(img)` | Second-order non-linearity: `counts_out = counts_in + β·counts_in²` |
| `addReciprocityFailure(img, exptime)` | Wavelength-dependent QE drop at low flux |
| `applyIPC(img)` | Inter-pixel capacitance (3×3 convolution kernel) |
| `applyPersistence(img, prev_exposures)` | Residual images from prior exposures |

**Not implemented** (planned): cosmic rays, saturation/bleeding, vignetting,
fringing.

### 10.5 Integration approach for our code

The current `simulate_image` in `src/gcphotom/simulator.py` uses
`photutils.datasets.make_model_image` to render Moffat sources.  Replacing
this with GalSim involves:

```python
import galsim as gs

# 1. Build profiles per source
profiles = []
for src in catalog:
    moffat = gs.Moffat(beta=alpha, scale_radius=gamma / beta, flux=src['flux'])
    moffat = moffat.shift(dx=src['x'], dy=src['y'])
    profiles.append(moffat)

# 2. Sum all profiles
scene = gs.Add(profiles)

# 3. Draw
image = scene.drawImage(nx=shape[1], ny=shape[0], scale=1.0,
                        dtype=np.float32, method='auto')
```

**Key adaptations needed:**

| Current (`photutils`) | GalSim equivalent |
|------------------------|-------------------|
| `make_model_image(coords, fluxes, flux_radius=gamma/alpha,…)` | `galsim.Moffat(beta, scale_radius).shift(dx,dy)` + `galsim.Add` + `drawImage` |
| Moffat only | Any `GSObject` (Kolmogorov, Airy, `OpticalPSF`, `InterpolatedImage`, …) |
| Additive Gaussian + Poisson noise | `galsim.GaussianNoise(rng, sigma)` + Poisson via `method='phot'` |
| WCS via pixel scale scalar | `galsim.PixelScale(scale)` or `galsim.AffineTransform` or `galsim.FitsWCS` |
| No chromaticity | `ChromaticObject` + `Bandpass` integration |

The Monte Carlo loop structure (`catalog_fn` → `simulate_image` → estimators)
remains unchanged — only `simulate_image` (and optionally `make_source_catalog`
to attach SEDs) needs modification.

**Dependency:** GalSim is a large package with a C++ core and requires FFTW.
Installed via `pip install galsim` or `conda install -c conda-forge galsim`.
It does not depend on JAX — the two can coexist, but GalSim will be the
bottleneck for Monte Carlo loops (it is optimised, but single-threaded in
Python).

### 10.6 Additional relevant capabilities

| Feature | Details |
|---------|---------|
| **Correlated noise** | `CorrelatedNoise` from image / power spectrum; `whitenImage()`, `symmetrizeImage()`; pre-built `getCOSMOSNoise()` for HST F814W |
| **Real galaxies** | `RealGalaxy` from HST COSMOS catalog (56k–87k galaxies); `RealGalaxyCatalog`; supports chromatic extension |
| **WCS** | Full hierarchy: `PixelScale`, `ShearWCS`, `AffineTransform`, `FitsWCS`, `TanWCS`, `FittedSIPWCS`, celestial frames |
| **Roman module** | `galsim.roman`: bandpasses, PSF per SCA+position, WCS, sky background, detector effects, scheduling (bestPA, allowedPos) |
| **Zernike utilities** | `galsim.zernike.Zernike` — evaluate, fit, compose Noll/annular polynomials; used by `OpticalPSF` |
| **Lookup tables** | `galsim.LookupTable` — 1-D interpolation for SEDs, bandpasses, tree ring profiles |
| **Config system** | YAML-based simulation descriptions for non-Python users (not relevant to our API) |

### 10.7 Summary

GalSim would strengthen our simulations by adding:

1. **Realistic PSF variety** — Kolmogorov, von Karman, aberrated optical, data-driven
2. **Chromatic effects** — DCR, λ-dependent seeing, bandpass integration
3. **Sensor effects** — brighter-fatter, charge diffusion, tree rings
4. **Correlated noise** — realistic sky subtraction residuals
5. **Real galaxies** — COSMOS-based training/testing

The main cost is API adaptation in `simulate_image` and a heavier dependency
(FFTW, C++ compilation).  The Monte Carlo loop structure, estimator API, and
bias analysis code need no changes.

---

## 11. Photon-shooting Gaia field demo

A demonstrator script (`examples/galsim_gaia_photon_shooting.py`) was written
to test GalSim's photon-shooting and FFT rendering on the three Gaia DR3 fields
used by `examples/mc_gaia_bias.py`.  The script queries Gaia for sources,
renders them with a Moffat PSF (γ=3.0, β=3.0, ZP=25, G<20), adds constant
background (100 ADU) and Gaussian read noise (5 ADU), and reports wall-clock
time.  Chromatic and sensor effects are ignored.

### 11.1 Timing results (512×512 images)

| Field | Sources | Total flux (ADU) | Photon shooting | FFT |
|-------|---------|-----------------|-----------------|-----|
| COSMOS (b≈+42°, sparse) | 67 | 3.6×10⁵ | **0.06 s** | 1.00 s |
| Gemini (b≈+15°, mid) | 406 | 4.3×10⁶ | **3.28 s** | 6.13 s |
| Cygnus (b≈0°, dense) | 2647 | 6.0×10⁶ | **27.20 s** | 39.97 s |

### 11.2 Key findings

**Photon shooting is faster than FFT** for all three fields at 512×512.
The speed ratio ranges from ~1.5× (Cygnus) to ~16× (COSMOS).

**Scaling:** Photon-shooting wall time scales linearly with total flux
(~6 µs per ADU).  FFT time scales with both source count and image size
(the k-space grid is fixed by image dimensions).  For sparse fields with
few bright sources, photon shooting is dramatically faster.

**Noise properties:** The photon-shooting images show pure Poisson noise
(no approximation), while the FFT images have Poisson noise approximated
by adding Gaussian noise to the rendered surface brightness.  The
difference maps (photon shooting − FFT) show structure dominated by
Poisson shot noise in bright sources and the read-noise floor in the
background.

**Limitations of this demo:**
- Moffat PSF only; no comparison with Kolmogorov, Airy, or data-driven PSFs.
- No chromatic effects (DCR, λ-dependent seeing, bandpass integration).
- No sensor effects (brighter-fatter, charge diffusion, tree rings).
- No correlated noise.
- Background is added as a flat constant after drawing, rather than being
  included in the photon shooting (which would correctly add Poisson noise
  to the background as well).
- Read noise is added as Gaussian after drawing, which is correct.
- 512×512 images — timing ratios may change for larger images where the FFT
  overhead is amortised over more pixels.

### 11.3 Implications for growth-curve photometry simulations

1. **Photon shooting is viable for Monte Carlo runs.**  Even the dense
   Cygnus field takes only 27 s per realisation.  A 100-realisation MC
   would take ~45 minutes for Cygnus, well within practical limits.

2. **The FFT path is also viable but slower.**  For COSMOS-like fields,
   FFT is 1 s vs 0.06 s — a 16× penalty that adds up over many realisations.

3. **The trade-off changes with image size.**  FFT cost grows as
   O(N_pix log N_pix), photon-shooting cost as O(total flux).  For larger
   formats (2048×2048, 4096×4096) the FFT path may become more competitive.

4. **Coordinate mapping works correctly** with the formula:
   `shift(dx, dy)` where `dx = x_src - nx/2 + 0.5`, `dy = y_src - ny/2 + 0.5`
   (0-indexed FITS pixel → GalSim world coordinate).

5. **The Monte Carlo loop structure is unaffected.**  Only `simulate_image`
   needs to be replaced; the `catalog_fn` → estimators → bias analysis chain
   requires no changes.
