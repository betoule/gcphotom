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
1. Draws a random catalog (sub-seeded from the master seed).
2. Simulates an image.
3. Runs `run_pipeline` with the configured estimators.
4. Collects the result.

Failed realizations are caught, a warning is issued, and the loop continues.
The number of successful realizations is `len(results)`.

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
    "det_cat": det_cat,     # detection catalog (Table)
    "bkg_map": bkg_map,     # background map (ndarray)
    "bkg_var_map": ...,     # background variance map (ndarray)
}
```

**Note:** estimators that need initial-guess quality flags compute them
internally from `det_cat` (e.g. `(ellipticity * area) > 6`).

---

## 4. Built-in estimators

| Function | Key in `default_estimators()` | Description |
|----------|-------------------------------|-------------|
| `gc_estimator` | `"GC"` | Two-step growth-curve fit with free background.  Returns `flux`, `back`, `gamma`, `alpha`, `ngoods`, `chi2` and their standard errors. |
| `gc_fixed_back_estimator` | `"GC (fixed back)"` | Same but background is fixed to the mean fitted value — isolates the effect of background misestimation on fluxes. |
| `aperture_estimator` | `"Aperture + AC"` | Aperture photometry with aperture correction derived from the fitted Moffat profile.  `best_fit` contains only `flux`. |
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
#     "PSF":             partial(psf_estimator, background=cfg.background),
#     "Aperture + AC":   partial(aperture_estimator, fit_kwargs=cfg.fit_kwargs),
# }
```

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
       │   ├── make_realistic_source_catalog()
       │   ├── simulate_image()
       │   ├── run_pipeline()
       │   │    ├── detect_and_segment()          → detections dict
       │   │    ├── extract_growth_curves()       → cog
       │   │    ├── cross_match(det_cat, truth)   → sim_cat
       │   │    └── estimator(img, detections, cog)  → result per estimator
       │   │                            (×N estimators)
       │   │
       │   └── append result dict
       │
       └── return list[dict] ──┐
                               ▼
                    compute_flux_bias()  →  plot_flux_bias()
                               │
                               ▼
                    save_results() / load_results()  (pickle)
```
