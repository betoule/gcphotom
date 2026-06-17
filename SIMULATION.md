# GCPhotom Simulation Pipeline Plan

## Goal

Build a test pipeline that simulates a realistic 1024×1024 CCD image with ~1000 stellar sources, extracts aperture growth curves, and fits Moffat profiles — enabling end-to-end validation of `gcphotom`'s growth-curve fitting.

---

## Architecture

```
src/gcphotom/
  simulator.py    ← image generation + noise
  aperture.py     ← growth curve extraction
  gcmodel.py      ← (existing) Moffat model, fwhm conversions
  stats.py        ← (existing) robust statistics
  plots.py        ← (existing) binplot

tests/
  test_simulator.py   ← image simulation tests
  test_aperture.py    ← growth curve extraction tests
  test_fit_pipeline.py ← end-to-end: simulate → extract → fit → validate
```

### Dependencies

Add `photutils>=1.10` to `pyproject.toml` core dependencies.

---

## Module 1: `simulator.py`

### `make_source_catalog(n_sources=1000, shape=(1024, 1024), seed=None)`

Generate a realistic source catalog:

- **Positions**: Uniform random placement in `[margin, shape[1]-margin] × [margin, shape[0]-margin]`, with `margin=20` pixels to avoid edge effects. Reject overlaps within a minimum separation (e.g., 5 pixels) — retry up to 10 times per source, drop if still overlapping.
- **Fluxes**: Power-law distribution `N(>F) ∝ F^(-alpha)` with `alpha=0.8` (typical for deep imaging surveys). Draw from:
  ```
  F = F_min * (1 - u)^(-1/(alpha-1))   # u ~ Uniform(0,1)
  ```
  Or equivalently, sample logarithmically: `log10(F) ~ Uniform(log10(F_min), log10(F_max))` with `F_min=100`, `F_max=1e6` ADU. The log-uniform distribution gives a roughly flat distribution in magnitude, which is realistic for crowded fields.
- **Return**: `astropy.table.Table` with columns `(x, y, flux)`.

### `make_moffat_psf(alpha, beta, shape=(21, 21))`

Build a normalized Moffat2D model:

- Uses `astropy.modeling.models.Moffat2D(amplitude=1, gamma=alpha, beta=beta, x_0=0, y_0=0)`
- Evaluate on a grid centered at (0, 0) with the given shape
- Normalize so the integral over the grid equals 1.0
- Return the `Moffat2D` model instance

### `simulate_image(shape, catalog, alpha, beta, background=0.0, read_noise=0.0, seed=None)`

Assemble the final image:

1. **PSF rendering**: `photutils.datasets.make_model_image(shape, Moffat2D, catalog)` with `discretize_method='oversample'` (factor=10) for accurate subpixel flux. Table columns map: `amplitude→flux`, `gamma→alpha`, `beta→beta`, `x_0→x`, `y_0→y`.
2. **Background**: Add constant `background` value to all pixels.
3. **Poisson noise**: `photutils.datasets.apply_poisson_noise(image, seed=seed)` — models photon shot noise.
4. **Read noise**: Add `np.random.normal(0, read_noise, shape)` — Gaussian readout noise.
5. **Return**: `(image, catalog)` tuple. The catalog retains injected truth values.

### Default Parameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `alpha` | 2.5 pix | Typical seeing-limited Moffat core |
| `beta` | 3.0 | Realistic wing index |
| `background` | 500 ADU | Typical sky background |
| `read_noise` | 5 ADU | Typical CCD read noise |
| `F_min` | 100 ADU | Faintest detectable source |
| `F_max` | 1e6 ADU | Brightest (near-saturation) |

---

## Module 2: `aperture.py`

### `extract_growth_curves(image, positions, radii, error=None)`

Extract circular growth curves for each source position:

1. For each `(x, y)` in `positions`:
   - `photutils.profiles.CurveOfGrowth(image, (x, y), radii, error=error)`
   - Collect `profile` (cumulative flux), `profile_error`, `radius`
2. **Return**: A dict with:
   ```python
   {
       "radius": ndarray,           # shape (n_radii,)
       "flux": ndarray,             # shape (n_sources, n_radii)
       "flux_err": ndarray,         # shape (n_sources, n_radii)
   }
   ```

### `extract_single_growth_curve(image, position, radii, error=None)`

Convenience wrapper for single-source extraction. Returns `(radius, profile, profile_error)`.

### `estimate_error(image, background, read_noise)`

Compute per-pixel error estimate:
```
error = sqrt(max(image - background, 0) + read_noise**2)
```
The `max(..., 0)` avoids negative values under the background.

---

## Module 3: Tests

### `tests/test_simulator.py`

| Test | What it checks |
|------|----------------|
| `test_catalog_shape` | `make_source_catalog` returns ~1000 sources within bounds |
| `test_flux_distribution` | Flux histogram follows expected log-uniform shape (KS test against uniform in log space) |
| `test_no_position_overlap` | No two sources closer than min separation |
| `test_psf_integral` | `make_moffat_psf` integral ≈ 1.0 |
| `test_image_flux_conservation` | Total image flux ≈ sum of injected fluxes (within Poisson noise tolerance) |
| `test_noise_statistics` | Background region has correct mean/std |
| `test_full_simulation` | `simulate_image` returns correct shape, finite values, no NaN |

### `tests/test_aperture.py`

| Test | What it checks |
|------|----------------|
| `test_single_source_flux` | Recovered flux at large radius matches injected flux (within 5%) |
| `test_growth_curve_shape` | Profile is monotonically increasing (for isolated source) |
| `test_error_propagation` | `profile_error` scales with sqrt(flux) as expected |
| `test_multi_source_extraction` | All sources in catalog get growth curves extracted |
| `test_error_estimate` | `estimate_error` gives correct values for known background + read noise |

### `tests/test_fit_pipeline.py` — End-to-end

| Test | What it checks |
|------|----------------|
| `test_fit_recovered_alpha` | Fitted `alpha` within 10% of injected value (high-S/N sources) |
| `test_fit_recovered_beta` | Fitted `beta` within 20% of injected value |
| `test_fit_recovered_flux` | Fitted flux within 5% of injected value (high-S/N sources) |
| `test_fit_convergence` | Fitter converges (loss decreases) for clean simulated data |
| `test_snr_dependency` | Recovery accuracy degrades gracefully with decreasing S/N |

---

## Implementation Order

1. **`simulator.py`** — `make_source_catalog` → `make_moffat_psf` → `simulate_image`
2. **`tests/test_simulator.py`** — Validate each component
3. **`aperture.py`** — `extract_growth_curves` + helpers
4. **`tests/test_aperture.py`** — Validate extraction
5. **`tests/test_fit_pipeline.py`** — End-to-end integration (will be fleshed out when `gcmodel.py` is refactored)

## Notes

- The Moffat2D model in astropy uses `gamma` (scale parameter) and `beta` (shape parameter), matching our `alpha` and `beta` notation. The column mapping in `make_model_image` will be `gamma=alpha_value`.
- `photutils.datasets.make_model_image` handles overlapping sources correctly (fluxes add).
- For the end-to-end fit test, we will initially use a simplified fitting function that doesn't depend on the `Fitter` class. Once `gcmodel.py` is refactored, the test will use the actual `Fitter`.
- The power-law flux distribution should produce sources spanning ~5 magnitudes, giving a realistic dynamic range for testing both faint and bright source recovery.
