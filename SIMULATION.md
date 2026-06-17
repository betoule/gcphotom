# GCPhotom Simulation Pipeline Plan

## Goal

Build a test pipeline that simulates a realistic 1024×1024 CCD image with ~1000 stellar sources, extracts aperture growth curves, and fits Moffat profiles — enabling end-to-end validation of `gcphotom`'s growth-curve fitting.

---

## Architecture

```
src/gcphotom/
  simulator.py    ← image generation + noise
  aperture.py     ← growth curve extraction

tests/
  test_simulator.py   ← image simulation tests
  test_aperture.py    ← growth curve extraction tests
```

### Dependencies

`photutils>=1.10` is a core dependency (already in `pyproject.toml`).

---

## Module 1: `simulator.py`

### `make_source_catalog(n_sources=1000, shape=(1024, 1024), margin=20, seed=None)`

Generate a realistic source catalog:

- **Positions**: Uniform random placement in `[margin, shape[1]-margin] × [margin, shape[0]-margin]`. No minimum separation — positions are truly random.
- **Fluxes**: Log-uniform distribution. Sample `log10(F) ~ Uniform(log10(F_min), log10(F_max))` with `F_min=100`, `F_max=1e6` ADU. This gives a roughly flat distribution in magnitude, realistic for crowded fields.
- **Return**: `astropy.table.Table` with columns `(x, y, flux)`.

### `simulate_image(shape=(1024, 1024), catalog=None, alpha=3, beta=3, background=100, read_noise=5, seed=None)`

Assemble the final image:

1. **Catalog**: If `catalog` is `None`, generate one via `make_source_catalog(shape=shape, seed=seed)`.
2. **PSF rendering**: `photutils.datasets.make_model_image(shape, Moffat2D, catalog)` with `discretize_method='oversample'` (factor=10) for accurate subpixel flux. The Moffat2D model is constructed directly with `amplitude = flux * (beta - 1) / (alpha**2 * pi)` to ensure correct total flux.
3. **Background**: Add constant `background` value to all pixels.
4. **Poisson noise**: `photutils.datasets.apply_poisson_noise(image, seed=seed)` — models photon shot noise.
5. **Read noise**: Add `np.random.normal(0, read_noise, shape)` — Gaussian readout noise.
6. **Return**: `(image, catalog)` tuple. The catalog retains injected truth values.

### Default Parameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `alpha` | 3 pix | Typical seeing-limited Moffat core |
| `beta` | 3 | Realistic wing index |
| `background` | 100 ADU | Typical sky background |
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
| `test_catalog_length` | `make_source_catalog` returns exact number of sources |
| `test_default_n_sources` | Default `n_sources` is 1000 |
| `test_positions_within_bounds` | All positions within `[margin, shape-margin]` |
| `test_flux_range` | Flux values in `[100, 1e6]` |
| `test_flux_log_uniform_mean` | Mean of `log10(flux)` ≈ `(log10(100) + log10(1e6)) / 2` |
| `test_image_shape` | `simulate_image` returns correct shape |
| `test_returns_catalog` | Returned catalog is the same object as input |
| `test_auto_generate_catalog` | `catalog=None` auto-generates a catalog |
| `test_no_nan` | Image contains no NaN or Inf values |
| `test_background_level` | Background region median matches injected value |
| `test_noise_statistics` | Background region std ≈ `sqrt(background + read_noise²)` |
| `test_flux_conservation` | Total image flux ≈ sum of injected fluxes (within Poisson tolerance) |
| `test_full_simulation` | Full pipeline produces correct shape, finite values, no negative pixels |

### `tests/test_aperture.py`

| Test | What it checks |
|------|----------------|
| `test_shape` | `estimate_error` returns correct shape |
| `test_known_values` | `estimate_error` gives expected value for known inputs |
| `test_read_noise_dominant` | Read noise dominates when signal is zero |
| `test_no_negative_signal` | Negative signal clamped to zero |
| `test_output_shapes` | Growth curve output arrays have correct shapes |
| `test_monotonic_increase` | Profile is mostly monotonically increasing (isolated source) |
| `test_flux_recovery` | Recovered flux at large radius matches injected flux (within tolerance) |
| `test_with_error` | Error map propagates to profile error |
| `test_multi_source` | All sources get growth curves extracted with correct shapes |
| `test_with_error_map` | Error estimates work with background-subtracted images |

---

## Implementation Order

1. **`simulator.py`** — `make_source_catalog` → `simulate_image`
2. **`tests/test_simulator.py`** — Validate each component
3. **`aperture.py`** — `extract_growth_curves` + helpers
4. **`tests/test_aperture.py`** — Validate extraction

---

## Notes

- The Moffat2D model in astropy uses `gamma` (scale parameter) and `alpha` (shape parameter). Our `alpha` maps to `gamma`, our `beta` maps to `alpha`.
- `photutils.datasets.make_model_image` handles overlapping sources correctly (fluxes add).
- No minimum separation constraint on source positions — locations are truly random, allowing natural overlap scenarios for realistic testing.
- The log-uniform flux distribution produces sources spanning ~4 magnitudes, giving a realistic dynamic range for testing both faint and bright source recovery.
