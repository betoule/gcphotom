# Growth Curve Fitter Refactoring Plan

## Overview

Refactor `gcmodel.py` so the `Fitter` class works with the output of `aperture.py`'s `extract_growth_curves()` instead of a proprietary numpy structured array catalog. Test on simulations with 1000 sources.

---

## Decisions (from user)

- **Drop `fit_tncg`** — keep only Adam-based `fit()`
- **No contamination flags initially** — rely on `detect_contamination()` for residual-based rejection
- **Test with 1000 sources** (simulator default, 1024x1024 images)

---

## Task 1 — Rewrite `gcmodel.py`

### Remove
- `from stardiceonline.processing.robuststat import mad, robust_average` (unused, external dep)
- `fit_tncg` method (depends on missing `jaxfitter.tncg`)
- `join`, `radii(cat)`, `cat_to_flux` functions (structured array helpers)
- `__main__` block tied to real data files
- `write_cat` method (numpy structured array writer)

### Keep unchanged
- `fwhm2alpha`, `alpha2fwhm`, `sigma2fwhm`, `fwhm2sigma`
- `moffat`, `moffat_flux`, `imoffat`
- `flux_and_couronnes`

### Modify

**`model(params, radii)`** — add `radii` parameter, eliminate global `rad` dependency:
```python
def model(params, radii):
    flux = params["flux"]
    alpha = params["alpha"]
    beta = params["beta"]
    return (
        flux[None, :] * moffat_flux(radii[:, None], alpha, beta)
        + params["back"][None, :] * radii[:, None] ** 2 * jnp.pi
    )
```

### New `Fitter` class

```python
class Fitter:
    def __init__(self, gc_result, model=model):
        # gc_result: dict from extract_growth_curves()
        #   {"radius": [R], "flux": [N, R], "flux_err": [N, R]}
        self.radii = jnp.array(gc_result["radius"])
        self.areas = flux_and_couronnes(self.radii**2 * jnp.pi)
        self._set_data(gc_result)
        self._cut()
        self.model = model

    def _set_data(self, gc_result):
        self.fluxes = flux_and_couronnes(jnp.array(gc_result["flux"]))
        var_cum = jnp.array(gc_result["flux_err"]) ** 2
        self.var = flux_and_couronnes(var_cum)
        self.var = jnp.clip(self.var, 1e-30, None)
        self.goods = jnp.isfinite(self.fluxes) & jnp.isfinite(self.var)

    def _cut(self):
        # Remove sources with fewer than 2 good data points
        valid = self.goods.sum(axis=0) > 1
        self.fluxes = self.fluxes[:, valid]
        self.var = self.var[:, valid]
        self.goods = self.goods[:, valid]

    def residuals(self, params, mask=False):
        m = self.model(self._flux(params), self.radii)
        r = self.fluxes - flux_and_couronnes(m)
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        return r

    def weighted_residuals(self, params, mask=False):
        m = flux_and_couronnes(self.model(self._flux(params), self.radii))
        residuals = self.fluxes - m
        noise = m * 0.01
        r = residuals / jnp.sqrt(self.var + noise**2) * self.goods
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        return r

    def chi2(self, params):
        return (self.weighted_residuals(params) ** 2).mean()

    def fit(self, initial_guess=None, niter=10000, learning_rate=5e-3, show=False):
        if initial_guess is None:
            initial_guess = self.initial_guess()
        chi2_fn = jax.jit(self.chi2)
        bf, extra = jaxfitter.fit_adam(
            chi2_fn, initial_guess, niter=niter,
            learning_rate=learning_rate, tol=None
        )
        if show:
            plt.plot(extra["loss"])
        return bf, extra
```

**`initial_guess`** — estimate alpha from growth curve shape instead of `gwmxx`/`gwmyy`:
- For each source, find radius where cumulative flux reaches 50% of estimated total
- This radius ≈ FWHM/2, convert to alpha via `fwhm2alpha(sigma2fwhm(r_50), beta)`

**`results(bf)`** — replaces `write_cat`, returns dict:
```python
def results(self, bf):
    par = self._flux(bf)
    ngoods = self.goods.sum(axis=0)
    wr = self.weighted_residuals(bf, mask=True)
    chi2 = np.nansum(wr**2, axis=0)
    return {
        "flux": np.array(par["flux"]),
        "back": np.array(par["back"]),
        "alpha": bf["alpha"],
        "beta": bf["beta"],
        "ngoods": np.array(ngoods),
        "chi2": np.array(chi2),
    }
```

**`plot_PSF`** — same logic, use `self.radii` and `self.areas` instead of globals.

**`detect_contamination`** — unchanged, uses residual-based rejection.

---

## Task 2 — Update `__init__.py`

```python
from gcphotom.aperture import estimate_error, extract_growth_curves
from gcphotom.gcmodel import Fitter
from gcphotom.simulator import make_source_catalog, simulate_image

__all__ = [
    "estimate_error",
    "extract_growth_curves",
    "Fitter",
    "make_source_catalog",
    "simulate_image",
]
```

---

## Task 3 — Tests (`tests/test_gcmodel.py`)

```python
class TestFitter:
    def test_flux_recovery(self):
        # Simulate image with known alpha=2.5, beta=3.0
        # Extract growth curves, fit, assert flux within 10% of injected

    def test_profile_parameter_recovery(self):
        # Assert fitted alpha within 20% of true alpha
        # Assert fitted beta within reasonable range

    def test_background_recovery(self):
        # Simulate with background=100, assert fitted back close to 100

    def test_full_pipeline_1000_sources(self):
        # End-to-end: simulate_image() -> extract_growth_curves() -> Fitter -> results()
        # Check median flux residual across all sources
```

Key test strategy: use the simulator with known Moffat parameters, then verify the fitter recovers them. Run on 1000 sources to validate vectorized performance.

---

## Task 4 — Update `README.md`

Replace the placeholder:
```python
# 3. Fit all growth curves with a common Moffat profile
fitter = gcp.Fitter(result)
best_params, extra = fitter.fit()
fitted = fitter.results(best_params)
```

---

## Task 5 — Update `pyproject.toml`

Remove `gcmodel` from coverage omit since it will now be tested:
```toml
[tool.coverage.run]
source = ["gcphotom"]
# omit removed
```

---

## Risks and Unknowns

1. **Alpha estimation from growth curve**: The 50%-flux method is approximate. May need tuning.
2. **Variance propagation**: `flux_and_couronnes(var_cum)` assumes independent cumulative errors. The actual error on annular flux is `var(r_i) + var(r_{i-1})` which is what `diff` gives — this should be correct.
3. **Convergence**: 10000 Adam iterations may be insufficient for 1000 sources. May need to adjust learning rate or iteration count.
4. **Shared alpha/beta**: The current model fits a single alpha/beta for all sources. This is correct for a common PSF assumption but may need per-source alpha for seeing variations.
