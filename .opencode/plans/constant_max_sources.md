# Idea: Pad arrays to a constant maximum number of sources

## Goal
Avoid JAX recompilation when the number of sources changes between fits
(particularly after `_cut` / `detect_contamination` or between Monte Carlo
realizations).  Currently, every shape change forces JAX to retrace and
recompile `_adam_step`, `wr_jit` and any function that captures `self.fluxes`,
`self.goods`, etc.

## Approach
- Add a `max_sources` parameter to `Fitter.__init__()`.
- Allocate `self.fluxes`, `self.bkg_var`, `self.goods`, and the parameter
  arrays to a fixed shape `(n_radii, max_sources)`.
- Padding entries are marked with `goods = False` and `kept = False` so that
  `weighted_residuals` zeroes them out (`r * self.goods` → 0).
- The loss function uses `jnp.nansum` / `jnp.where(goods, ...)` to exclude
  padded entries from the mean instead of `jnp.mean`.
- `_cut` no longer truncates arrays; it only updates `self.kept` and
  `self.goods`.
- `detect_contamination` still flips `goods` entries from True to False but
  never removes columns.
- `results` / `rescale_params` / `expand_to_original` already handle missing
  entries via `self.kept`.

## Why we are not doing this (yet)
JAX's compilation cache keys on **the full jaxpr** — including constant values
of captured arrays.  When `self.goods` values change (detect_contamination) or
different Monte Carlo realisations produce different `self.fluxes`, the jaxpr
hash changes and the cache misses anyway.  The only scenario that *would*
benefit is repeated `fit()` calls on *identical* data where `_cut` drops
sources — the shapes stay constant with padding, so the avals for
`_adam_step` remain unchanged.

For large Monte Carlo runs where all realisations generate the same number
of sources, padding would keep the array shapes identical across
realisations.  Combined with passing data arrays through the step function
(instead of capturing them as constants), this could enable cache hits
across realisations.  That refactoring is left for future work.
