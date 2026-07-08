"""Monte Carlo framework for photometry bias analysis.

Each realization runs a pipeline of independent estimators on a simulated
image.  N realizations is a list of pipeline result dicts, saved/loaded
with pickle.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from dataclasses import dataclass, field
from functools import partial, wraps

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import gcphotom as gcp


@dataclass
class SimulationConfig:
    """Simulation parameters for one Monte Carlo realization.

    Parameters
    ----------
    n_sources : int
        Number of sources per realization.
    shape : tuple of int
        Image shape (ny, nx).
    gamma : float
        Moffat PSF scale parameter in pixels.
    alpha : float
        Moffat PSF shape parameter.
    background : float
        Constant background level in ADU.
    read_noise : float
        Gaussian read noise standard deviation in ADU.
    n_pixels : int
        Number of pixels for background mesh in ``detect_and_segment``.
    fit_kwargs : dict
        Keyword arguments passed to ``Fitter.fit()``.
    """

    n_sources: int = 1000
    shape: tuple = (1024, 1024)
    gamma: float = 3.0
    alpha: float = 3.0
    background: float = 100.0
    read_noise: float = 5.0
    n_pixels: int = 5
    fit_kwargs: dict = field(
        default_factory=lambda: {"learning_rate": 1e-2, "niter": 2000}
    )


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------


def timed_estimator(f):
    """Decorator that adds *estimation_time* (wall-clock seconds) to
    the ``extra`` dict returned by an estimator function."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = f(*args, **kwargs)
        extra = result.setdefault("extra", {})
        extra["estimation_time"] = time.perf_counter() - t0
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Built-in estimator functions
# ---------------------------------------------------------------------------


def _bads(det_cat):
    """Return a boolean mask of sources with poor initial guesses."""
    return (det_cat.ellipticity * det_cat.area).value > 6


@timed_estimator
def gc_estimator(image, detections, cog, fit_kwargs=None):
    """Two-step growth-curve fit with estimated background."""
    fit_kwargs = fit_kwargs or {}
    fitter = gcp.Fitter(cog, bads=_bads(detections["det_cat"]))
    bf, _ = fitter.fit(**fit_kwargs, desc="GC (1)")
    fitter.detect_contamination(bf)
    bf, _ = fitter.fit(**fit_kwargs, compute_uncertainty=True, desc="GC (2)")
    fitted = fitter.results(bf)
    return {
        "best_fit": fitted,
        "uncertainty": fitted.get("std_errors"),
    }


@timed_estimator
def gc_fixed_back_estimator(image, detections, cog, fit_kwargs=None):
    """Two-step growth-curve fit with background fixed to the detection
    background map."""
    fit_kwargs = fit_kwargs or {}

    # Extract background from the detection background map at each
    # source position.
    bkg_map = detections["bkg_map"]
    det_cat = detections["det_cat"]
    x = np.clip(
        np.round(np.asarray(det_cat.x_centroid)).astype(int), 0, bkg_map.shape[1] - 1
    )
    y = np.clip(
        np.round(np.asarray(det_cat.y_centroid)).astype(int), 0, bkg_map.shape[0] - 1
    )
    bkg_local = np.asarray(bkg_map[y, x], dtype=float)

    fitter = gcp.Fitter(cog, bads=_bads(det_cat))
    back_fixed = bkg_local[fitter.kept]

    bf, _ = fitter.fit(**fit_kwargs, fix={"back": back_fixed}, desc="GC fixed back (1)")
    bf["back"] = back_fixed
    fitter.detect_contamination(bf)

    back_fixed = bkg_local[fitter.kept]
    bf, _ = fitter.fit(
        **fit_kwargs,
        fix={"back": back_fixed},
        compute_uncertainty=True,
        desc="GC fixed back (2)",
    )
    bf["back"] = back_fixed
    fitted = fitter.results(bf)
    return {
        "best_fit": fitted,
        "uncertainty": fitted.get("std_errors"),
    }


@timed_estimator
def aperture_estimator(image, detections, cog):
    """Aperture photometry with aperture correction from bright isolated stars.

    The aperture correction converts small-aperture flux to total flux
    using a data-driven PSF model: the PSF shape parameters (gamma,
    alpha) are inferred from the ratio of fluxes at two intermediate
    radii measured on bright, isolated sources, and the correction to
    total flux is computed from those parameters.
    This avoids both the large confusion bias at wide apertures and
    the need for a full growth-curve fit.
    """
    radii = np.asarray(cog["radius"])
    n_radii = len(radii)

    # Aperture indices: small ~ FWHM, large at a moderate radius where the
    # PSF captures nearly all flux but neighbour confusion is still low.
    r_small_idx = min(2, n_radii - 2)
    r_large_idx = min(n_radii - 2, max(4, n_radii - 5))
    r_small = radii[r_small_idx]
    r_large = radii[r_large_idx]

    # Global background map
    bkg_map = detections["bkg_map"]
    det_cat = detections["det_cat"]
    x = np.clip(
        np.round(np.asarray(det_cat.x_centroid)).astype(int), 0, bkg_map.shape[1] - 1
    )
    y = np.clip(
        np.round(np.asarray(det_cat.y_centroid)).astype(int), 0, bkg_map.shape[0] - 1
    )
    bkg_local = np.asarray(bkg_map[y, x], dtype=float)

    flux_small = cog["flux_clean"][:, r_small_idx] - bkg_local * np.pi * r_small**2
    flux_large = cog["flux_clean"][:, r_large_idx] - bkg_local * np.pi * r_large**2

    valid = (
        np.isfinite(flux_small)
        & np.isfinite(flux_large)
        & (flux_small > 0)
        & (flux_large > 0)
    )

    if valid.sum() < 5:
        return {
            "best_fit": {"flux": np.full(len(det_cat), np.nan)},
            "uncertainty": None,
        }

    bads = _bads(detections["det_cat"])

    # Isolation based on contamination at the large aperture radius
    contam_frac = np.where(
        cog["flux"][:, r_large_idx] > 0,
        cog["contamination"][:, r_large_idx] / cog["flux"][:, r_large_idx],
        0.0,
    )
    ct = np.percentile(contam_frac[valid], 30)

    flux_threshold = np.median(flux_small[valid])
    bright = flux_small > flux_threshold

    good = valid & bright & ~bads & (contam_frac <= ct)

    if good.sum() < 5:
        good = valid & ~bads & (contam_frac <= ct)
    if good.sum() < 3:
        good = valid & ~bads

    # --- Aperture correction ---
    # The measured large/small ratio equals moffat_flux(r_large) /
    # moffat_flux(r_small).  Dividing by the former converts the ratio
    # to the total flux correction 1 / moffat_flux(r_small).
    # We assume a Moffat profile with alpha=3 and estimate gamma from
    # the ratio at two different radii.
    r_mid_idx = min(n_radii - 2, max(3, r_large_idx - 2))
    flux_mid = (
        cog["flux_clean"][:, r_mid_idx] - bkg_local * np.pi * radii[r_mid_idx] ** 2
    )

    valid_mid = valid & np.isfinite(flux_mid) & (flux_mid > 0)
    good_mid = good & valid_mid

    if good_mid.sum() >= 5:
        large_ratio = np.median(flux_large[good_mid] / flux_small[good_mid])
        mid_ratio = np.median(flux_mid[good_mid] / flux_small[good_mid])
        shape_ratio = large_ratio / mid_ratio

        alpha_assumed = 3.0
        gamma_grid = np.linspace(1.0, 10.0, 500)
        expected_shape = gcp.gcmodel.moffat_flux(r_large, gamma_grid, alpha_assumed) / (
            gcp.gcmodel.moffat_flux(radii[r_mid_idx], gamma_grid, alpha_assumed) + 1e-30
        )
        gamma_best = float(gamma_grid[np.argmin(np.abs(expected_shape - shape_ratio))])
        f_large = float(gcp.gcmodel.moffat_flux(r_large, gamma_best, alpha_assumed))
        ac = large_ratio / f_large
    else:
        # Fallback: assume gamma=3 (typical ground-based seeing)
        large_ratio = np.median(flux_large[good] / flux_small[good])
        f_large = float(gcp.gcmodel.moffat_flux(r_large, 3.0, 3.0))
        ac = large_ratio / f_large

    if not 1.0 < ac < 5.0:
        ac = large_ratio

    flux_ap = flux_small * ac

    return {
        "best_fit": {"flux": flux_ap},
        "uncertainty": None,
    }


@timed_estimator
def psf_estimator(image, detections, cog, nstars=30, fit_shape=25):
    """PSF photometry estimator."""
    psf_results, _ = gcp.psf_photometry(
        image - detections["bkg_map"],
        detections["det_cat"],
        nstars=nstars,
        fit_shape=fit_shape,
    )
    return {
        "best_fit": {"flux": np.asarray(psf_results["flux_fit"])},
        "uncertainty": None,
    }


def det_cat_to_table(det_cat):
    """Extract a lightweight `~astropy.table.Table` from a `SourceCatalog`.

    The full `SourceCatalog` is expensive to serialise (it holds
    references to the data, segmentation image, cached properties,
    etc.).  This function copies only the columns typically needed
    for later analysis into a plain `~astropy.table.Table`.

    Parameters
    ----------
    det_cat : `~photutils.segmentation.SourceCatalog`
        Detection catalog from :func:`~gcphotom.aperture.detect_and_segment`.

    Returns
    -------
    `~astropy.table.Table`
        Table with columns ``x``, ``y``, ``area``, ``ellipticity``,
        ``kron_flux``.
    """
    from astropy.table import Table

    return Table(
        {
            "x": np.asarray(det_cat.x_centroid, dtype=float),
            "y": np.asarray(det_cat.y_centroid, dtype=float),
            "area": np.asarray(det_cat.area, dtype=float),
            "ellipticity": np.asarray(det_cat.ellipticity, dtype=float),
            "kron_flux": np.asarray(det_cat.kron_flux, dtype=float),
        }
    )


def default_estimators(cfg):
    """Build a dict of default estimators configured from *cfg*.

    Returns
    -------
    dict of str -> callable
        Keys are ``"GC"``, ``"GC (fixed back)"``, ``"PSF"``,
        ``"Aperture + AC"``.
    """
    return {
        "GC": partial(gc_estimator, fit_kwargs=cfg.fit_kwargs),
        "GC (fixed back)": partial(gc_fixed_back_estimator, fit_kwargs=cfg.fit_kwargs),
        "PSF": psf_estimator,
        "Aperture + AC": aperture_estimator,
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(image, sim_cat, cfg, estimators):
    """Run a set of estimators on a simulated image.

    Parameters
    ----------
    image : ndarray
        Simulated image.
    sim_cat : Table
        Truth catalog from the simulation.
    cfg : SimulationConfig
        Simulation configuration.
    estimators : dict of str -> callable
        Mapping from estimator name to estimator function.  Each function
        receives ``(image, detections, cog)`` and returns
        ``{"best_fit": ..., "uncertainty": ..., "extra": ...}``.

    Returns
    -------
    dict
        Keys: ``"sim_cat"``, ``"det_cat"``, ``"params"``, plus one key per
        estimator.
    """
    seg, det_cat, bkg_map, bkg_var_map = gcp.detect_and_segment(
        image, n_pixels=cfg.n_pixels
    )
    detections = {
        "seg": seg,
        "det_cat": det_cat,
        "bkg_map": bkg_map,
        "bkg_var_map": bkg_var_map,
    }

    cog = gcp.extract_growth_curves(
        image,
        det_cat,
        segmentation_image=seg,
        background_variance=bkg_var_map,
        desc="COG",
    )

    input_cat = gcp.cross_match(det_cat, sim_cat)

    result = {
        "sim_cat": input_cat,
        "det_cat": det_cat_to_table(det_cat),
        "params": cfg,
    }
    for name, estimator in estimators.items():
        result[name] = estimator(image, detections, cog)

    return result


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------


class MonteCarlo:
    """Run a set of estimators over multiple independent realizations.

    Parameters
    ----------
    config : SimulationConfig
        Simulation parameters.
    n_realizations : int
        Number of independent realizations.
    seed : int or None
        Master random seed.  Each realization draws a sub-seed.
    estimators : dict of str -> callable, optional
        Estimator functions.  Defaults to :func:`default_estimators`.
    catalog_fn : callable, optional
        Function ``f(seed) -> Table`` that generates the source catalog for
        each realization.  Defaults to ``make_realistic_source_catalog``
        configured with *config.n_sources* and *config.shape*.
    simulate_fn : callable, optional
        Function ``f(shape, catalog, gamma, alpha, background, read_noise,
        seed) -> (image, catalog)`` for image simulation.  Defaults to
        :func:`gcphotom.simulate_image`.
    """

    def __init__(
        self,
        config: SimulationConfig,
        n_realizations: int,
        seed: int | None = None,
        estimators: dict | None = None,
        catalog_fn=None,
        simulate_fn=None,
    ):
        self.config = config
        self.n_realizations = n_realizations
        self.seed = seed
        self.estimators = estimators or default_estimators(config)
        if catalog_fn is None:
            catalog_fn = partial(
                gcp.make_realistic_source_catalog,
                n_sources=self.config.n_sources,
                shape=self.config.shape,
            )
        self._catalog_fn = catalog_fn
        self._simulate_fn = simulate_fn or gcp.simulate_image
        self._results: list[dict] = []

    @property
    def results(self) -> list[dict]:
        return self._results

    def run(self, verbose: bool = True, show_progress: bool = True) -> list[dict]:
        """Execute all realizations.

        Parameters
        ----------
        verbose : bool
            Print progress every 10 realizations.
        show_progress : bool
            If ``True``, display a progress bar.

        Returns
        -------
        list of dict
            One dict per successful realization.
        """
        self._results = []
        rng = np.random.default_rng(self.seed)
        seeds = rng.integers(0, 2**31, self.n_realizations)

        pbar = tqdm(
            total=self.n_realizations,
            desc="Monte Carlo",
            disable=not show_progress,
            unit="real",
        )
        for i, sd in enumerate(seeds):
            if (
                not show_progress
                and verbose
                and i % max(1, self.n_realizations // 10) == 0
            ):
                print(f"  Realization {i + 1}/{self.n_realizations}")

            catalog = self._catalog_fn(seed=int(sd))
            image, _ = self._simulate_fn(
                shape=self.config.shape,
                catalog=catalog,
                gamma=self.config.gamma,
                alpha=self.config.alpha,
                background=self.config.background,
                read_noise=self.config.read_noise,
                seed=int(sd),
            )

            try:
                res = run_pipeline(image, catalog, self.config, self.estimators)
                self._results.append(res)
            except Exception as exc:
                warnings.warn(f"Realization {i + 1} failed: {exc}")
            finally:
                pbar.update(1)

        pbar.close()
        return self._results


# ---------------------------------------------------------------------------
# Flux bias computation and plotting
# ---------------------------------------------------------------------------


def compute_flux_bias(results, estimators=None, nbins=10):
    """Compute binned flux bias for one or more estimators.

    Parameters
    ----------
    results : list of dict
        Per-realization outputs from :class:`MonteCarlo`.
    estimators : list of str, optional
        Estimator names.  Defaults to all estimator keys in *results*.
    nbins : int
        Number of log-spaced bins.

    Returns
    -------
    dict
        Mapping from estimator name to ``{"xbins", "bias", "bias_err"}``.
    """
    if estimators is None:
        estimators = [
            k for k in results[0] if k not in ("sim_cat", "det_cat", "params")
        ]

    out = {}
    for name in estimators:
        xs, ests, truths = [], [], []
        for res in results:
            flux_truth = np.asarray(res["sim_cat"]["flux"])
            flux_est = np.asarray(res[name]["best_fit"]["flux"])
            mask = np.isfinite(flux_truth) & np.isfinite(flux_est) & (flux_truth > 0)
            xs.append(flux_truth[mask])
            ests.append(flux_est[mask])
            truths.append(flux_truth[mask])

        x_all = np.concatenate(xs)
        est_all = np.concatenate(ests)
        truth_all = np.concatenate(truths)
        bias = (est_all / truth_all - 1.0) * 100.0

        from gcphotom.stats import _build_bins, bin_statistic

        valid = np.isfinite(x_all) & np.isfinite(bias)
        edges = _build_bins(x_all[valid], nbins, logbins=True)
        xbinned, bias_binned, bias_err = bin_statistic(
            x_all,
            bias,
            nbins=nbins,
            bins=edges,
            logbins=True,
            method="median",
            scale_err=True,
        )
        out[name] = {"xbins": xbinned, "bias": bias_binned, "bias_err": bias_err}

    return out


def plot_flux_bias(bias_stats, figsize=(7, 5)):
    """Plot flux bias vs simulated flux for one or more estimators.

    Parameters
    ----------
    bias_stats : dict
        Output from :func:`compute_flux_bias`.
    figsize : tuple
        Figure size.

    Returns
    -------
    Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "GC": "k",
        "GC (fixed back)": "r",
        "PSF": "b",
        "Aperture + AC": "m",
    }

    for label, s in bias_stats.items():
        color = colors.get(label, "k")
        ax.errorbar(
            s["xbins"],
            s["bias"],
            yerr=s["bias_err"],
            marker="o",
            ls="none",
            color=color,
            label=label,
            zorder=5,
        )

    ax.axhline(0, color="k", ls="--", alpha=0.3, zorder=0)
    ax.set_xlabel("Simulated flux [ADU]")
    ax.set_xscale("log")
    ax.set_ylabel("Flux bias (%)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return ax


def _estimators_with_scalar(results, param):
    """Return names of estimators in *results* that have scalar *param*."""
    names = set()
    for r in results:
        for k, v in r.items():
            if k in ("sim_cat", "det_cat", "params"):
                continue
            bf = v.get("best_fit", {})
            p = bf.get(param)
            if p is not None and not hasattr(p, "__len__"):
                names.add(k)
    return sorted(names)


def plot_scalar_bias(results, params=("gamma", "alpha"), figsize=None):
    """Per-realisation estimates of scalar parameters for each estimator.

    One panel per parameter.  Each panel shows per-realisation fitted
    values as dots, with a horizontal line at the true value from the
    simulation config.

    Parameters
    ----------
    results : list of dict
        Per-realisation outputs from :class:`MonteCarlo`.
    params : tuple of str
        Scalar parameter names to plot.  Defaults to ``("gamma",
        "alpha")``.
    figsize : tuple or None
        Figure size.  Automatically sized if ``None``.

    Returns
    -------
    Axes
    """
    n = len(params)
    figsize = figsize or (6, 3 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=False)

    if n == 1:
        axes = [axes]

    # Default colour sequence used when no colour is assigned to an estimator.
    _default_colors = ["k", "r", "b", "m", "g", "c", "y", "orange"]

    for ax, param in zip(axes, params):
        true_val = getattr(results[0]["params"], param)
        names = _estimators_with_scalar(results, param)

        if not names:
            ax.set_ylabel(param)
            ax.set_title(f"{param} — true = {true_val:.2f}")
            continue

        for idx, name in enumerate(names):
            vals, errs = [], []
            for r in results:
                bf = r.get(name, {}).get("best_fit", {})
                v = bf.get(param)
                if v is not None and not hasattr(v, "__len__"):
                    vals.append(float(v))
                    se = bf.get("std_errors", {})
                    errs.append(
                        float(se.get(param, 0.0)) if isinstance(se, dict) else 0.0
                    )

            color = _default_colors[idx % len(_default_colors)]
            x = np.full(len(vals), idx) + np.random.default_rng(42).uniform(
                -0.15, 0.15, len(vals)
            )
            ax.errorbar(
                x,
                vals,
                yerr=errs,
                fmt="o",
                color=color,
                alpha=0.5,
                markersize=3,
                capsize=0,
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.axhline(true_val, color="gray", ls="--", alpha=0.7)
        ax.set_ylabel(param)
        ax.set_title(f"{param} — true = {true_val:.2f}")

    fig.tight_layout()
    return axes


def plot_estimation_times(results, estimators=None, bins=20, figsize=(7, 5)):
    """Histogram of per-realisation estimation time for each estimator.

    Parameters
    ----------
    results : list of dict
        Per-realisation outputs from :class:`MonteCarlo`.
    estimators : list of str, optional
        Estimator names to include.  Defaults to all estimator keys
        found in the results.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.

    Returns
    -------
    Axes
    """
    if estimators is None:
        estimators = [
            k for k in results[0] if k not in ("sim_cat", "det_cat", "params")
        ]

    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "GC": "k",
        "GC (fixed back)": "r",
        "PSF": "b",
        "Aperture + AC": "m",
    }

    for name in estimators:
        times = [
            r[name]["extra"]["estimation_time"]
            for r in results
            if name in r and "extra" in r[name]
        ]
        if not times:
            continue
        color = colors.get(name, "k")
        ax.hist(times, bins=bins, alpha=0.5, color=color, label=name)

    ax.set_xlabel("Estimation time [s]")
    ax.set_ylabel("Number of realisations")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_results(path: str, results: list[dict]) -> str:
    """Save MC results to a pickle file.

    Parameters
    ----------
    path : str
        Output file path (``.pkl`` appended if missing).
    results : list of dict
        Per-realization outputs from :class:`MonteCarlo`.

    Returns
    -------
    str
        The actual path written to.
    """
    if not path.endswith(".pkl"):
        path += ".pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    return path


def load_results(path: str) -> list[dict]:
    """Load MC results saved with :func:`save_results`.

    Parameters
    ----------
    path : str
        Path to ``.pkl`` file.

    Returns
    -------
    list of dict
        Per-realization outputs.
    """
    if not os.path.exists(path):
        candidate = path + ".pkl"
        if os.path.exists(candidate):
            path = candidate
        else:
            raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)
