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
    """Two-step growth-curve fit with background fixed to the mean
    fitted value of the free-background fit."""
    fit_kwargs = fit_kwargs or {}
    fitter = gcp.Fitter(cog, bads=_bads(detections["det_cat"]))
    bf, _ = fitter.fit(**fit_kwargs, desc="GC fix back (1)")
    fitter.detect_contamination(bf)
    bf, _ = fitter.fit(**fit_kwargs, compute_uncertainty=True, desc="GC fix back (2)")
    ref = fitter.results(bf)
    mean_back = float(np.mean(ref["back"]))
    bf_fixed, _ = fitter.fit(
        **fit_kwargs,
        fix={"back": np.full(len(ref["back"]), mean_back)},
        compute_uncertainty=True,
        desc="GC fix back (3)",
    )
    fitted = fitter.results(bf_fixed)
    return {
        "best_fit": fitted,
        "uncertainty": fitted.get("std_errors"),
    }


@timed_estimator
def aperture_estimator(image, detections, cog, fit_kwargs=None):
    """Aperture photometry with aperture correction derived from the
    fitted Moffat profile."""
    fit_kwargs = fit_kwargs or {}
    fitter = gcp.Fitter(cog, bads=_bads(detections["det_cat"]))
    bf, _ = fitter.fit(**fit_kwargs, desc="AC prep (1)")
    fitter.detect_contamination(bf)
    bf, _ = fitter.fit(**fit_kwargs, compute_uncertainty=True, desc="AC prep (2)")
    fitted = fitter.results(bf)

    fwhm = gcp.gcmodel.gamma2fwhm(fitted["gamma"], fitted["alpha"])
    r_core_idx = np.argmin(np.abs(cog["radius"] - fwhm))
    r_corr_idx = np.argmin(np.abs(cog["radius"] - 3 * fwhm))
    r_core = cog["radius"][r_core_idx]
    r_corr = cog["radius"][r_corr_idx]
    ac = gcp.gcmodel.moffat_flux(
        r_corr, fitted["gamma"], fitted["alpha"]
    ) / gcp.gcmodel.moffat_flux(r_core, fitted["gamma"], fitted["alpha"])
    bkg = fitted["back"][fitter.kept]
    flux_ap = (
        cog["flux_clean"][:, r_core_idx][fitter.kept] - bkg * np.pi * r_core**2
    ) * ac
    flux_ap_full = np.full(len(detections["det_cat"]), np.nan)
    flux_ap_full[fitter.kept] = flux_ap

    return {
        "best_fit": {"flux": flux_ap_full},
        "uncertainty": None,
    }


@timed_estimator
def psf_estimator(image, detections, cog, background=100.0, nstars=30, fit_shape=11):
    """PSF photometry estimator."""
    psf_results, _ = gcp.psf_photometry(
        image - background, detections["det_cat"], nstars=nstars, fit_shape=fit_shape
    )
    return {
        "best_fit": {"flux": np.asarray(psf_results["flux_fit"])},
        "uncertainty": None,
    }


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
        "PSF": partial(psf_estimator, background=cfg.background),
        "Aperture + AC": partial(aperture_estimator, fit_kwargs=cfg.fit_kwargs),
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
        "det_cat": det_cat,
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

    Each realization draws a new random catalog.

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
    """

    def __init__(
        self,
        config: SimulationConfig,
        n_realizations: int,
        seed: int | None = None,
        estimators: dict | None = None,
    ):
        self.config = config
        self.n_realizations = n_realizations
        self.seed = seed
        self.estimators = estimators or default_estimators(config)
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

            catalog = gcp.make_realistic_source_catalog(
                n_sources=self.config.n_sources,
                shape=self.config.shape,
                seed=int(sd),
            )
            image, _ = gcp.simulate_image(
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
