"""Monte Carlo framework for flux estimator bias and coverage analysis.

Provides tools to run multiple realizations of a photometry pipeline and
compute bias and coverage statistics for the fitted fluxes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

import gcphotom as gcp
from gcphotom.stats import bin_statistic


@dataclass
class SimulationConfig:
    """Configuration for a single Monte Carlo simulation run.

    Parameters
    ----------
    catalog : Table
        Source catalog with columns ``x``, ``y``, ``flux``. Kept fixed across
        realizations; only the noise varies.
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
    nbins : int
        Number of flux bins for bias/coverage statistics.
    """

    catalog: Any
    shape: tuple = (1024, 1024)
    gamma: float = 3.0
    alpha: float = 3.0
    background: float = 100.0
    read_noise: float = 5.0
    n_pixels: int = 5
    fit_kwargs: dict = field(
        default_factory=lambda: {"learning_rate": 1e-2, "niter": 2000}
    )
    nbins: int = 10


def default_pipeline(image: np.ndarray, cfg: SimulationConfig) -> dict:
    """Default photometry pipeline mirroring the flux_reconstruction_quality example.

    Runs detection, COG extraction, iterative fitting with contamination
    detection, and a fixed-background refit. Also computes PSF photometry
    and aperture photometry with a constant aperture correction.

    Returns
    -------
    dict
        Keys: ``fitted``, ``fitted_no_back``, ``psf``, ``aperture``,
        ``input_cat``, ``fitter``.
    """
    seg, det_cat, bkg_map, bkg_var_map = gcp.detect_and_segment(
        image, n_pixels=cfg.n_pixels
    )
    bads = (det_cat.ellipticity * det_cat.area).value > 6

    cog = gcp.extract_growth_curves(
        image, det_cat, segmentation_image=seg, background_variance=bkg_var_map
    )

    fitter = gcp.Fitter(cog, bads=bads)
    best_fit, _ = fitter.fit(**cfg.fit_kwargs)
    fitter.detect_contamination(best_fit)
    best_fit, _ = fitter.fit(**cfg.fit_kwargs)

    best_fit_no_back, _ = fitter.fit(
        **cfg.fit_kwargs,
        fix={"back": np.full(len(best_fit["back"]), np.mean(best_fit["back"]))},
    )

    fitted = fitter.results(best_fit)
    fitted_no_back = fitter.results(best_fit_no_back)
    input_cat = gcp.cross_match(det_cat, cfg.catalog)

    # PSF photometry
    psf_results, _ = gcp.psf_photometry(
        image - cfg.background, det_cat, nstars=30, fit_shape=11
    )
    psf_cat = gcp.cross_match(psf_results, cfg.catalog)

    # Aperture photometry with constant correction
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
    flux_ap_full = np.full(len(det_cat), np.nan)
    flux_ap_full[fitter.kept] = flux_ap
    kept_cat = input_cat[fitter.kept]

    return {
        "fitted": fitted,
        "fitted_no_back": fitted_no_back,
        "fitter": fitter,
        "input_cat": input_cat,
        "psf": {"results": psf_results, "cat": psf_cat},
        "aperture": {"flux": flux_ap_full, "cat": kept_cat},
    }


@dataclass
class MonteCarloResults:
    """Container for per-realization results.

    Attributes
    ----------
    realized : int
        Number of successfully completed realizations.
    total : int
        Total number of attempted realizations.
    results : list of dict
        Per-realization output from the pipeline callable.
    """

    realized: int
    total: int
    results: list[dict]


class MonteCarlo:
    """Run a photometry pipeline over multiple noisy realizations.

    Parameters
    ----------
    config : SimulationConfig
        Fixed simulation parameters and catalog.
    n_realizations : int
        Number of independent noise realizations.
    seed : int or None
        Master random seed. Each realization draws a sub-seed.
    pipeline : callable, optional
        Function ``(image, config) -> dict``. Defaults to
        :func:`default_pipeline`.
    """

    def __init__(
        self,
        config: SimulationConfig,
        n_realizations: int,
        seed: int | None = None,
        pipeline: Callable | None = None,
    ):
        self.config = config
        self.n_realizations = n_realizations
        self.seed = seed
        self.pipeline = pipeline or default_pipeline
        self._results: list[dict] = []

    @property
    def results(self) -> list[dict]:
        """List of per-realization pipeline outputs."""
        return self._results

    def run(self, verbose: bool = True) -> MonteCarloResults:
        """Execute all realizations.

        Parameters
        ----------
        verbose : bool
            Print progress every 10 realizations.

        Returns
        -------
        MonteCarloResults
        """
        self._results = []
        rng = np.random.default_rng(self.seed)
        seeds = rng.integers(0, 2**31, self.n_realizations)

        for i, seed in enumerate(seeds):
            if verbose and i % max(1, self.n_realizations // 10) == 0:
                print(f"  Realization {i + 1}/{self.n_realizations}")

            image, _ = gcp.simulate_image(
                shape=self.config.shape,
                catalog=self.config.catalog,
                gamma=self.config.gamma,
                alpha=self.config.alpha,
                background=self.config.background,
                read_noise=self.config.read_noise,
                seed=int(seed),
            )

            try:
                res = self.pipeline(image, self.config)
                self._results.append(res)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                warnings.warn(f"Realization {i + 1} failed: {exc}")

        return MonteCarloResults(
            realized=len(self._results),
            total=self.n_realizations,
            results=self._results,
        )


def _estimator_data(
    results: list[dict],
    get_cat: Callable[[dict], Any],
    get_flux: Callable[[dict], Any],
    get_std: Callable[[dict], Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract (true_flux, est_flux, std_err) arrays from MC results.

    Parameters
    ----------
    results : list of dict
        Per-realization pipeline outputs.
    get_cat : callable
        Extracts the matched catalog from a result dict.
    get_flux : callable
        Extracts the estimated flux array from a result dict.
    get_std : callable or None
        Extracts the standard error array from a result dict.

    Returns
    -------
    true_flux, est_flux, std_err : ndarray
        Flattened arrays over all realizations (NaNs dropped).
    """
    true_fluxes = []
    est_fluxes = []
    std_errs = []

    for res in results:
        cat = get_cat(res)
        true_f = np.asarray(cat["flux"])
        est_f = np.asarray(get_flux(res))

        mask = np.isfinite(true_f) & np.isfinite(est_f)
        true_fluxes.append(true_f[mask])
        est_fluxes.append(est_f[mask])

        if get_std is not None:
            std_f = np.asarray(get_std(res))
            std_errs.append(std_f[mask])

    tf = np.concatenate(true_fluxes)
    ef = np.concatenate(est_fluxes)
    se = np.concatenate(std_errs) if std_errs else None
    return tf, ef, se


def compute_bias_coverage(
    results: list[dict],
    *,
    estimators: dict[str, dict] | None = None,
    nbins: int = 10,
    bins: np.ndarray | None = None,
    sigma_levels: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, dict]:
    """Compute binned bias and coverage for one or more flux estimators.

    Parameters
    ----------
    results : list of dict
        Per-realization pipeline outputs from :class:`MonteCarlo`.
    estimators : dict, optional
        Mapping from label to extractor specification. Each value is a dict
        with keys ``flux_key``, ``std_key`` (optional), and ``cat_key``
        (optional). Defaults cover the standard estimators from
        :func:`default_pipeline`.
    nbins : int
        Number of flux bins.
    bins : ndarray, optional
        Explicit bin edges.
    sigma_levels : tuple of float
        Sigma levels for coverage computation.

    Returns
    -------
    dict
        Mapping from label to dict with keys ``xbins``, ``bias``, ``bias_err``,
        and ``coverage_<sigma>`` for each sigma level.
    """
    if estimators is None:
        estimators = {
            "GC (est. back)": {
                "get_cat": lambda r: r["input_cat"],
                "get_flux": lambda r: r["fitted"]["flux"],
                "get_std": lambda r: r["fitted"]["std_errors"]["flux"],
            },
            "GC (fixed back)": {
                "get_cat": lambda r: r["input_cat"],
                "get_flux": lambda r: r["fitted_no_back"]["flux"],
                "get_std": lambda r: r["fitted_no_back"]["std_errors"]["flux"],
            },
            "PSF photometry": {
                "get_cat": lambda r: r["psf"]["cat"],
                "get_flux": lambda r: r["psf"]["results"]["flux_fit"],
                "get_std": None,
            },
            "Aperture + AC": {
                "get_cat": lambda r: r["aperture"]["cat"],
                "get_flux": lambda r: r["aperture"]["flux"],
                "get_std": None,
            },
        }

    out = {}

    for label, spec in estimators.items():
        get_cat = spec["get_cat"]
        get_flux = spec["get_flux"]
        get_std = spec.get("get_std")

        tf, ef, se = _estimator_data(results, get_cat, get_flux, get_std)

        if len(tf) == 0:
            warnings.warn(f"No data for estimator '{label}', skipping.")
            continue

        bias = (ef / tf - 1.0) * 100.0

        if bins is None:
            valid = np.isfinite(tf) & np.isfinite(bias)
            xb, _, _ = _quick_bins(tf[valid], nbins)
        else:
            xb = 0.5 * (bins[:-1] + bins[1:])

        _, bias_binned, bias_err = bin_statistic(
            tf,
            bias,
            nbins=nbins,
            bins=bins,
            logbins=bins is None,
            method="median",
            scale_err=True,
        )

        entry = {"xbins": xb, "bias": bias_binned, "bias_err": bias_err}

        if se is not None:
            resids = np.abs(ef - tf)
            for sigma in sigma_levels:
                covered = (resids <= sigma * se).astype(float)
                _, cov_binned, _ = bin_statistic(
                    tf,
                    covered,
                    nbins=nbins,
                    bins=bins,
                    logbins=bins is None,
                    method="mean",
                    scale_err=False,
                )
                entry[f"coverage_{sigma}sigma"] = cov_binned
        else:
            for sigma in sigma_levels:
                entry[f"coverage_{sigma}sigma"] = np.full_like(bias_binned, np.nan)

        out[label] = entry

    return out


def _quick_bins(x: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build log-spaced bins and return centers, edges, and a dummy."""
    from gcphotom.stats import _build_bins

    edges = _build_bins(x, nbins, logbins=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges, np.zeros(len(centers))


def plot_bias_coverage(
    stats: dict[str, dict],
    *,
    ax_bias: plt.Axes | None = None,
    ax_coverage: plt.Axes | None = None,
    sigma_level: float = 1.0,
    expected_coverage: float | None = None,
    figsize: tuple = (12, 8),
) -> tuple[plt.Axes, plt.Axes]:
    """Plot bias and coverage as a function of simulated flux.

    Parameters
    ----------
    stats : dict
        Output from :func:`compute_bias_coverage`.
    ax_bias, ax_coverage : Axes, optional
        Target axes. Created if None.
    sigma_level : float
        Which sigma level's coverage to plot.
    expected_coverage : float, optional
        Expected coverage fraction at ``sigma_level`` (e.g. 0.683 for 1-sigma).
        Plotted as a horizontal reference line.
    figsize : tuple
        Figure size when creating new axes.

    Returns
    -------
    ax_bias, ax_coverage : Axes
    """
    if ax_bias is None or ax_coverage is None:
        fig, (ax_bias, ax_coverage) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1]}
        )

    cov_key = f"coverage_{sigma_level}sigma"

    colors = {
        "GC (est. back)": "k",
        "GC (fixed back)": "r",
        "PSF photometry": "b",
        "Aperture + AC": "m",
    }

    for label, s in stats.items():
        color = colors.get(label, "k")

        ax_bias.errorbar(
            s["xbins"],
            s["bias"],
            yerr=s["bias_err"],
            marker="o",
            ls="none",
            color=color,
            label=label,
            zorder=5,
        )

        if cov_key in s:
            ax_coverage.errorbar(
                s["xbins"],
                s[cov_key],
                marker="s",
                ls="none",
                color=color,
                label=label,
                zorder=5,
            )

    ax_bias.axhline(0, color="k", ls="--", alpha=0.3, zorder=0)
    ax_bias.set_xlabel("Simulated flux [ADU]")
    ax_bias.set_xscale("log")
    ax_bias.set_ylabel("Bias (%)")
    ax_bias.legend(loc="best", frameon=False)

    if expected_coverage is None:
        from scipy import stats as sp_stats

        expected_coverage = float(
            sp_stats.norm.cdf(sigma_level) - sp_stats.norm.cdf(-sigma_level)
        )

    ax_coverage.axhline(expected_coverage, color="k", ls="--", alpha=0.3, zorder=0)
    ax_coverage.set_xlabel("Simulated flux [ADU]")
    ax_coverage.set_xscale("log")
    ax_coverage.set_ylabel(f"Coverage ({sigma_level}-sigma)")
    ax_coverage.set_ylim(0, 1.05)
    ax_coverage.legend(loc="best", frameon=False)

    return ax_bias, ax_coverage
