"""Monte Carlo framework for photometry bias and coverage analysis.

Provides tools to run multiple realizations of a photometry pipeline and
compute bias and coverage statistics for fitted parameters.

Parameter types
---------------
The framework distinguishes three kinds of parameters:

- **Flux** — per-source flux estimates, studied as bias vs true flux
  (:func:`compute_bias_coverage`, :func:`plot_bias_coverage`).
- **Background** — per-source background estimates, also vs flux
  (:func:`compute_bias_coverage`, :func:`plot_background_bias`).
- **Nuisance** — global scalar parameters (gamma, alpha) fitted once
  per realization (:func:`compute_nuisance_stats`,
  :func:`plot_nuisance_summary`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import gcphotom as gcp
from gcphotom.stats import bin_statistic


@dataclass
class SimulationConfig:
    """Configuration for a Monte Carlo simulation.

    A new random catalog is drawn for each realization.

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


def default_pipeline(image: np.ndarray, catalog: Any, cfg: SimulationConfig) -> dict:
    """Default photometry pipeline mirroring the flux_reconstruction_quality example.

    Parameters
    ----------
    image : ndarray
        Simulated image.
    catalog : Table
        Source catalog used for this realization.
    cfg : SimulationConfig
        Simulation configuration.

    Returns
    -------
    dict
        Keys: ``fitted``, ``fitted_no_back``, ``psf``, ``aperture``,
        ``input_cat``, ``fitter``, ``catalog``.
    """
    seg, det_cat, bkg_map, bkg_var_map = gcp.detect_and_segment(
        image, n_pixels=cfg.n_pixels
    )
    bads = (det_cat.ellipticity * det_cat.area).value > 6

    cog = gcp.extract_growth_curves(
        image,
        det_cat,
        segmentation_image=seg,
        background_variance=bkg_var_map,
        desc="COG",
    )

    fitter = gcp.Fitter(cog, bads=bads)
    best_fit, _ = fitter.fit(**cfg.fit_kwargs, desc="Fit (1)")
    fitter.detect_contamination(best_fit)
    best_fit, _ = fitter.fit(**cfg.fit_kwargs, compute_uncertainty=True, desc="Fit (2)")

    best_fit_no_back, _ = fitter.fit(
        **cfg.fit_kwargs,
        fix={"back": np.full(len(best_fit["back"]), np.mean(best_fit["back"]))},
        compute_uncertainty=True,
        desc="Fit (fix back)",
    )

    fitted = fitter.results(best_fit)
    fitted_no_back = fitter.results(best_fit_no_back)
    input_cat = gcp.cross_match(det_cat, catalog)

    # PSF photometry
    psf_results, _ = gcp.psf_photometry(
        image - cfg.background, det_cat, nstars=30, fit_shape=11
    )
    psf_cat = gcp.cross_match(psf_results, catalog)

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
        "catalog": catalog,
        "psf": {"results": psf_results, "cat": psf_cat},
        "aperture": {"flux": flux_ap_full},
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
    """Run a photometry pipeline over multiple independent realizations.

    Each realization draws a new random catalog.

    Parameters
    ----------
    config : SimulationConfig
        Simulation parameters (PSF, noise, image size, etc.).
    n_realizations : int
        Number of independent realizations.
    seed : int or None
        Master random seed. Each realization draws a sub-seed.
    pipeline : callable, optional
        Function ``(image, catalog, config) -> dict``. Defaults to
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

    def run(
        self, verbose: bool = True, show_progress: bool = True
    ) -> MonteCarloResults:
        """Execute all realizations.

        Parameters
        ----------
        verbose : bool
            Print progress every 10 realizations (legacy, use ``show_progress``
            instead).
        show_progress : bool
            If ``True``, display a progress bar.

        Returns
        -------
        MonteCarloResults
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
        for i, seed in enumerate(seeds):
            if (
                not show_progress
                and verbose
                and i % max(1, self.n_realizations // 10) == 0
            ):
                print(f"  Realization {i + 1}/{self.n_realizations}")

            catalog = gcp.make_realistic_source_catalog(
                n_sources=self.config.n_sources,
                shape=self.config.shape,
                seed=int(seed),
            )
            image, _ = gcp.simulate_image(
                shape=self.config.shape,
                catalog=catalog,
                gamma=self.config.gamma,
                alpha=self.config.alpha,
                background=self.config.background,
                read_noise=self.config.read_noise,
                seed=int(seed),
            )

            try:
                res = self.pipeline(image, catalog, self.config)
                self._results.append(res)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                warnings.warn(f"Realization {i + 1} failed: {exc}")
            finally:
                pbar.update(1)

        pbar.close()
        return MonteCarloResults(
            realized=len(self._results),
            total=self.n_realizations,
            results=self._results,
        )


def _collect_data(
    results: list[dict],
    get_x: Callable[[dict], np.ndarray],
    get_truth: Callable[[dict], np.ndarray],
    get_estimate: Callable[[dict], np.ndarray],
    get_std: Callable[[dict], np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Collect (x, truth, estimate, std) arrays from MC results.

    Parameters
    ----------
    results : list of dict
        Per-realization pipeline outputs.
    get_x : callable
        Returns the binning variable (e.g. true flux).
    get_truth : callable
        Returns the truth value for the parameter of interest.
    get_estimate : callable
        Returns the estimated value.
    get_std : callable or None
        Returns the standard error array.

    Returns
    -------
    x, truth, estimate, std : ndarray
        Flattened arrays over all realizations (NaNs dropped).
    """
    xs, truths, ests, stds = [], [], [], []

    for res in results:
        x = np.asarray(get_x(res))
        t = np.asarray(get_truth(res))
        e = np.asarray(get_estimate(res))

        mask = np.isfinite(x) & np.isfinite(t) & np.isfinite(e)
        xs.append(x[mask])
        truths.append(t[mask])
        ests.append(e[mask])

        if get_std is not None:
            s = np.asarray(get_std(res))
            stds.append(s[mask])

    return (
        np.concatenate(xs),
        np.concatenate(truths),
        np.concatenate(ests),
        np.concatenate(stds) if stds else None,
    )


def _build_estimator_spec(
    get_x: Callable,
    get_truth: Callable,
    get_estimate: Callable,
    get_std: Callable | None = None,
    bias_scale: float = 100.0,
) -> dict:
    """Build an estimator spec dict from extractor callables."""
    return {
        "get_x": get_x,
        "get_truth": get_truth,
        "get_estimate": get_estimate,
        "get_std": get_std,
        "bias_scale": bias_scale,
    }


def _default_flux_estimators() -> dict[str, dict]:
    """Build default specs for flux estimators."""
    return {
        "GC (est. back)": _build_estimator_spec(
            get_x=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_truth=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_estimate=lambda r: np.asarray(r["fitted"]["flux"]),
            get_std=lambda r: np.asarray(r["fitted"]["std_errors"]["flux"]),
        ),
        "GC (fixed back)": _build_estimator_spec(
            get_x=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_truth=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_estimate=lambda r: np.asarray(r["fitted_no_back"]["flux"]),
            get_std=lambda r: np.asarray(r["fitted_no_back"]["std_errors"]["flux"]),
        ),
        "PSF photometry": _build_estimator_spec(
            get_x=lambda r: np.asarray(r["psf"]["cat"]["flux"]),
            get_truth=lambda r: np.asarray(r["psf"]["cat"]["flux"]),
            get_estimate=lambda r: np.asarray(r["psf"]["results"]["flux_fit"]),
            get_std=None,
        ),
        "Aperture + AC": _build_estimator_spec(
            get_x=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_truth=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_estimate=lambda r: np.asarray(r["aperture"]["flux"]),
            get_std=None,
        ),
    }


def _build_nuisance_spec(
    param: str,
    truth_value: float,
    std_key: str | None,
    bias_scale: float = 100.0,
) -> dict:
    """Build spec for a scalar nuisance parameter (gamma, alpha).

    The parameter is fitted as a single scalar per realization. The spec
    is designed for use with :func:`compute_nuisance_stats`.
    """

    def get_estimate(r: dict) -> np.ndarray:
        return np.asarray(r["fitted"][param])

    return {
        "truth": truth_value,
        "get_estimate": get_estimate,
        "std_key": std_key,
        "bias_scale": bias_scale,
    }


def _build_background_spec(cfg: SimulationConfig) -> dict:
    """Build estimator spec for per-source background estimates.

    Unlike global nuisance parameters, the fitted background is
    per-source and can be studied as a function of flux.
    """

    def get_x(r: dict) -> np.ndarray:
        return np.asarray(r["input_cat"]["flux"])

    def get_truth(r: dict) -> np.ndarray:
        return np.full(len(r["input_cat"]), cfg.background)

    def get_estimate(r: dict) -> np.ndarray:
        return np.asarray(r["fitted"]["back"])

    def get_std(r: dict) -> np.ndarray | None:
        se = r["fitted"].get("std_errors", {})
        try:
            return np.asarray(se["back"])
        except (KeyError, TypeError):
            return np.full(len(r["input_cat"]), np.nan)

    return _build_estimator_spec(
        get_x=get_x,
        get_truth=get_truth,
        get_estimate=get_estimate,
        get_std=get_std,
        bias_scale=100.0,
    )


def build_default_estimators(cfg: SimulationConfig) -> dict[str, dict]:
    """Build all default estimators, grouped by parameter type.

    Returns a dict with three keys, each containing estimator specs
    suitable for the corresponding compute/plot functions:

    **flux** — per-source flux bias and coverage vs simulated flux.
    Use with :func:`compute_bias_coverage` and :func:`plot_bias_coverage`.

    **background** — per-source background bias vs simulated flux.
    Use with :func:`compute_bias_coverage` and :func:`plot_background_bias`.

    **nuisance** — global scalar parameters (gamma, alpha) fitted once
    per realization. Use with :func:`compute_nuisance_stats` and
    :func:`plot_nuisance_summary`.
    """
    return {
        "flux": _default_flux_estimators(),
        "background": {"Background": _build_background_spec(cfg)},
        "nuisance": {
            "Gamma": _build_nuisance_spec(
                "gamma", cfg.gamma, "gamma", bias_scale=100.0
            ),
            "Alpha": _build_nuisance_spec(
                "alpha", cfg.alpha, "alpha", bias_scale=100.0
            ),
        },
    }


def compute_bias_coverage(
    results: list[dict],
    *,
    estimators: dict[str, dict] | None = None,
    nbins: int = 10,
    bins: np.ndarray | None = None,
    sigma_levels: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, dict]:
    """Compute binned bias, RMS and coverage for one or more estimators.

    Parameters
    ----------
    results : list of dict
        Per-realization pipeline outputs from :class:`MonteCarlo`.
    estimators : dict, optional
        Mapping from label to spec dict with keys ``get_x``, ``get_truth``,
        ``get_estimate``, and optionally ``get_std`` and ``bias_scale``.
    nbins : int
        Number of bins along the x-axis.
    bins : ndarray, optional
        Explicit bin edges.
    sigma_levels : tuple of float
        Sigma levels for coverage computation.

    Returns
    -------
    dict
        Mapping from label to dict with keys ``xbins``, ``bias``,
        ``bias_err``, ``rms``, and ``coverage_<sigma>`` for each
        sigma level.
    """
    out = {}

    for label, spec in estimators.items():
        get_x = spec["get_x"]
        get_truth = spec["get_truth"]
        get_estimate = spec["get_estimate"]
        get_std = spec.get("get_std")
        bias_scale = spec.get("bias_scale", 100.0)

        x, truth, estimate, se = _collect_data(
            results, get_x, get_truth, get_estimate, get_std
        )

        if len(x) == 0:
            warnings.warn(f"No data for estimator '{label}', skipping.")
            continue

        bias = (estimate / truth - 1.0) * bias_scale

        if bins is None:
            from gcphotom.stats import _build_bins

            valid = np.isfinite(x) & np.isfinite(bias)
            edges = _build_bins(x[valid], nbins, logbins=True)
        else:
            edges = bins
        xb = 0.5 * (edges[:-1] + edges[1:])

        _, bias_binned, bias_err = bin_statistic(
            x,
            bias,
            nbins=nbins,
            bins=edges,
            logbins=bins is None,
            method="median",
            scale_err=True,
        )

        _, rms_binned, _ = bin_statistic(
            x,
            np.abs(bias),
            nbins=nbins,
            bins=edges,
            logbins=bins is None,
            method="median",
            scale_err=True,
        )

        entry = {
            "xbins": xb,
            "bias": bias_binned,
            "bias_err": bias_err,
            "rms": rms_binned,
        }

        if se is not None:
            for sigma in sigma_levels:
                resids = np.abs(estimate - truth)
                covered = (resids <= sigma * se).astype(float)
                _, cov_binned, _ = bin_statistic(
                    x,
                    covered,
                    nbins=nbins,
                    bins=edges,
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


def plot_bias_coverage(
    stats: dict[str, dict],
    *,
    ax_bias: plt.Axes | None = None,
    ax_coverage: plt.Axes | None = None,
    sigma_level: float = 1.0,
    expected_coverage: float | None = None,
    figsize: tuple = (12, 8),
) -> tuple[plt.Axes, plt.Axes]:
    """Plot flux bias (with RMS shaded region) and coverage vs simulated flux.

    Designed for per-source flux estimators. For nuisance parameters
    (gamma, alpha) use :func:`plot_nuisance_summary` instead.

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
        "Background": "c",
        "Gamma": "g",
        "Alpha": "orange",
    }

    for label, s in stats.items():
        color = colors.get(label, "k")

        # Shaded RMS region
        if "rms" in s:
            ax_bias.fill_between(
                s["xbins"],
                s["bias"] - s["rms"],
                s["bias"] + s["rms"],
                alpha=0.15,
                color=color,
                zorder=3,
            )

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


def compute_nuisance_stats(
    results: list[dict],
    *,
    estimators: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Compute bias statistics for scalar nuisance parameters across realizations.

    Parameters
    ----------
    results : list of dict
        Per-realization pipeline outputs from :class:`MonteCarlo`.
    estimators : dict, optional
        Mapping from label to spec dict with keys ``truth``,
        ``get_estimate``, and optionally ``bias_scale``.

    Returns
    -------
    dict
        Mapping from label to dict with keys:

        - ``truth`` — scalar truth value
        - ``estimates`` — array of per-realization estimated values
        - ``bias`` — per-realization bias (percent, unless overridden)
        - ``mean_bias`` — mean bias across realizations
        - ``std_bias`` — standard deviation of bias
        - ``rms_bias`` — RMS bias
    """
    out = {}
    for label, spec in estimators.items():
        truth = spec["truth"]
        bias_scale = spec.get("bias_scale", 100.0)
        get_estimate = spec["get_estimate"]

        estimates = []
        for res in results:
            raw = np.asarray(get_estimate(res))
            estimates.append(float(np.nanmean(raw)))
        estimates = np.array(estimates)

        valid = np.isfinite(estimates)
        if not np.any(valid):
            warnings.warn(
                f"No valid estimates for nuisance parameter '{label}', skipping."
            )
            continue

        bias = (estimates / truth - 1.0) * bias_scale

        out[label] = {
            "truth": truth,
            "estimates": estimates,
            "bias": bias,
            "mean_bias": float(np.mean(bias[valid])),
            "std_bias": float(np.std(bias[valid])),
            "rms_bias": float(np.sqrt(np.mean(bias[valid] ** 2))),
        }

    return out


def plot_nuisance_summary(
    nuisance_stats: dict[str, dict],
    *,
    figsize: tuple = (10, 4),
    nbins: int = 30,
) -> plt.Figure:
    """Plot histogram summary of nuisance parameter bias across realizations.

    For each nuisance parameter (e.g. Gamma, Alpha) a histogram of
    fitted values is shown with the truth value marked as a vertical
    dashed line. The mean bias and scatter are annotated.

    Parameters
    ----------
    nuisance_stats : dict
        Output from :func:`compute_nuisance_stats`.
    figsize : tuple
        Figure width, height.
    nbins : int
        Number of histogram bins.

    Returns
    -------
    Figure
    """
    n_params = len(nuisance_stats)
    if n_params == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No nuisance parameter data", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_params, figsize=figsize, squeeze=False)
    for ax, (label, s) in zip(axes[0], nuisance_stats.items()):
        est = s["estimates"]
        truth = s["truth"]
        valid = np.isfinite(est)
        if not np.any(valid):
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            continue

        ax.hist(est[valid], bins=nbins, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(truth, color="k", ls="--", lw=2, label=f"Truth = {truth:.3g}")
        ax.set_xlabel(f"Fitted {label}")
        ax.set_ylabel("Realizations")

        # Annotate bias stats
        mean_b = s["mean_bias"]
        std_b = s["std_bias"]
        ax.annotate(
            f"Bias: {mean_b:+.2f}% ± {std_b:.2f}%",
            xycoords="axes fraction",
            xy=(0.05, 0.95),
            va="top",
            fontsize=9,
        )
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


def plot_background_bias(
    stats: dict[str, dict],
    *,
    ax: plt.Axes | None = None,
    figsize: tuple = (6, 5),
) -> plt.Axes:
    """Plot background bias (with RMS shaded region) vs simulated flux.

    Parameters
    ----------
    stats : dict
        Output from :func:`compute_bias_coverage` for a background
        estimator.
    ax : Axes, optional
        Target axes. Created if None.
    figsize : tuple
        Figure size when creating a new axes.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for label, s in stats.items():
        if "rms" in s:
            ax.fill_between(
                s["xbins"],
                s["bias"] - s["rms"],
                s["bias"] + s["rms"],
                alpha=0.15,
                color="c",
                zorder=3,
            )
        ax.errorbar(
            s["xbins"],
            s["bias"],
            yerr=s["bias_err"],
            marker="o",
            ls="none",
            color="c",
            label=label,
            zorder=5,
        )

    ax.axhline(0, color="k", ls="--", alpha=0.3, zorder=0)
    ax.set_xlabel("Simulated flux [ADU]")
    ax.set_xscale("log")
    ax.set_ylabel("Background bias (%)")
    ax.legend(loc="best", frameon=False)

    return ax
