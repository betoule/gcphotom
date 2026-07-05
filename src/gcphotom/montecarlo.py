"""Monte Carlo framework for photometry bias and coverage analysis.

Each realization runs a photometry pipeline on a random simulated image and
returns a flat dict of per-estimator results.  N realizations is simply a
list of such dicts, saved/loaded with pickle.
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import gcphotom as gcp
from gcphotom.stats import bin_statistic


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


def default_pipeline(image: np.ndarray, catalog, cfg: SimulationConfig) -> dict:
    """Run the default photometry pipeline and return per-estimator results.

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
        Keys are ``"params"`` (the *cfg*) and one entry per estimator
        (e.g. ``"GC (est. back)"``, ``"Background"``, ``"Gamma"``, etc.).
        Each estimator entry is ``{"best_fit": ..., "uncertainty": ...,
        "extra": {"truth": ..., "bias_scale": 100.0}}``.
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

    fitted = fitter.results(best_fit)

    best_fit_no_back, _ = fitter.fit(
        **cfg.fit_kwargs,
        fix={"back": np.full(len(best_fit["back"]), np.mean(best_fit["back"]))},
        compute_uncertainty=True,
        desc="Fit (fix back)",
    )

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

    flux_truth = np.asarray(input_cat["flux"])
    bg_truth = np.full(len(input_cat), cfg.background)

    return {
        "params": cfg,
        "GC (est. back)": {
            "best_fit": np.asarray(fitted["flux"]),
            "uncertainty": np.asarray(fitted["std_errors"]["flux"]),
            "extra": {"truth": flux_truth, "bias_scale": 100.0},
        },
        "GC (fixed back)": {
            "best_fit": np.asarray(fitted_no_back["flux"]),
            "uncertainty": np.asarray(fitted_no_back["std_errors"]["flux"]),
            "extra": {"truth": flux_truth, "bias_scale": 100.0},
        },
        "PSF photometry": {
            "best_fit": np.asarray(psf_results["flux_fit"]),
            "uncertainty": None,
            "extra": {"truth": np.asarray(psf_cat["flux"]), "bias_scale": 100.0},
        },
        "Aperture + AC": {
            "best_fit": flux_ap_full,
            "uncertainty": None,
            "extra": {"truth": flux_truth, "bias_scale": 100.0},
        },
        "Background": {
            "best_fit": np.asarray(fitted["back"]),
            "uncertainty": np.asarray(fitted["std_errors"]["back"]),
            "extra": {"truth": bg_truth, "x": flux_truth, "bias_scale": 100.0},
        },
        "Gamma": {
            "best_fit": fitted["gamma"],
            "uncertainty": fitted["std_errors"]["gamma"],
            "extra": {"truth": cfg.gamma, "bias_scale": 100.0},
        },
        "Alpha": {
            "best_fit": fitted["alpha"],
            "uncertainty": fitted["std_errors"]["alpha"],
            "extra": {"truth": cfg.alpha, "bias_scale": 100.0},
        },
    }


class MonteCarlo:
    """Run a photometry pipeline over multiple independent realizations.

    Each realization draws a new random catalog.

    Parameters
    ----------
    config : SimulationConfig
        Simulation parameters.
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
        pipeline=None,
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

    def run(self, verbose: bool = True, show_progress: bool = True) -> list[dict]:
        """Execute all realizations.

        Parameters
        ----------
        verbose : bool
            Print progress every 10 realizations (legacy, use *show_progress*
            instead).
        show_progress : bool
            If ``True``, display a progress bar.

        Returns
        -------
        list of dict
            One dict per successful realization. Each dict contains the
            simulation params and per-estimator results.
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
            except Exception as exc:
                warnings.warn(f"Realization {i + 1} failed: {exc}")
            finally:
                pbar.update(1)

        pbar.close()
        return self._results


def _estimator_names(results: list[dict]) -> list[str]:
    """Return estimator keys from the first result, excluding ``"params"``."""
    return [k for k in results[0] if k != "params"]


def _is_scalar(val) -> bool:
    """Return True for scalar or 0‑d array values."""
    return np.ndim(val) == 0


def compute_bias_coverage(
    results: list[dict],
    *,
    estimators: list[str] | None = None,
    nbins: int = 10,
    bins: np.ndarray | None = None,
    sigma_levels: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, dict]:
    """Compute binned bias, RMS and coverage for one or more estimators.

    Parameters
    ----------
    results : list of dict
        Per-realization outputs from :class:`MonteCarlo`.
    estimators : list of str, optional
        Estimator names to process.  Defaults to all per-source (non-scalar)
        estimators in *results*.
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

    if estimators is None:
        all_names = _estimator_names(results)
        estimators = [n for n in all_names if not _is_scalar(results[0][n]["best_fit"])]

    for label in estimators:
        xs, truths, ests, stds = [], [], [], []
        bias_scale = None

        for res in results:
            entry = res[label]
            x = np.asarray(entry["extra"].get("x", entry["extra"]["truth"]))
            t = np.asarray(entry["extra"]["truth"])
            e = np.asarray(entry["best_fit"])
            s = entry.get("uncertainty")
            bs = entry["extra"].get("bias_scale", 100.0)

            if bias_scale is None:
                bias_scale = bs

            mask = np.isfinite(x) & np.isfinite(t) & np.isfinite(e)
            xs.append(x[mask])
            truths.append(t[mask])
            ests.append(e[mask])
            if s is not None:
                s = np.asarray(s)
                stds.append(s[mask])

        if len(xs) == 0:
            warnings.warn(f"No data for estimator '{label}', skipping.")
            continue

        x_all = np.concatenate(xs)
        truth_all = np.concatenate(truths)
        est_all = np.concatenate(ests)
        se_all = np.concatenate(stds) if stds else None

        bias = (est_all / truth_all - 1.0) * bias_scale

        if bins is None:
            from gcphotom.stats import _build_bins

            valid = np.isfinite(x_all) & np.isfinite(bias)
            edges = _build_bins(x_all[valid], nbins, logbins=True)
        else:
            edges = bins

        xbinned, bias_binned, bias_err = bin_statistic(
            x_all,
            bias,
            nbins=nbins,
            bins=edges,
            logbins=bins is None,
            method="median",
            scale_err=True,
        )

        _, rms_binned, _ = bin_statistic(
            x_all,
            np.abs(bias),
            nbins=nbins,
            bins=edges,
            logbins=bins is None,
            method="median",
            scale_err=True,
        )

        entry = {
            "xbins": xbinned,
            "bias": bias_binned,
            "bias_err": bias_err,
            "rms": rms_binned,
        }

        if se_all is not None:
            for sigma in sigma_levels:
                resids = np.abs(est_all - truth_all)
                covered = (resids <= sigma * se_all).astype(float)
                _, cov_binned, _ = bin_statistic(
                    x_all,
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

    Parameters
    ----------
    stats : dict
        Output from :func:`compute_bias_coverage`.
    ax_bias, ax_coverage : Axes, optional
        Target axes. Created if None.
    sigma_level : float
        Which sigma level's coverage to plot.
    expected_coverage : float, optional
        Expected coverage fraction at *sigma_level* (e.g. 0.683 for 1-sigma).
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
    estimators: list[str] | None = None,
) -> dict[str, dict]:
    """Compute bias statistics for scalar nuisance parameters across realizations.

    Parameters
    ----------
    results : list of dict
        Per-realization outputs from :class:`MonteCarlo`.
    estimators : list of str, optional
        Estimator names to process.  Defaults to all scalar estimators in
        *results*.

    Returns
    -------
    dict
        Mapping from label to dict with keys ``truth``, ``estimates``,
        ``bias``, ``mean_bias``, ``std_bias``, ``rms_bias``.
    """
    if estimators is None:
        all_names = _estimator_names(results)
        estimators = [n for n in all_names if _is_scalar(results[0][n]["best_fit"])]

    out = {}
    for label in estimators:
        truth = None
        bias_scale = 100.0
        estimates = []

        for res in results:
            entry = res[label]
            if truth is None:
                truth = entry["extra"]["truth"]
                bias_scale = entry["extra"].get("bias_scale", 100.0)
            estimates.append(float(np.nanmean(np.asarray(entry["best_fit"]))))

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
        Output from :func:`compute_bias_coverage` for a background estimator.
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


# ---------------------------------------------------------------------------
# Save / load raw MC results (pickle)
# ---------------------------------------------------------------------------


def save_results(path: str, results: list[dict]) -> str:
    """Save MC results to a pickle file.

    Parameters
    ----------
    results : list of dict
        Per-realization outputs from :class:`MonteCarlo`.
    path : str
        Output file path (``.pkl`` appended if missing).

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
        Per-realization outputs, one dict per successful realization.
    """
    if not os.path.exists(path):
        candidate = path + ".pkl"
        if os.path.exists(candidate):
            path = candidate
        else:
            raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Save / load computed stats (pickle, for plot reuse)
# ---------------------------------------------------------------------------


def save_stats(path: str, stats: dict[str, dict]) -> str:
    """Save computed stats to a pickle file for later plotting.

    Accepts the output of :func:`compute_bias_coverage` or
    :func:`compute_nuisance_stats`.

    Parameters
    ----------
    stats : dict
        Stats dict from :func:`compute_bias_coverage` or
        :func:`compute_nuisance_stats`.
    path : str
        Output file path (``.pkl`` appended if missing).

    Returns
    -------
    str
        The actual path written to.
    """
    if not path.endswith(".pkl"):
        path += ".pkl"
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    return path


def load_stats(path: str) -> dict[str, dict]:
    """Load stats saved with :func:`save_stats`.

    Parameters
    ----------
    path : str
        Path to ``.pkl`` file.

    Returns
    -------
    dict
        Stats dict suitable for :func:`plot_bias_coverage`,
        :func:`plot_background_bias`, or :func:`plot_nuisance_summary`.
    """
    if not os.path.exists(path):
        candidate = path + ".pkl"
        if os.path.exists(candidate):
            path = candidate
        else:
            raise FileNotFoundError(f"Stats file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)
