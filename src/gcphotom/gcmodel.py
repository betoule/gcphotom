"""Growth curve model and fitter for Moffat profile photometry."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from . import jaxfitter


def fwhm2gamma(fwhm, alpha):
    """Convert FWHM to Moffat gamma parameter."""
    return fwhm / (2.0 * jnp.sqrt(2.0 ** (1.0 / alpha) - 1.0))


def gamma2fwhm(gamma, alpha):
    """Convert Moffat gamma parameter to FWHM."""
    return 2.0 * gamma * jnp.sqrt(2.0 ** (1.0 / alpha) - 1.0)


def sigma2fwhm(sigma):
    """Convert Gaussian sigma to FWHM."""
    return sigma * jnp.sqrt(8 * jnp.log(2))


def fwhm2sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / jnp.sqrt(8 * jnp.log(2))


def moffat(r2, gamma, alpha):
    """Moffat profile value at squared radius r2."""
    return (alpha - 1) / (jnp.pi * gamma**2) * (1 + r2 / gamma**2) ** (-alpha)


def moffat_flux(R, gamma, alpha):
    """Integrated Moffat flux up to radius R (normalized to 1 at infinity)."""
    return 1 - (1 + R**2 / gamma**2) ** (1 - alpha)


def imoffat(x, gamma, alpha):
    """Inverse Moffat: radius at which cumulative flux fraction equals x."""
    return jnp.sqrt((x * jnp.pi * gamma**2 / (alpha - 1)) ** (-1 / alpha) - 1) * gamma


def annular_fluxes(x):
    """Compute annular fluxes: diff of cumulative flux along axis 0."""
    return jnp.diff(x, prepend=0, axis=0)


def moffat_model(params, radii):
    """Growth curve model: Moffat profile + linear background.

    Parameters
    ----------
    params : dict
        Keys: ``flux``, ``gamma``, ``alpha``, ``back``.
    radii : array_like
        Aperture radii.

    Returns
    -------
    model_flux : 2D array (n_radii, n_sources)
        Cumulative model flux at each radius.
    """
    flux = params["flux"]
    gamma = params["gamma"]
    alpha = params["alpha"]
    back = params["back"]
    return (
        flux[None, :] * moffat_flux(radii[:, None], gamma, alpha)
        + back[None, :] * radii[:, None] ** 2 * jnp.pi
    )


class Fitter:
    """Fit growth curves with a common Moffat profile.

    Parameters
    ----------
    gc_result : dict
        Output of :func:`gcphotom.aperture.extract_growth_curves`, which always
        contains ``radius``, ``flux``, ``flux_err``, ``flux_clean``, and
        ``contamination``. The fitter uses ``flux_clean`` for the data to fit.
    model : callable, optional
        Model function ``f(params, radii) -> cumulative_flux``.
        Defaults to :func:`moffat_model`.

    Attributes
    ----------
    kept : ndarray of bool
        Boolean mask of length equal to the number of sources passed to
        ``extract_growth_curves`` (i.e., original input length). ``True`` for
        sources that survived all cuts up to the last ``results`` or
        ``detect_contamination`` call. Use to align fitted results back to
        the input order when sources have been dropped.
    """

    def __init__(self, gc_result, model=None):
        if model is None:
            model = moffat_model

        self.radii = jnp.array(gc_result["radius"])
        self.areas = annular_fluxes(self.radii**2 * jnp.pi)
        self.model = model
        self.estimate = None
        n = len(gc_result["flux"])
        self.kept = np.ones(n, dtype=bool)
        self._set_data(gc_result)
        self._cut()

    def _set_data(self, gc_result):
        """Extract annular fluxes and variances from cumulative growth curves.

        gc_result["flux_clean"] has shape (n_sources, n_radii); we transpose to
        (n_radii, n_sources) to match the model convention.
        """
        cum_flux = jnp.array(gc_result["flux_clean"]).T
        self.fluxes = annular_fluxes(cum_flux)
        var_cum = jnp.array(gc_result["flux_err"]).T ** 2
        self.var = annular_fluxes(var_cum)
        self.var = jnp.clip(self.var, 1e-30, None)
        self.goods = jnp.isfinite(self.fluxes) & jnp.isfinite(self.var)

    def _cut(self):
        """Remove sources with fewer than 2 good data points."""
        valid = np.asarray(self.goods.sum(axis=0) > 1)
        self.kept = np.asarray(self.kept)
        self.kept[self.kept] = valid
        if getattr(self, "estimate", None) is not None:
            self.estimate = np.asarray(self.estimate)[valid]
        self.fluxes = self.fluxes[:, valid]
        self.var = self.var[:, valid]
        self.goods = self.goods[:, valid]

    def _flux(self, params):
        """Scale flux by the estimate factor from initial guess."""
        return {**params, "flux": params["flux"] * self.estimate}

    def residuals(self, params, mask=False):
        """Annular residuals: data - model."""
        m = self.model(self._flux(params), self.radii)
        r = self.fluxes - annular_fluxes(m)
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        return r

    def weighted_residuals(self, params, mask=False):
        """Weighted annular residuals."""
        m = annular_fluxes(self.model(self._flux(params), self.radii))
        residuals = self.fluxes - m
        noise = jnp.maximum(m, 1e-30)
        r = residuals / jnp.sqrt(noise) * self.goods
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        return r

    def chi2(self, params):
        """Mean squared weighted residual."""
        return jnp.mean(self.weighted_residuals(params) ** 2)

    def fit(self, initial_guess=None, niter=10000, learning_rate=5e-3, show=False):
        """Fit using Adam optimizer.

        Parameters
        ----------
        initial_guess : dict or None
            Initial parameter values.  If ``None``, :meth:`initial_guess` is used.
        niter : int
            Maximum optimizer iterations.
        learning_rate : float
            Adam learning rate.
        show : bool
            If ``True``, plot the loss curve.

        Returns
        -------
        best_params : dict
            Optimized parameters.
        extra : dict
            Contains ``loss`` and ``timings`` arrays.
        """
        if self.fluxes.shape[1] == 0:
            raise ValueError("No sources remaining with sufficient data points to fit.")
        if initial_guess is None:
            initial_guess = self.initial_guess()

        chi2_fn = jax.jit(self.chi2)
        bf, extra = jaxfitter.fit_adam(
            chi2_fn, initial_guess, niter=niter, learning_rate=learning_rate, tol=None
        )
        if show:
            plt.plot(extra["loss"])
        return bf, extra

    def initial_guess(self, alpha=3.0):
        """Heuristic initial parameter guess.

                Estimates total flux from inner aperture scaling, and gamma from
        the 50%-flux radius of each growth curve.
        """
        nsrc = self.fluxes.shape[1]
        if nsrc == 0:
            self.estimate = jnp.array([], dtype=float)
            return {
                "gamma": 3.0,
                "alpha": alpha,
                "flux": jnp.array([]),
                "back": jnp.array([]),
            }
        n_radii = self.fluxes.shape[0]
        a1 = max(2, n_radii // 5)
        a2 = max(5, n_radii // 2)
        f_inner = self.fluxes[:a1, :].sum(axis=0)
        f_outer = self.fluxes[:a2, :].sum(axis=0)
        ac = float(jnp.nanmedian(f_outer / f_inner))
        estimate = f_inner * ac
        self.estimate = estimate
        self.background_estimate = (self.fluxes[-1, :] - self.fluxes[-2, :]) / (
            jnp.pi * (self.radii[-1] ** 2 - self.radii[-2] ** 2)
        )
        gamma_est = 3.0  # self._estimate_gamma(estimate * ac, alpha)

        return {
            "gamma": gamma_est,
            "alpha": alpha,
            "flux": jnp.ones(nsrc),
            "back": self.background_estimate,
        }

    def _estimate_gamma(self, total_flux, alpha):
        """Estimate gamma from the 50%-flux radius of each source."""
        cum_flux = self.fluxes.sum(axis=0)
        half_flux = total_flux * 0.5
        median_radii = []

        for i in range(cum_flux.shape[0]):
            target = half_flux[i]
            above = cum_flux >= target
            if jnp.any(above):
                idx = int(jnp.argmax(above))
                if idx > 0:
                    r_lo = float(self.radii[idx - 1])
                    r_hi = float(self.radii[idx])
                    f_lo = float(cum_flux[idx - 1])
                    f_hi = float(cum_flux[idx])
                    if f_hi != f_lo:
                        frac = (target - f_lo) / (f_hi - f_lo)
                        median_radii.append(r_lo + frac * (r_hi - r_lo))
                    else:
                        median_radii.append(r_hi)
                else:
                    median_radii.append(float(self.radii[0]))
            else:
                median_radii.append(float(self.radii[-1]))

        median_r = jnp.array(median_radii)
        median_r = median_r[median_r > 0]
        if len(median_r) == 0:
            median_r = jnp.array([2.0])

        return float(fwhm2gamma(sigma2fwhm(jnp.median(median_r)), alpha))

    def plot_PSF(self, bf, axes=None):
        """Plot the median PSF and residuals.

        Parameters
        ----------
        bf : dict
            Best-fit parameters.
        axes : tuple of Axes or None
            ``(ax1, ax2)`` for PSF and residual plots.

        Returns
        -------
        ax1, ax2 : matplotlib Axes
        """
        if axes is None:
            fig = plt.figure("PSF residuals")
            ax1, ax2 = fig.subplots(2, 1, sharex=True)
        else:
            ax1, ax2 = axes

        PSF = (self.fluxes - bf["back"] * self.areas[:, None]) / (
            bf["flux"] * self.estimate
        )
        PSF = PSF.at[~self.goods].set(jnp.nan)

        r = np.array(self.residuals(bf, mask=True) / (bf["flux"] * self.estimate))

        ax1.plot(
            self.radii,
            annular_fluxes(
                self.model(
                    {**bf, "flux": np.array([1.0]), "back": np.array([0.0])},
                    self.radii,
                )
            ),
            "k-",
        )
        ax1.plot(self.radii, np.nanmedian(PSF, axis=1), "o")
        ax2.plot(self.radii, np.nanmean(r, axis=1), "o")
        return ax1, ax2

    def detect_contamination(self, bf):
        """Reject data points with weighted residuals > 5."""
        wr = self.weighted_residuals(bf)
        self.goods = self.goods & (jnp.abs(wr) < 5)
        self._cut()

    def results(self, bf):
        """Extract fitted results as a dictionary.

        Parameters
        ----------
        bf : dict
            Best-fit parameters.

        Returns
        -------
        dict with keys ``flux``, ``back``, ``gamma``, ``alpha``, ``ngoods``, ``chi2``.
        """
        par = self._flux(bf)
        ngoods = self.goods.sum(axis=0)
        wr = self.weighted_residuals(bf, mask=True)
        chi2 = np.nansum(wr**2, axis=0)
        return {
            "flux": np.array(par["flux"]),
            "back": np.array(par["back"]),
            "gamma": float(bf["gamma"]),
            "alpha": float(bf["alpha"]),
            "ngoods": np.array(ngoods),
            "chi2": np.array(chi2),
        }
