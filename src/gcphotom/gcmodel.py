"""Growth curve model and fitter for Moffat profile photometry."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from . import jaxfitter


def fwhm2alpha(fwhm, beta):
    """Convert FWHM to Moffat alpha parameter."""
    return fwhm / (2.0 * jnp.sqrt(2.0 ** (1.0 / beta) - 1.0))


def sigma2fwhm(sigma):
    """Convert Gaussian sigma to FWHM."""
    return sigma * jnp.sqrt(8 * jnp.log(2))


def fwhm2sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / jnp.sqrt(8 * jnp.log(2))


def moffat(r2, alpha, beta):
    """Moffat profile value at squared radius r2."""
    return (beta - 1) / (jnp.pi * alpha**2) * (1 + r2 / alpha**2) ** (-beta)


def moffat_flux(R, alpha, beta):
    """Integrated Moffat flux up to radius R (normalized to 1 at infinity)."""
    return 1 - (1 + R**2 / alpha**2) ** (1 - beta)


def imoffat(x, alpha, beta):
    """Inverse Moffat: radius at which cumulative flux fraction equals x."""
    return jnp.sqrt((x * jnp.pi * alpha**2 / (beta - 1)) ** (-1 / beta) - 1) * alpha


def flux_and_couronnes(x):
    """Compute annular fluxes: diff of cumulative flux along axis 0."""
    return jnp.diff(x, prepend=0, axis=0)


def moffat_model(params, radii):
    """Growth curve model: Moffat profile + linear background.

    Parameters
    ----------
    params : dict
        Keys: ``flux``, ``alpha``, ``beta``, ``back``.
    radii : array_like
        Aperture radii.

    Returns
    -------
    model_flux : 2D array (n_radii, n_sources)
        Cumulative model flux at each radius.
    """
    flux = params["flux"]
    alpha = params["alpha"]
    beta = params["beta"]
    back = params["back"]
    return (
        flux[None, :] * moffat_flux(radii[:, None], alpha, beta)
        + back[None, :] * radii[:, None] ** 2 * jnp.pi
    )


class Fitter:
    """Fit growth curves with a common Moffat profile.

    Parameters
    ----------
    gc_result : dict
        Output of :func:`gcphotom.aperture.extract_growth_curves`, with keys
        ``radius``, ``flux``, and ``flux_err``.
    model : callable, optional
        Model function ``f(params, radii) -> cumulative_flux``.
        Defaults to :func:`moffat_model`.
    """

    def __init__(self, gc_result, model=None):
        if model is None:
            model = moffat_model

        self.radii = jnp.array(gc_result["radius"])
        self.areas = flux_and_couronnes(self.radii**2 * jnp.pi)
        self.model = model
        self.estimate = None
        self._set_data(gc_result)
        self._cut()

    def _set_data(self, gc_result):
        """Extract annular fluxes and variances from cumulative growth curves.

        gc_result["flux"] has shape (n_sources, n_radii); we transpose to
        (n_radii, n_sources) to match the model convention.
        """
        cum_flux = jnp.array(gc_result["flux"]).T
        self.fluxes = flux_and_couronnes(cum_flux)
        var_cum = jnp.array(gc_result["flux_err"]).T ** 2
        self.var = flux_and_couronnes(var_cum)
        self.var = jnp.clip(self.var, 1e-30, None)
        self.goods = jnp.isfinite(self.fluxes) & jnp.isfinite(self.var)

    def _cut(self):
        """Remove sources with fewer than 2 good data points."""
        valid = self.goods.sum(axis=0) > 1
        self.fluxes = self.fluxes[:, valid]
        self.var = self.var[:, valid]
        self.goods = self.goods[:, valid]

    def _flux(self, params):
        """Scale flux by the estimate factor from initial guess."""
        return {**params, "flux": params["flux"] * self.estimate}

    def residuals(self, params, mask=False):
        """Annular residuals: data - model."""
        m = self.model(self._flux(params), self.radii)
        r = self.fluxes - flux_and_couronnes(m)
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        return r

    def weighted_residuals(self, params, mask=False):
        """Weighted annular residuals."""
        m = flux_and_couronnes(self.model(self._flux(params), self.radii))
        residuals = self.fluxes - m
        noise = m * 0.01
        r = residuals / jnp.sqrt(self.var + noise**2) * self.goods
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
        if initial_guess is None:
            initial_guess = self.initial_guess()

        chi2_fn = jax.jit(self.chi2)
        bf, extra = jaxfitter.fit_adam(
            chi2_fn, initial_guess, niter=niter, learning_rate=learning_rate, tol=None
        )
        if show:
            plt.plot(extra["loss"])
        return bf, extra

    def initial_guess(self, beta=4.0):
        """Heuristic initial parameter guess.

        Estimates total flux from inner aperture scaling, and alpha from
        the 50%-flux radius of each growth curve.
        """
        n_radii = self.fluxes.shape[0]
        a1 = max(2, n_radii // 5)
        a2 = max(5, n_radii // 2)
        f_inner = self.fluxes[:a1, :].sum(axis=0)
        f_outer = self.fluxes[:a2, :].sum(axis=0)
        ac = float(jnp.nanmedian(f_outer / f_inner))
        estimate = f_inner * ac
        self.estimate = estimate

        alpha_est = self._estimate_alpha(estimate * ac, beta)

        return {
            "alpha": alpha_est,
            "beta": beta,
            "flux": jnp.ones(self.fluxes.shape[1]),
            "back": jnp.zeros(self.fluxes.shape[1]),
        }

    def _estimate_alpha(self, total_flux, beta):
        """Estimate alpha from the 50%-flux radius of each source."""
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

        return float(fwhm2alpha(sigma2fwhm(jnp.median(median_r)), beta))

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
            flux_and_couronnes(
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
        dict with keys ``flux``, ``back``, ``alpha``, ``beta``, ``ngoods``, ``chi2``.
        """
        par = self._flux(bf)
        ngoods = self.goods.sum(axis=0)
        wr = self.weighted_residuals(bf, mask=True)
        chi2 = np.nansum(wr**2, axis=0)
        return {
            "flux": np.array(par["flux"]),
            "back": np.array(par["back"]),
            "alpha": float(bf["alpha"]),
            "beta": float(bf["beta"]),
            "ngoods": np.array(ngoods),
            "chi2": np.array(chi2),
        }
