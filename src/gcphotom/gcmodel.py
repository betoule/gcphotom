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


def moffat_object_flux(params, radii):
    """Cumulative object flux (Moffat profile, no background).

    Parameters
    ----------
    params : dict
        Keys: ``flux``, ``gamma``, ``alpha``.
    radii : array_like
        Aperture radii.

    Returns
    -------
    cum_flux : 2D array (n_radii, n_sources)
        Cumulative object flux at each radius.
    """
    flux = params["flux"]
    gamma = params["gamma"]
    alpha = params["alpha"]
    return flux[None, :] * moffat_flux(radii[:, None], gamma, alpha)


class Fitter:
    """Fit growth curves with a common Moffat profile.

    Parameters
    ----------
    gc_result : dict
        Output of :func:`gcphotom.aperture.extract_growth_curves`, which always
        contains ``radius``, ``flux``, ``background_var``, ``flux_clean``, and
        ``contamination``. The fitter uses ``flux_clean`` for the data to fit.
    object_model : callable, optional
        Function ``f(params, radii) -> cumulative_object_flux`` that returns
        the cumulative flux of the object alone (no background contribution).
        Defaults to :func:`moffat_object_flux`.

    Attributes
    ----------
    kept : ndarray of bool
        Boolean mask of length equal to the number of sources passed to
        ``extract_growth_curves`` (i.e., original input length). ``True`` for
        sources that survived all cuts up to the last ``results`` or
        ``detect_contamination`` call. Use to align fitted results back to
        the input order when sources have been dropped.
    """

    def __init__(self, gc_result, object_model=None, bads=None):
        if object_model is None:
            object_model = moffat_object_flux

        self.radii = jnp.array(gc_result["radius"])
        self.areas = annular_fluxes(self.radii**2 * jnp.pi)
        self.object_model = object_model
        self.estimate = None
        n = len(gc_result["flux"])
        self._orig_n = n
        self.kept = np.ones(n, dtype=bool)
        self._set_data(gc_result)
        self._cut(bads)

    def _set_data(self, gc_result):
        """Extract annular fluxes and background variances from cumulative growth curves.

        gc_result["flux_clean"] has shape (n_sources, n_radii); we transpose to
        (n_radii, n_sources) to match the model convention.
        """
        cum_flux = jnp.array(gc_result["flux_clean"]).T
        self.fluxes = annular_fluxes(cum_flux)
        var_cum = jnp.array(gc_result["background_var"]).T
        self.bkg_var = annular_fluxes(var_cum)
        self.bkg_var = jnp.clip(self.bkg_var, 1e-30, None)
        self.goods = jnp.isfinite(self.fluxes) & jnp.isfinite(self.bkg_var)

    def _cut(self, bads=None):
        """Remove sources with fewer than 2 good data points, and close contaminants."""
        valid = np.asarray(self.goods.sum(axis=0) > 1) & self.goods[:3, :].any()
        if bads is not None:
            valid &= ~bads
        self.kept = np.asarray(self.kept)
        self.kept[self.kept] = valid
        if getattr(self, "estimate", None) is not None:
            self.estimate = np.asarray(self.estimate)[valid]
        self.fluxes = self.fluxes[:, valid]
        self.bkg_var = self.bkg_var[:, valid]
        self.goods = self.goods[:, valid]

    def _flux(self, params):
        """Scale flux by the estimate factor from initial guess."""
        return {**params, "flux": params["flux"] * self.estimate}

    def _background_cumulative(self, back):
        """Cumulative background flux at each radius: back * pi * r^2."""
        return back[None, :] * self.radii[:, None] ** 2 * jnp.pi

    def model(self, params):
        """Full cumulative model including background.

        Parameters
        ----------
        params : dict
            Parameters with keys ``flux``, ``gamma``, ``alpha``, ``back``.

        Returns
        -------
        cum_flux : 2D array (n_radii, n_sources)
            Cumulative model flux at each radius.
        """
        scaled = self._flux(params)
        object_cum = self.object_model(scaled, self.radii)
        return object_cum + self._background_cumulative(scaled["back"])

    def residuals(self, params, mask=False):
        """Annular residuals: data - model."""
        m = annular_fluxes(self.model(params))
        r = self.fluxes - m
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        return r

    def weighted_residuals(self, params, mask=False):
        """Weighted annular residuals.

        The noise estimate uses only the object photon noise (Poisson
        variance approximated by the object-only annular flux), combined
        with the a priori background variance. The background contribution
        is added back for the residual itself.
        """
        scaled = self._flux(params)
        object_cum = self.object_model(scaled, self.radii)
        object_ann = annular_fluxes(object_cum)
        back_ann = annular_fluxes(self._background_cumulative(scaled["back"]))
        total_ann = object_ann + back_ann
        residuals = self.fluxes - total_ann
        noise = jnp.maximum(object_ann + self.bkg_var, 1e-30)
        r = residuals / jnp.sqrt(noise) * self.goods
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        return r

    def chi2(self, params):
        """Mean squared weighted residual."""
        return jnp.mean(self.weighted_residuals(params) ** 2)

    def fit(
        self,
        initial_guess=None,
        niter=10000,
        learning_rate=5e-3,
        show=False,
        show_progress=True,
        desc=None,
        loss=None,
        fix=None,
        compute_uncertainty=False,
    ):
        """Fit using Adam optimizer with a robust M-estimator loss.

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
        show_progress : bool
            If ``True``, display a progress bar during optimization.
        desc : str or None
            Label for the progress bar.  If ``None``, a default is used.
        loss : callable or None
            Loss function ``f(weighted_residual) -> per-element loss``.
            Defaults to Tukey's bisquare with ``c=4.685``
            (``gcphotom.tukey()``).

            Common choices::

                fit(loss=gcp.tukey(c=4.685))        — Tukey bisquare (default)
                fit(loss=gcp.pseudo_huber(c=2.0))    — pseudo-Huber
                fit(loss=gcp.cauchy(c=2.0))          — Cauchy
                fit(loss=lambda x: x**2)             — standard chi2
                fit(loss=lambda x: jnp.abs(x))       — L1 loss

        compute_uncertainty : bool
            If ``True``, compute parameter covariance and standard
            errors via the Jacobian of weighted residuals (adds ~1.5 s to
            the fit).  ``False`` by default — set to ``True`` when
            uncertainties are needed.  When ``False``, ``extra`` will
            contain ``None`` for both ``covariance`` and ``std_errors``.

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

        if loss is None:
            loss_fn_ = jaxfitter.pseudo_huber(c=2.0)
        else:
            loss_fn_ = loss

        if fix is None:
            wr = self.weighted_residuals
        else:
            for p in fix:
                initial_guess.pop(p)
            wr = lambda p: self.weighted_residuals({**p, **fix})

        # JIT-compile the residual function so that jacfwd in
        # parameter_uncertainty traces through compiled jaxprs instead of
        # compiling each operation eagerly — cuts ~2.5s off the first fit.
        wr_jit = jax.jit(wr)

        loss_fn = lambda p: jnp.mean(loss_fn_(wr_jit(p)))
        bf, extra = jaxfitter.fit_adam(
            loss_fn,
            initial_guess,
            niter=niter,
            learning_rate=learning_rate,
            tol=None,
            show_progress=show_progress,
            desc=desc or "Fitting",
        )
        if show:
            plt.plot(extra["loss"])

        if compute_uncertainty:
            extra["covariance"], extra["std_errors"] = jaxfitter.parameter_uncertainty(
                lambda p: wr_jit(p)[self.goods], bf
            )
            self._std_errors = extra["std_errors"]
        else:
            extra["covariance"] = extra["std_errors"] = None

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

        # Use numpy for these one-off setup computations to avoid per-op
        # JAX compilation; the initial-guess dict is converted to JAX
        # arrays automatically when traced inside the JIT-compiled step.
        n_radii = self.fluxes.shape[0]
        fluxes_np = np.asarray(self.fluxes)
        radii_np = np.asarray(self.radii)
        a1 = max(2, n_radii // 5)
        a2 = max(5, n_radii // 2)
        f_inner = fluxes_np[:a1, :].sum(axis=0)
        f_outer = fluxes_np[:a2, :].sum(axis=0)
        ac = float(np.nanmedian(f_outer / f_inner))
        estimate = f_inner * ac
        self.estimate = jnp.array(estimate)
        self.background_estimate = jnp.array(
            fluxes_np[-1, :] / (np.pi * (radii_np[-1] ** 2 - radii_np[-2] ** 2))
        )

        return {
            "gamma": 3.0,
            "alpha": alpha,
            "flux": jnp.ones(nsrc),
            "back": self.background_estimate,
        }

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
                self.object_model(
                    {**bf, "flux": np.array([1.0])},
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

    def rescale_params(self, bf):
        """Rescale normalized fit parameters to physical units.

        During fitting, the flux parameter is normalized by
        ``self.estimate`` to keep all parameters near unity for numerical
        stability. This method reverses that scaling and propagates
        uncertainties.

        Parameters
        ----------
        bf : dict
            Best-fit parameters (returned by :meth:`fit`).

        Returns
        -------
        dict with keys ``flux``, ``back``, ``gamma``, ``alpha``, and
        ``std_errors`` (when :meth:`fit` has been called). Per-source
        arrays have the current (possibly truncated) length.
        """
        n_cur = (
            int(self.fluxes.shape[1])
            if getattr(self, "fluxes", None) is not None
            else 0
        )

        if n_cur == 0:
            result = {
                "flux": np.array([]),
                "back": np.array([]),
                "gamma": float(bf["gamma"]),
                "alpha": float(bf["alpha"]),
            }
            if hasattr(self, "_std_errors") and self._std_errors is not None:
                result["std_errors"] = self._std_errors
            return result

        est = (
            np.asarray(self.estimate, dtype=float)
            if self.estimate is not None
            else np.ones(n_cur)
        )

        flux = np.asarray(bf["flux"], dtype=float)
        back = np.asarray(bf.get("back", np.zeros(n_cur)), dtype=float)
        gamma = float(bf["gamma"])
        alpha = float(bf["alpha"])

        result = {
            "flux": flux * est,
            "back": back,
            "gamma": gamma,
            "alpha": alpha,
        }

        if hasattr(self, "_std_errors") and self._std_errors is not None:
            se = self._std_errors
            result["std_errors"] = {
                k: (se[k] * est if k == "flux" else se[k]) for k in se
            }

        return result

    def expand_to_original(self, arr):
        """Expand a per-source array to the original input length.

        Sources dropped by :meth:`_cut` or :meth:`detect_contamination`
        are filled with NaN.

        Parameters
        ----------
        arr : ndarray
            Per-source array of current length.

        Returns
        -------
        ndarray of length ``_orig_n`` with NaN for dropped sources.
        """
        n = getattr(self, "_orig_n", 0)
        kept = np.asarray(self.kept)
        full = np.full(n, np.nan, dtype=float)
        if np.any(kept):
            full[kept] = np.asarray(arr, dtype=float)
        return full

    def goodness(self, bf):
        """Compute per-source ngoods and chi2.

        Parameters
        ----------
        bf : dict
            Best-fit parameters with normalized flux.

        Returns
        -------
        dict with keys ``ngoods`` and ``chi2``.
        """
        n_cur = (
            int(self.fluxes.shape[1])
            if getattr(self, "fluxes", None) is not None
            else 0
        )

        if n_cur == 0:
            return {"ngoods": np.array([]), "chi2": np.array([])}

        unit = np.asarray(bf["flux"], dtype=float)
        bck = np.asarray(bf.get("back", np.zeros(n_cur)), dtype=float)

        tmp_bf = {
            "flux": unit,
            "back": bck,
            "gamma": float(bf["gamma"]),
            "alpha": float(bf["alpha"]),
        }

        ngoods = np.asarray(self.goods.sum(axis=0))
        wr = self.weighted_residuals(tmp_bf, mask=True)
        chi2 = np.nansum(np.asarray(wr) ** 2, axis=0)

        return {"ngoods": ngoods, "chi2": chi2}

    def results(self, bf):
        """Extract fitted results as a dictionary.

        Convenience method combining :meth:`rescale_params`,
        :meth:`goodness`, and :meth:`expand_to_original`.

        Parameters
        ----------
        bf : dict
            Best-fit parameters (returned by :meth:`fit`).

        Returns
        -------
        dict with keys ``flux``, ``back``, ``gamma``, ``alpha``, ``ngoods``,
        ``chi2``, and ``std_errors`` (when :meth:`fit` has been called).
        Per-source arrays (flux, back, ngoods, chi2) have length equal to the
        original number of sources passed to extract_growth_curves. Dropped
        sources (via internal cuts or detect_contamination) are represented
        by NaN.
        """
        res = self.rescale_params(bf)
        g = self.goodness(bf)

        for k in ("ngoods", "chi2"):
            res[k] = self.expand_to_original(g[k])
        for k in ("flux", "back"):
            res[k] = self.expand_to_original(res[k])

        if "std_errors" in res:
            for k in ("flux", "back"):
                if k in res["std_errors"]:
                    res["std_errors"][k] = self.expand_to_original(res["std_errors"][k])

        return res
