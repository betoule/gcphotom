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

    def __init__(self, gc_result, object_model=None):
        if object_model is None:
            object_model = moffat_object_flux

        self.radii = jnp.array(gc_result["radius"])
        self.areas = annular_fluxes(self.radii**2 * jnp.pi)
        self.object_model = object_model
        self.estimate = None
        self._loss_fn = None
        n = len(gc_result["flux"])
        self._orig_n = n
        self.kept = np.ones(n, dtype=bool)
        self._set_data(gc_result)
        self._cut()

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

    def _cut(self):
        """Remove sources with fewer than 2 good data points."""
        valid = np.asarray(self.goods.sum(axis=0) > 1)
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
        self, initial_guess=None, niter=10000, learning_rate=5e-3, show=False, loss=None
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
            self._loss_fn = jaxfitter.tukey(c=4.685)
        else:
            self._loss_fn = loss

        loss_fn = jax.jit(lambda p: jnp.mean(self._loss_fn(self.weighted_residuals(p))))
        bf, extra = jaxfitter.fit_adam(
            loss_fn, initial_guess, niter=niter, learning_rate=learning_rate, tol=None
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
        self.background_estimate = self.fluxes[-1, :] / (
            jnp.pi * (self.radii[-1] ** 2 - self.radii[-2] ** 2)
        )
        gamma_est = 3.0

        return {
            "gamma": gamma_est,
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

    def _compute_uncertainty(self, bf, robust=False):
        """Parameter covariance and standard errors at the best-fit point.

        Uses the standard non-linear least-squares formula

            Cov = (J^T W J)^{-1}

        where J is the Jacobian of the *unweighted* annular model and
        W = diag(1/σ²) are the inverse-variance weights from the noise model.
        The covariance is optionally scaled by the reduced χ².

        When ``robust=True`` and the loss function used during :meth:`fit`
        is available, a sandwich estimator is used instead to account for
        the robust loss:

            Cov = B^{-1} · M · B^{-1}

        where B = J^T diag(w · ψ′) J, M = J^T diag(w · ψ²) J, and ψ is the
        influence function of the loss.

        Parameters
        ----------
        bf : pytree
            Best-fit parameters (same structure as returned by :meth:`fit`).
        robust : bool, optional
            If ``True``, apply the sandwich correction using the stored loss
            function.  Default ``False``.

        Returns
        -------
        cov : (n_params, n_params) ndarray
            Covariance matrix in flattened-parameter space.
        se : pytree
            Standard errors in a pytree matching the structure of *bf*.
        """
        from jax.flatten_util import ravel_pytree

        p0, unravel = ravel_pytree(bf)
        n_params = p0.size
        goods_mask = self.goods
        n_good = int(goods_mask.sum())

        if n_good <= n_params:
            se = jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), bf)
            return jnp.full((n_params, n_params), jnp.nan), se

        def flat_model(pf):
            p = unravel(pf)
            m = annular_fluxes(self.model(p))
            return m[goods_mask].ravel()

        J = jax.jacfwd(flat_model)(p0)

        scaled = self._flux(bf)
        object_ann = annular_fluxes(self.object_model(scaled, self.radii))
        noise = jnp.maximum(object_ann + self.bkg_var, 1e-30)
        w = (1.0 / noise)[goods_mask].ravel()

        JTWJ = (J * w[:, None]).T @ J
        cov = jnp.linalg.inv(JTWJ)

        wr = self.weighted_residuals(bf)[goods_mask].ravel()
        chi2 = jnp.sum(wr**2)
        dof = n_good - n_params
        cov = jnp.where(dof > 0, cov * (chi2 / dof), cov)

        if robust and self._loss_fn is not None:
            from .jaxfitter import loss_derivatives

            psi, psi_deriv = loss_derivatives(self._loss_fn)
            r = self.weighted_residuals(bf)[goods_mask].ravel()
            Jw = J * w[:, None]
            bread = Jw.T @ (psi_deriv(r)[:, None] * Jw)
            meat = Jw.T @ ((psi(r) ** 2)[:, None] * Jw)
            cov = jnp.linalg.inv(bread) @ meat @ jnp.linalg.inv(bread)

        se_flat = jnp.sqrt(jnp.diag(cov))
        leaves, treedef = jax.tree_util.tree_flatten(bf)
        se_leaves = []
        start = 0
        for leaf in leaves:
            size = leaf.size
            se_leaves.append(se_flat[start : start + size].reshape(leaf.shape))
            start += size
        se = jax.tree_util.tree_unflatten(treedef, se_leaves)

        return cov, se

    def results(self, bf, compute_errors="diag", robust=False):
        """Extract fitted results as a dictionary.

        Parameters
        ----------
        bf : dict
            Best-fit parameters.
        compute_errors : bool or str, optional
            Controls uncertainty output:

            - ``False`` or ``"none"`` — no uncertainty.
            - ``"diag"`` (default) — include ``std_errors`` pytree.
            - ``"full"`` — include ``std_errors``, ``covariance``, and
              ``correlation``.
        robust : bool, optional
            If ``True``, use the sandwich estimator (requires the loss
            function stored by :meth:`fit`).  Default ``False``.

        Returns
        -------
        dict with keys ``flux``, ``back``, ``gamma``, ``alpha``, ``ngoods``,
        ``chi2`` and, if *compute_errors* is set, ``std_errors``
        (and optionally ``covariance``, ``correlation``).
        Per-source arrays (flux, back, ngoods, chi2) have length equal to the
        original number of sources passed to extract_growth_curves. Dropped
        sources (via internal cuts or detect_contamination) are represented
        by NaN.
        """
        n = getattr(self, "_orig_n", 0)
        n_cur = (
            int(self.fluxes.shape[1])
            if getattr(self, "fluxes", None) is not None
            else 0
        )
        kept = np.asarray(self.kept)

        if n_cur == 0:
            full = lambda: np.full(n, np.nan)
            g = (
                float(bf.get("gamma", np.nan))
                if isinstance(bf, dict)
                else float(getattr(bf, "gamma", np.nan))
            )
            a = (
                float(bf.get("alpha", np.nan))
                if isinstance(bf, dict)
                else float(getattr(bf, "alpha", np.nan))
            )
            return {
                "flux": full(),
                "back": full(),
                "gamma": g,
                "alpha": a,
                "ngoods": full(),
                "chi2": full(),
            }

        # safe unit vectors matching current kept count
        unit = np.asarray(
            bf.get("flux", np.ones(n_cur)) if isinstance(bf, dict) else np.ones(n_cur),
            dtype=float,
        )
        if len(unit) != n_cur:
            unit = np.ones(n_cur)
        bck = np.asarray(
            (
                bf.get("back", np.zeros(n_cur))
                if isinstance(bf, dict)
                else np.zeros(n_cur)
            ),
            dtype=float,
        )
        if len(bck) != n_cur:
            bck = np.zeros(n_cur)

        est = np.asarray(self.estimate) if self.estimate is not None else np.ones(n_cur)
        if len(est) != n_cur:
            est = np.ones(n_cur)

        red_flux = unit * est
        red_back = bck

        # compute ngoods/chi2 using current internal state + consistent tmp bf
        tmp_bf = {
            "flux": unit,
            "back": bck,
            "gamma": (
                bf.get("gamma", 3.0)
                if isinstance(bf, dict)
                else getattr(bf, "gamma", 3.0)
            ),
            "alpha": (
                bf.get("alpha", 3.0)
                if isinstance(bf, dict)
                else getattr(bf, "alpha", 3.0)
            ),
        }
        ngoods = self.goods.sum(axis=0)
        wr = self.weighted_residuals(tmp_bf, mask=True)
        chi2 = np.nansum(wr**2, axis=0)

        full_flux = np.full(n, np.nan)
        full_back = np.full(n, np.nan)
        full_ng = np.full(n, np.nan)
        full_chi = np.full(n, np.nan)
        if np.any(kept):
            full_flux[kept] = red_flux
            full_back[kept] = red_back
            full_ng[kept] = ngoods
            full_chi[kept] = chi2

        g = (
            float(bf.get("gamma", np.nan))
            if isinstance(bf, dict)
            else float(getattr(bf, "gamma", np.nan))
        )
        a = (
            float(bf.get("alpha", np.nan))
            if isinstance(bf, dict)
            else float(getattr(bf, "alpha", np.nan))
        )
        result = {
            "flux": full_flux,
            "back": full_back,
            "gamma": g,
            "alpha": a,
            "ngoods": full_ng,
            "chi2": full_chi,
        }

        if compute_errors and compute_errors != "none":
            cov, se_pytree = self._compute_uncertainty(tmp_bf, robust=robust)
            result["std_errors"] = se_pytree
            if compute_errors == "full":
                se_flat = np.sqrt(np.diag(np.asarray(cov)))
                corr = np.asarray(cov) / np.outer(se_flat, se_flat)
                result["covariance"] = np.asarray(cov)
                result["correlation"] = corr

        return result
