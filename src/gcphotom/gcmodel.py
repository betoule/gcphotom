import jax

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)  # Raises early with stack trace
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from stardiceonline.processing import jaxfitter
from stardiceonline.processing.robuststat import mad, robust_average
from functools import partial
from stardiceonline.tools.header_tools import join


def make_index(index):
    s = np.argsort(index)
    n = np.bincount(index.astype("int"))
    n = n[n != 0]
    l = []
    p = 0
    for i in n:
        l.append(s[p : p + i])
        p = p + i
    return l


def binplot(
    x,
    y,
    nbins=10,
    robust=False,
    data=True,
    scale=True,
    bins=None,
    weights=None,
    ls="none",
    dotkeys={"color": "k"},
    xerr=True,
    ax=None,
    **keys,
):
    """Bin the y data into n bins of x and plot the average and
    dispersion of each bins.

    Arguments:
    ----------
    nbins: int
      Number of bins

    robust: bool
      If True, use median and nmad as estimators of the bin average
      and bin dispersion.

    data: bool
      If True, add data points on the plot

    scale: bool
      Whether the error bars should present the error on the mean or
      the dispersion in the bin

    bins: list
      The bin definition

    weights: array(len(x))
      If not None, use weights in the computation of the mean.
      Provide 1/sigma**2 for optimal weighting with Gaussian noise

    dotkeys: dict
      To keys to pass to plot when drawing data points

    ax: matplotlib axes instance. If None plot to the current axes

    **keys:
      The keys to pass to plot when drawing bins

    Exemples:
    ---------
    >>> x = np.arange(1000); y = np.random.rand(1000);
    >>> binplot(x,y)
    """
    ind = ~np.isnan(x) & ~np.isnan(y)
    x = x[ind]
    y = y[ind]
    if weights is not None:
        weights = weights[ind]
    if bins is None:
        bins = np.linspace(x.min(), x.max() + abs(x.max() * 1e-7), nbins + 1)
    ind = (x < bins.max()) & (x >= bins.min())
    x = x[ind]
    y = y[ind]
    if weights is not None:
        weights = weights[ind]
    yd = np.digitize(x, bins)
    index = make_index(yd)
    ybinned = [y[e] for e in index]
    xbinned = 0.5 * (bins[:-1] + bins[1:])
    usedbins = np.array(np.sort(list(set(yd)))) - 1
    xbinned = xbinned[usedbins]
    bins = bins[usedbins + 1]
    if ax is None:
        ax = plt.gca()
    if data and not "noplot" in keys:
        ax.plot(x, y, ",", **dotkeys)

    if robust is True:
        yplot = [np.median(e) for e in ybinned]
        yerr = np.array([mad(e) for e in ybinned])
    elif robust:
        yres = [
            robust_average(e, sigma=None, clip=robust, mad=False, axis=0)
            for e in ybinned
        ]
        yplot = [e[0] for e in yres]
        yerr = [np.sqrt(e[3]) for e in yres]
    elif weights is not None:
        wbinned = [weights[e] for e in index]
        yplot = [np.average(e, weights=w) for e, w in zip(ybinned, wbinned)]
        if not scale:
            # yerr = np.array([np.std((e - a) * np.sqrt(w))
            #                 for e, w, a in zip(ybinned, wbinned, yplot)])
            yerr = np.array(
                [
                    np.sqrt(np.std((e - a) * np.sqrt(w)) ** 2 / sum(w))
                    for e, w, a in zip(ybinned, wbinned, yplot)
                ]
            )
        else:
            yerr = np.array(
                [np.sqrt(1 / sum(w)) for e, w, a in zip(ybinned, wbinned, yplot)]
            )
        scale = False
        print(yplot)
    else:
        yplot = [np.mean(e) for e in ybinned]
        yerr = np.array([np.std(e) for e in ybinned])

    if scale:
        yerr /= np.sqrt(np.bincount(yd)[usedbins + 1])

    if xerr:
        xerr = np.array([bins, bins]) - np.array([xbinned, xbinned])
    else:
        xerr = None
    if not "noplot" in keys:
        ax.errorbar(xbinned, yplot, yerr=yerr, xerr=xerr, ls=ls, **keys)
    return xbinned, yplot, yerr


def fwhm2alpha(fwhm, beta):
    return fwhm / (2.0 * jnp.sqrt((2.0 ** (1.0 / beta) - 1.0)))


def alpha2fwhm(alpha, beta):
    return alpha * (2.0 * jnp.sqrt((2.0 ** (1.0 / beta) - 1.0)))


def sigma2fwhm(sigma):
    return sigma * jnp.sqrt(8 * jnp.log(2))


def fwhm2sigma(fwhm):
    return fwhm / jnp.sqrt(8 * jnp.log(2))


def moffat(r2, alpha, beta):
    return (beta - 1) / (jnp.pi * alpha**2) * (1 + r2 / alpha**2) ** (-beta)


def moffat_flux(R, alpha, beta):
    """Integrated flux up to a given radius"""
    return 1 - (1 + R**2 / alpha**2) ** (1 - beta)


def imoffat(x, alpha, beta):
    return (
        jnp.sqrt((x * jnp.pi * (alpha * alpha) / (beta - 1)) ** (-1 / beta) - 1) * alpha
    )


def flux_and_couronnes(x):
    return jnp.diff(x, prepend=0, axis=0)


def model(params):
    flux = params["flux"]
    alpha = params["alpha"]  # + params['bf'] * flux/10000.
    beta = params["beta"]
    return (
        flux[None, :] * moffat_flux(rad[:, None], alpha, beta)
        + params["back"][None, :] * rad[:, None] ** 2 * jnp.pi
    )


def radii(cat, lim=slice(5, None)):
    return np.array(
        [
            float(a.split("_")[-1])
            for a in cat.dtype.names
            if a[:5] == "apfl_"
            if a.split("_")[-1] != "ap"
        ]
    )[lim]


def cat_to_flux(cat, rad=None, prefix="apfl"):
    if rad is None:
        rad = radii(cat)
    return jnp.vstack([cat[f"{prefix}_{r:.2f}"] for r in rad])


def extract(cat):
    fluxes = flux_and_couronnes(cat_to_flux(cat))
    var = flux_and_couronnes(jnp.array(cat_to_flux(cat, prefix="apvar")))
    weights = jnp.sqrt(1 / var)
    cont = cat_to_flux(cat, prefix="apother")
    bad = flux_and_couronnes(cat_to_flux(cat, prefix="apbad"))
    goods = (cont == 0) & (bad == 0) & np.isfinite(weights)
    weights = weights.at[~goods].set(0)
    return fluxes, var, goods


class Fitter:
    def __init__(self, cat, model=model):
        self._set_cat(cat)
        self._cut()
        self.model = model

    def _set_cat(self, cat):
        self.cat = cat
        self.fluxes, self.var, self.goods = extract(cat)

    def _cut(self):
        self._set_cat(self.cat[self.goods.sum(axis=0) > 1])

    def _flux(self, params):
        return {**params, "flux": params["flux"] * self.estimate}

    def residuals(self, params, mask=False):
        m = self.model(self._flux(params))
        r = self.fluxes - flux_and_couronnes(m)
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        else:
            return r

    def weighted_residuals(self, params, mask=False):
        m = flux_and_couronnes(self.model(self._flux(params)))
        residuals = self.fluxes - m
        noise = m * 0.01
        r = residuals * 1 / jnp.sqrt(self.var + noise**2) * self.goods
        if mask:
            return r.at[~self.goods].set(jnp.nan)
        else:
            return r

    def chi2(self, params):
        return (self.weighted_residuals(params) ** 2).mean()

    def fit(self, initial_guess=None, niter=10000, learning_rate=5e-3, show=False):
        if initial_guess is None:
            initial_guess = self.initial_guess()

        chi2 = jax.jit(self.chi2)
        bf, extra = jaxfitter.fit_adam(
            chi2, initial_guess, niter=niter, learning_rate=learning_rate, tol=None
        )
        if show:
            plt.plot(extra["loss"])
        return bf, extra

    def fit_tncg(
        self,
        initial_guess=None,
        niter=10,
        lmbda=0.0,
        max_iter_tncg=1000.0,
        verbose=True,
        tol=None,
        show=False,
    ):
        if initial_guess is None:
            initial_guess = self.initial_guess()
        # fwres = lambda x: self.weighted_residuals(x).flatten()
        bf, extra = jaxfitter.tncg(
            self.chi2,
            initial_guess,
            niter=niter,
            lmbda=lmbda,
            max_iter_tncg=max_iter_tncg,
            verbose=verbose,
            tol=tol,
        )
        if show:
            plt.plot(extra["loss"])
        return bf, extra

    def plot_PSF(self, bf, axes=None):
        if axes is None:
            fig = plt.figure("PSF residuals")
            ax1, ax2 = fig.subplots(2, 1, sharex=True)
        else:
            ax1, ax2 = axes
        PSF = (self.fluxes - bf["back"] * area[:, None]) / (bf["flux"] * self.estimate)
        PSF = PSF.at[~self.goods].set(jnp.nan)

        r = np.array(self.residuals(bf, mask=True) / (bf["flux"] * self.estimate))

        ax1.plot(
            rad,
            flux_and_couronnes(
                self.model({**bf, "flux": np.array([1]), "back": np.array([0])})
            ),
            "k-",
        )
        ax1.plot(rad, np.nanmedian(PSF, axis=1), "o")

        ax2.plot(rad, np.nanmean(r, axis=1), "o")
        return ax1, ax2

    def initial_guess(self, beta=4.0, fix=[], a1=5, a2=10):
        f2 = self.fluxes[:a2, :].sum(axis=0)
        f1 = self.fluxes[:a1, :].sum(axis=0)
        ac = np.nanmedian(f2 / f1)
        estimate = f1 * ac
        self.estimate = estimate
        guess = {
            "alpha": fwhm2alpha(
                sigma2fwhm(
                    np.nanmedian((self.cat["gwmxx"] * self.cat["gwmyy"]) ** 0.25)
                ),
                beta,
            ),
            "beta": beta,
            # 'bf': 0.,
            "flux": jnp.ones(len(self.cat)),
            "back": jnp.zeros(len(self.cat)),
        }
        for par in fix:
            guess.pop(par)
        return guess

    def detect_contamination(self, bf):
        wr = self.weighted_residuals(bf)
        self.goods = self.goods & (wr < 5)
        # self._cut()

    def write_cat(self, fname, bf):
        par = self._flux(bf)
        ngoods = self.goods.sum(axis=0)
        wr = self.weighted_residuals(bf, mask=True)
        chi2 = np.nansum(wr**2, axis=0)
        cat = join(
            self.cat, mflux=par["flux"], mback=par["back"], mgoods=ngoods, mchi2=chi2
        )
        np.save(fname, cat)


if __name__ == "__main__":
    from glob import glob

    # fnames = ['catalog_forced_D1_g_08Am02_2008-03-10_975262.npy',
    #           'catalog_forced_D1_g_08Am01_2008-02-11_967395.npy']
    fnames = glob("catalog_forced_D1_g*.npy")
    fnames.sort()
    axes = None
    results = []
    for fname in fnames:
        cat = np.load(fname)

        star_cat = np.load("./avg_cat_D1.npy")
        star123 = np.zeros(star_cat["index"].max() + 1).astype(bool)
        star123[star_cat["index"]] = (star_cat["star"]) & (star_cat["flux_g"] > 10000)

        goods = ~cat["windowed"] & ~cat["saturated"]
        goods &= star123[cat["bindex"]]
        cat = cat[goods]
        rad = radii(cat)
        area = flux_and_couronnes(rad**2 * np.pi)

        f = Fitter(cat)
        bf, extra = f.fit(learning_rate=5e-3)
        f.detect_contamination(bf)
        bf, extra = f.fit(learning_rate=5e-3)
        axes = f.plot_PSF(bf, axes=axes)
        f.write_cat(fname.replace("forced", "mophot"), bf)
        results.append(bf)
    plt.show()

    # goods = goods.at[jnp.abs(weighted_residuals(bf))>50].set(False)
    # kept = goods[0]
    #
    #
    # bf.update(flux=bf['flux'][kept], back=bf['back'][kept])
    # fluxes, var, goods = extract(cat[kept])
    # bf, extra = jaxfitter.fit_adam(chi2, bf, niter=10000, learning_rate=5e-3)
