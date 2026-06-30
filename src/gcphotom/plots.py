import matplotlib.pyplot as plt
import numpy as np

from gcphotom.stats import _build_bins, bin_statistic


def binplot(
    x,
    y,
    nbins=10,
    bins=None,
    logbins=False,
    *,
    weights=None,
    method="mean",
    sigma_clip=5.0,
    scale_err=True,
    data=True,
    dotkeys=None,
    ls="none",
    xerr=True,
    ax=None,
    noplot=False,
    **keys,
):
    """Bin the y data into bins of x and plot the average and dispersion.

    Parameters
    ----------
    x : array_like
        Bin variable.
    y : array_like
        Value variable.
    nbins : int
        Number of bins (ignored if *bins* is provided).
    bins : array_like or None
        Explicit bin edges.
    logbins : bool
        If True, use logarithmically spaced bins (requires all x > 0).
    weights : array_like or None
        Per-point weights (1/sigma**2 for optimal Gaussian weighting).
    method : {"mean", "median", "sigma_clip"}
        Statistic to compute per bin.
    sigma_clip : float
        Clip factor for ``method="sigma_clip"``.
    scale_err : bool
        If True and method=="mean" (no weights), divide error by sqrt(N)
        to get error on the mean. For "median" or "sigma_clip" this has
        no effect (those return robust dispersion, not error-on-mean).
    data : bool
        If True, overlay raw data points on the plot.
    dotkeys : dict or None
        Keyword arguments passed to ``ax.plot`` for data points.
    ls : str
        Linestyle for the binned errorbar line.
    xerr : bool or array_like
        If True, compute x-error from bin edges.
    ax : matplotlib.axes.Axes or None
        Target axes. Uses ``plt.gca()`` if None.
    noplot : bool
        If True, skip plotting and return binned values only.
    **keys
        Additional keyword arguments passed to ``ax.errorbar``.

    Returns
    -------
    xbinned : ndarray
        Bin centers.
    yplot : ndarray
        Per-bin statistic.
    yerr : ndarray
        Per-bin error estimate.
    """
    if dotkeys is None:
        dotkeys = {"color": "k"}

    # Filter NaN for bin construction
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    valid = ~np.isnan(x_arr) & ~np.isnan(y_arr)

    # Build bin edges if not provided
    if bins is None:
        bins = _build_bins(x_arr[valid], nbins, logbins=logbins)

    xbinned, yplot, yerr = bin_statistic(
        x,
        y,
        nbins=nbins,
        bins=bins,
        logbins=logbins,
        weights=weights,
        method=method,
        sigma_clip=sigma_clip,
        scale_err=scale_err,
    )

    if noplot:
        return xbinned, yplot, yerr

    if ax is None:
        ax = plt.gca()

    if data:
        ax.plot(x, y, ",", **dotkeys)

    if xerr is True:
        # Find used bins and compute x-error from bin edges
        labels = np.digitize(x_arr[valid], bins)
        used = np.sort(np.unique(labels))
        bin_slice = used - 1
        xerr = np.array([xbinned - bins[bin_slice], bins[bin_slice + 1] - xbinned])
    elif xerr is False:
        xerr = None

    ax.errorbar(xbinned, yplot, yerr=yerr, xerr=xerr, ls=ls, **keys)
    return xbinned, yplot, yerr
