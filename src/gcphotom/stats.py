import numpy as np


def mad(arr):
    """Median Absolute Deviation with NMAD scaling factor.

    Parameters
    ----------
    arr : array_like
        Input data.

    Returns
    -------
    float
        NMAD = 1.4826 * median(|x - median(x)|)
    """
    return 1.4826 * np.median(np.abs(arr - np.median(arr)))


def robust_average(arr, clip=5.0):
    """Sigma-clipped mean and variance.

    Parameters
    ----------
    arr : array_like
        Input data.
    clip : float
        Number of standard deviations for clipping.

    Returns
    -------
    mean : float
        Sigma-clipped mean.
    var : float
        Sigma-clipped variance.
    n_used : int
        Number of points retained after clipping.
    """
    arr = np.asarray(arr)
    mask = np.isfinite(arr)
    for _ in range(10):
        mean = np.mean(arr[mask])
        std = np.std(arr[mask])
        if std == 0:
            break
        new_mask = mask & (np.abs(arr - mean) <= clip * std)
        if np.sum(new_mask) == np.sum(mask):
            break
        mask = new_mask
    n_used = int(np.sum(mask))
    return np.mean(arr[mask]), np.var(arr[mask]), n_used


def get_bin_indices(bin_labels):
    """Return sorted indices grouped by bin label.

    Parameters
    ----------
    bin_labels : array_like of int
        Bin assignment for each element (1-based from np.digitize).

    Returns
    -------
    list of ndarray
        List of index arrays, one per non-empty bin in order.
    """
    sorted_idx = np.argsort(bin_labels)
    counts = np.bincount(bin_labels.astype(int))
    counts = counts[counts != 0]
    result = []
    start = 0
    for count in counts:
        result.append(sorted_idx[start : start + count])
        start += count
    return result


def _build_bins(x, nbins, logbins=False):
    """Build bin edges.

    x must be finite and for logbins=True all x > 0.
    """
    x = np.asarray(x)
    xmin = x.min()
    xmax = x.max()
    if logbins:
        if xmin <= 0:
            raise ValueError("log-spaced bins require all x > 0")
        if xmax > 0:
            xmax = xmax * (1 + 1e-7)
        else:
            xmax = xmax + 1e-7
        return np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    return np.linspace(xmin, xmax + abs(xmax) * 1e-7, nbins + 1)


def _median_se_from_mad(disp, ns):
    """SE of median from MAD (NMAD) under normality: MAD * sqrt(pi/2) / sqrt(n)."""
    factor = np.sqrt(np.pi / 2)
    return disp * factor / np.sqrt(ns)


def _se_from_disp_and_n(disp, n):
    """SE = disp / sqrt(n) with safe handling for n=0."""
    safe = np.where(n > 0, n, 1)
    return np.where(n > 0, disp / np.sqrt(safe), np.nan)


def _compute_median_stat(ybinned, ns, scale_err):
    yplot = np.array([np.median(vals) for vals in ybinned])
    disp = np.array([mad(vals) for vals in ybinned])
    yerr = _median_se_from_mad(disp, ns) if scale_err else disp
    return yplot, yerr


def _compute_sigma_clip_stat(ybinned, sigma_clip, scale_err):
    results = [robust_average(vals, clip=sigma_clip) for vals in ybinned]
    yplot = np.array([r[0] for r in results])
    vars_ = np.array([r[1] for r in results])
    n_used = np.array([r[2] for r in results])
    disp = np.sqrt(vars_)
    yerr = _se_from_disp_and_n(disp, n_used) if scale_err else disp
    return yplot, yerr


def _compute_weighted_stat(ybinned, wbinned, scale_err):
    yplot = np.array([np.average(vals, weights=w) for vals, w in zip(ybinned, wbinned)])
    if scale_err:
        yerr = np.array([np.sqrt(1.0 / np.sum(w)) for w in wbinned])
    else:
        yerr = np.array(
            [
                np.sqrt(np.std((vals - m) * np.sqrt(w)) ** 2 / np.sum(w))
                for vals, w, m in zip(ybinned, wbinned, yplot)
            ]
        )
    return yplot, yerr


def _compute_mean_stat(ybinned):
    yplot = np.array([np.mean(vals) for vals in ybinned])
    yerr = np.array([np.std(vals) for vals in ybinned])
    return yplot, yerr


def _prepare_bin_data(x, y, weights, bins, nbins, logbins):
    """Filter NaNs, build bins, clip range, assign groups, return structures.

    Returns
    -------
    bins : ndarray
    used : ndarray
    xbinned : ndarray
    ybinned : list of arrays
    groups : list of index arrays
    ns : ndarray
    wbinned : list or None
    bin_labels : ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if weights is not None:
        weights = np.asarray(weights)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if weights is not None:
        weights = weights[mask]

    if bins is None:
        bins = _build_bins(x, nbins, logbins=logbins)

    mask = (x >= bins.min()) & (x < bins.max())
    x = x[mask]
    y = y[mask]
    if weights is not None:
        weights = weights[mask]

    bin_labels = np.digitize(x, bins)
    groups = get_bin_indices(bin_labels)
    ybinned = [y[idx] for idx in groups]
    used = np.sort(np.unique(bin_labels)) - 1
    xbinned = 0.5 * (bins[used] + bins[used + 1])
    ns = np.array([len(g) for g in groups])
    wbinned = [weights[idx] for idx in groups] if weights is not None else None
    return bins, used, xbinned, ybinned, groups, ns, wbinned, bin_labels


def bin_statistic(
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
):
    """Compute binned statistics of y over x.

    Parameters
    ----------
    x : array_like
        Bin variable.
    y : array_like
        Value variable.
    nbins : int
        Number of bins (ignored if bins is provided).
    bins : array_like or None
        Explicit bin edges.
    logbins : bool
        If True, generate logarithmically spaced bins (requires x > 0).
    weights : array_like or None
        Per-point weights (1/sigma**2 for optimal Gaussian weighting).
    method : {"mean", "median", "sigma_clip"}
        Statistic to compute per bin.
    sigma_clip : float
        Clip factor for ``method="sigma_clip"``.
    scale_err : bool
        If True, return an estimate of the standard error on the per-bin
        central value (the reported "mean" statistic). For ``method="mean"``
        this is the standard error on the mean. For ``method="median"`` it is
        an estimate of the standard error of the median derived from the MAD
        (under normality). For ``method="sigma_clip"`` it is the standard
        error of the clipped mean. If False, return a dispersion measure
        (standard deviation, MAD, or clipped std) instead. Default True.
        When weights are provided and ``scale_err=True`` with a robust
        method, a ValueError is raised (weighted robust SE not supported).

    Returns
    -------
    xbinned : ndarray
        Bin centers.
    yplot : ndarray
        Per-bin statistic.
    yerr : ndarray
        Per-bin error estimate.
    """
    bins_arr, used, xbinned, ybinned, groups, ns, wbinned, bin_labels = (
        _prepare_bin_data(x, y, weights, bins, nbins, logbins)
    )

    # Compute per-bin statistics
    if method == "median":
        if wbinned is not None and scale_err:
            raise ValueError("scale_err=True is not supported for weighted median")
        yplot, yerr = _compute_median_stat(ybinned, ns, scale_err)
    elif method == "sigma_clip":
        if wbinned is not None and scale_err:
            raise ValueError("scale_err=True is not supported for weighted sigma_clip")
        yplot, yerr = _compute_sigma_clip_stat(ybinned, sigma_clip, scale_err)
    elif wbinned is not None:
        yplot, yerr = _compute_weighted_stat(ybinned, wbinned, scale_err)
    else:
        yplot, yerr = _compute_mean_stat(ybinned)

    # Apply outer scaling only for plain (unweighted) mean with scale_err.
    # Robust methods and weighted mean handle their SE inside the branches.
    if scale_err and wbinned is None and method not in ("median", "sigma_clip"):
        counts = np.bincount(bin_labels)[used + 1]
        yerr /= np.sqrt(counts)

    return xbinned, yplot, yerr
