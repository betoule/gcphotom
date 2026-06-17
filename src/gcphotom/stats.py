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
    return np.mean(arr[mask]), np.var(arr[mask])


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


def bin_statistic(
    x,
    y,
    nbins=10,
    bins=None,
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
    weights : array_like or None
        Per-point weights (1/sigma**2 for optimal Gaussian weighting).
    method : {"mean", "median", "sigma_clip"}
        Statistic to compute per bin.
    sigma_clip : float
        Clip factor for ``method="sigma_clip"``.
    scale_err : bool
        If True, divide error by sqrt(N) to get error on the mean.
        If False, return raw bin dispersion.

    Returns
    -------
    xbinned : ndarray
        Bin centers.
    yplot : ndarray
        Per-bin statistic.
    yerr : ndarray
        Per-bin error estimate.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if weights is not None:
        weights = np.asarray(weights)

    # Remove NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if weights is not None:
        weights = weights[mask]

    # Build bins
    if bins is None:
        bins = np.linspace(x.min(), x.max() + abs(x.max()) * 1e-7, nbins + 1)

    # Clip to bin range
    mask = (x >= bins.min()) & (x < bins.max())
    x = x[mask]
    y = y[mask]
    if weights is not None:
        weights = weights[mask]

    # Assign bins and group indices
    bin_labels = np.digitize(x, bins)
    groups = get_bin_indices(bin_labels)
    ybinned = [y[idx] for idx in groups]

    # Used bin indices (0-based)
    used = np.sort(np.unique(bin_labels)) - 1
    xbinned = 0.5 * (bins[used] + bins[used + 1])

    # Compute per-bin statistics
    if method == "median":
        yplot = np.array([np.median(vals) for vals in ybinned])
        yerr = np.array([mad(vals) for vals in ybinned])
        scale_err = False
    elif method == "sigma_clip":
        results = [robust_average(vals, clip=sigma_clip) for vals in ybinned]
        yplot = np.array([r[0] for r in results])
        yerr = np.array([np.sqrt(r[1]) for r in results])
        scale_err = False
    elif weights is not None:
        wbinned = [weights[idx] for idx in groups]
        yplot = np.array(
            [np.average(vals, weights=w) for vals, w in zip(ybinned, wbinned)]
        )
        if scale_err:
            yerr = np.array([np.sqrt(1.0 / np.sum(w)) for w in wbinned])
        else:
            yerr = np.array(
                [
                    np.sqrt(np.std((vals - m) * np.sqrt(w)) ** 2 / np.sum(w))
                    for vals, w, m in zip(ybinned, wbinned, yplot)
                ]
            )
    else:
        yplot = np.array([np.mean(vals) for vals in ybinned])
        yerr = np.array([np.std(vals) for vals in ybinned])

    if scale_err:
        counts = np.bincount(bin_labels)[used + 1]
        yerr /= np.sqrt(counts)

    return xbinned, yplot, yerr
