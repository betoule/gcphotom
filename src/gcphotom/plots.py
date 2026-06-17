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
