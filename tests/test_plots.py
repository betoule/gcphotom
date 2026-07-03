import numpy as np

import matplotlib

matplotlib.use("Agg")

from gcphotom.plots import binplot


class TestBinplot:
    def test_noplot(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)

        xbinned, yplot, yerr = binplot(x, y, nbins=5, noplot=True)

        assert len(xbinned) == 5
        assert np.all(np.isfinite(yplot))

    def test_xerr_false(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)

        xbinned, yplot, yerr = binplot(x, y, nbins=5, xerr=False)

        assert len(xbinned) == 5
        assert np.all(np.isfinite(yplot))

    def test_log_bins(self):
        np.random.seed(42)
        x = np.logspace(0, 2, 100)
        y = np.random.randn(100)

        xbinned, yplot, yerr = binplot(x, y, nbins=5, logbins=True)

        assert len(xbinned) == 5
        assert np.all(np.diff(xbinned) > 0)
        assert xbinned[0] > 1.0
