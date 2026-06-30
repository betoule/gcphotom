import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gcphotom.plots import binplot


class TestBinplot:
    def test_basic(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.randn(100) * 0.1

        fig, ax = plt.subplots()
        xbinned, yplot, yerr = binplot(x, y, nbins=10, ax=ax)

        assert len(xbinned) == 10
        assert len(yplot) == 10
        assert len(yerr) == 10
        assert np.all(yerr >= 0)
        plt.close(fig)

    def test_with_nan(self):
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([10, 20, 30, np.nan, 50])

        fig, ax = plt.subplots()
        xbinned, yplot, yerr = binplot(x, y, nbins=3, ax=ax)

        assert not np.any(np.isnan(yplot))
        plt.close(fig)

    def test_weighted(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)
        weights = np.ones(50)

        fig, ax = plt.subplots()
        xbinned, yplot, yerr = binplot(x, y, nbins=5, weights=weights, ax=ax)

        assert len(xbinned) == 5
        assert np.all(yerr >= 0)
        plt.close(fig)

    def test_median_method(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)

        fig, ax = plt.subplots()
        xbinned, yplot, yerr = binplot(x, y, nbins=5, method="median", ax=ax)

        assert len(xbinned) == 5
        assert np.all(yerr >= 0)
        plt.close(fig)

    def test_sigma_clip_method(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)

        fig, ax = plt.subplots()
        xbinned, yplot, yerr = binplot(x, y, nbins=5, method="sigma_clip", ax=ax)

        assert len(xbinned) == 5
        plt.close(fig)

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
        # ensure it did not produce linear bins (first center should be >1)
        assert xbinned[0] > 1.0
