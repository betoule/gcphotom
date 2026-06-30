import numpy as np

from gcphotom.stats import mad, robust_average, get_bin_indices, bin_statistic


class TestMad:
    def test_constant(self):
        assert mad([1.0, 1.0, 1.0]) == 0.0

    def test_known(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mad(vals)
        assert result > 0
        assert np.isclose(result, 1.4826 * 1.0)

    def test_scale(self):
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(mad(base * 10), mad(base) * 10)


class TestRobustAverage:
    def test_clean_data(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, var = robust_average(vals)
        assert np.isclose(mean, 3.0)
        assert var >= 0

    def test_outliers_removed(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        mean, _ = robust_average(vals, clip=2.0)
        assert np.isclose(mean, np.mean(vals[:5]), atol=0.5)

    def test_constant(self):
        mean, var = robust_average([5.0, 5.0, 5.0])
        assert np.isclose(mean, 5.0)
        assert np.isclose(var, 0.0)


class TestGetBinIndices:
    def test_simple(self):
        labels = np.array([1, 2, 1, 3, 2])
        groups = get_bin_indices(labels)
        assert len(groups) == 3
        assert set(groups[0]) == {0, 2}
        assert set(groups[1]) == {1, 4}
        assert set(groups[2]) == {3}

    def test_single_bin(self):
        labels = np.array([1, 1, 1])
        groups = get_bin_indices(labels)
        assert len(groups) == 1
        assert len(groups[0]) == 3


class TestBinStatistic:
    def test_mean(self):
        x = np.linspace(0, 10, 100)
        y = np.ones(100)
        _, yplot, _ = bin_statistic(x, y, nbins=10, method="mean")
        np.testing.assert_allclose(yplot, 1.0)

    def test_median(self):
        x = np.linspace(0, 10, 100)
        y = np.arange(100)
        _, yplot, yerr = bin_statistic(x, y, nbins=10, method="median")
        assert len(yplot) == 10
        assert np.all(yerr >= 0)

    def test_sigma_clip(self):
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)
        _, yplot, yerr = bin_statistic(
            x, y, nbins=5, method="sigma_clip", sigma_clip=3.0
        )
        assert len(yplot) == 5
        assert np.all(yerr >= 0)

    def test_weighted(self):
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)
        weights = np.ones(50)
        _, yplot, yerr = bin_statistic(x, y, nbins=5, weights=weights, scale_err=True)
        assert len(yplot) == 5
        assert np.all(yerr >= 0)

    def test_nan_handling(self):
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([10, 20, 30, np.nan, 50])
        xbinned, yplot, yerr = bin_statistic(x, y, nbins=3)
        assert not np.any(np.isnan(yplot))
        assert len(xbinned) == len(yplot) == len(yerr)

    def test_explicit_bins(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        bins = np.array([0, 5, 10])
        xbinned, yplot, yerr = bin_statistic(x, y, bins=bins)
        assert len(xbinned) == 2

    def test_scale_err_false(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 100)
        y = np.random.randn(100)
        _, _, err_scaled = bin_statistic(x, y, nbins=10, scale_err=True)
        _, _, err_raw = bin_statistic(x, y, nbins=10, scale_err=False)
        assert np.all(err_raw >= err_scaled)

    def test_weighted_no_scale(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)
        w = np.ones(50)
        _, _, err = bin_statistic(x, y, nbins=5, weights=w, scale_err=False)
        assert len(err) == 5
        assert np.all(err >= 0)

    def test_log_bins(self):
        np.random.seed(42)
        x = np.logspace(0, 2, 100)
        y = np.random.randn(100)
        xbinned, yplot, yerr = bin_statistic(x, y, nbins=5, logbins=True)
        assert len(xbinned) == 5
        # bins should be strictly increasing and log-spaced
        assert np.all(np.diff(xbinned) > 0)
        # centers should be close to geometric means of edges
        # (computed internally) - just sanity check range
        assert xbinned[0] > 1.0 and xbinned[-1] < 100.0

    def test_log_bins_requires_positive(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        try:
            bin_statistic(x, y, nbins=3, logbins=True)
            raised = False
        except ValueError:
            raised = True
        assert raised
