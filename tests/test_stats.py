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
        mean, var, n = robust_average(vals)
        assert np.isclose(mean, 3.0)
        assert var >= 0
        assert n == 5

    def test_outliers_removed(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        mean, _, n = robust_average(vals, clip=2.0)
        assert np.isclose(mean, np.mean(vals[:5]), atol=0.5)
        assert n == 5

    def test_constant(self):
        mean, var, n = robust_average([5.0, 5.0, 5.0])
        assert np.isclose(mean, 5.0)
        assert np.isclose(var, 0.0)
        assert n == 3


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

    def test_median_scale_err_is_se(self):
        np.random.seed(42)
        # 4 bins with exactly 25 points each
        n_per = 25
        x = np.concatenate(
            [
                np.full(n_per, 0.0),
                np.full(n_per, 1.0),
                np.full(n_per, 2.0),
                np.full(n_per, 3.0),
            ]
        )
        x = x + np.random.uniform(-0.1, 0.1, len(x))
        y = np.random.randn(len(x))
        bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
        _, _, disp = bin_statistic(x, y, bins=bins, method="median", scale_err=False)
        _, _, se = bin_statistic(x, y, bins=bins, method="median", scale_err=True)
        n = n_per
        factor = np.sqrt(np.pi / 2.0)
        expected = disp * factor / np.sqrt(n)
        np.testing.assert_allclose(se, expected, rtol=1e-8)

    def test_sigma_clip_scale_err_uses_nkept(self):
        np.random.seed(123)
        n_clean = 30
        n_out = 5
        y_clean = np.random.randn(n_clean)
        y_out = np.random.randn(n_out) * 10 + 50
        y = np.concatenate([y_clean, y_out])
        x = np.zeros(len(y))
        bins = np.array([-0.5, 0.5])
        _, _, disp = bin_statistic(
            x, y, bins=bins, method="sigma_clip", scale_err=False, sigma_clip=3.0
        )
        _, _, se = bin_statistic(
            x, y, bins=bins, method="sigma_clip", scale_err=True, sigma_clip=3.0
        )
        _, var, n_used = robust_average(y, clip=3.0)
        assert n_used == n_clean
        expected = np.sqrt(var) / np.sqrt(n_used)
        np.testing.assert_allclose(se, expected, rtol=1e-8)

    def test_weighted_mean_scale_err_correct(self):
        np.random.seed(42)
        # bin0: 10 pts w=2 -> sumw=20; bin1: 5 pts w=4 -> sumw=20
        x = np.concatenate([np.zeros(10), np.ones(5)])
        x = x + np.random.uniform(-0.01, 0.01, 15)
        y = np.random.randn(15)
        w = np.concatenate([np.full(10, 2.0), np.full(5, 4.0)])
        bins = np.array([-0.5, 0.5, 1.5])
        _, _, se = bin_statistic(x, y, bins=bins, weights=w, scale_err=True)
        expected = np.array([1.0 / np.sqrt(20), 1.0 / np.sqrt(20)])
        np.testing.assert_allclose(se, expected, rtol=1e-10)

    def test_weighted_robust_scale_err_raises(self):
        x = np.linspace(0, 5, 10)
        y = np.random.randn(10)
        w = np.ones(10)
        for meth in ("median", "sigma_clip"):
            try:
                bin_statistic(x, y, nbins=2, weights=w, method=meth, scale_err=True)
                raised = False
            except ValueError:
                raised = True
            assert raised

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
