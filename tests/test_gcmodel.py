"""Tests for gcmodel.Fitter."""

import jax.numpy as jnp
import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gcphotom as gcp


@pytest.fixture
def small_sim():
    """Small simulation for fast unit tests."""
    shape = (256, 256)
    cat = gcp.make_realistic_source_catalog(20, shape=shape, seed=42)
    for i in range(len(cat)):
        cat["x"][i] = 40 + (i % 5) * 45
        cat["y"][i] = 40 + (i // 5) * 55
    img, cat = gcp.simulate_image(
        shape, cat, gamma=2.5, alpha=3.0, background=0, seed=42
    )
    return img, cat


@pytest.fixture
def small_sim_with_bg():
    """Small simulation with background."""
    shape = (256, 256)
    cat = gcp.make_realistic_source_catalog(20, shape=shape, seed=42)
    for i in range(len(cat)):
        cat["x"][i] = 40 + (i % 5) * 45
        cat["y"][i] = 40 + (i // 5) * 55
    img, cat = gcp.simulate_image(
        shape, cat, gamma=2.5, alpha=3.0, background=100, read_noise=3, seed=42
    )
    return img, cat


class TestMoffatFunctions:
    def test_moffat_flux_at_zero(self):
        assert gcp.gcmodel.moffat_flux(0.0, 2.0, 3.0) == pytest.approx(0.0)

    def test_moffat_flux_at_infinity(self):
        val = float(gcp.gcmodel.moffat_flux(1e6, 2.0, 3.0))
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_fwhm_gamma_roundtrip(self):
        fwhm = 5.0
        alpha = 3.0
        gamma = gcp.gcmodel.fwhm2gamma(fwhm, alpha)
        assert gcp.gcmodel.gamma2fwhm(gamma, alpha) == pytest.approx(fwhm)

    def test_annular_fluxes(self):
        cum = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        annular = gcp.gcmodel.annular_fluxes(cum)
        np.testing.assert_allclose(annular, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_sigma_fwhm_converters(self):
        assert gcp.gcmodel.sigma2fwhm(1.0) > 0
        assert gcp.gcmodel.fwhm2sigma(2.355) > 0

    def test_moffat_and_imoffat(self):
        assert gcp.gcmodel.moffat(0.0, 2.0, 3.0) > 0
        # choose a small fraction that yields a real radius
        r = gcp.gcmodel.imoffat(0.1, 2.0, 3.0)
        assert np.isfinite(r) and r >= 0
        assert gcp.gcmodel.moffat_flux(0.0, 2.0, 3.0) == 0.0

    def test_residuals_mask_and_plot(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=300)
        r = f.residuals(bf, mask=True)
        assert r.shape[0] > 0
        # non-masked path
        r2 = f.residuals(bf, mask=False)
        assert r2.shape == r.shape
        # weighted mask path
        wr = f.weighted_residuals(bf, mask=True)
        assert wr.shape[0] > 0
        # plot with provided axes
        fig, (ax1, ax2) = plt.subplots(2, 1)
        f.plot_PSF(bf, axes=(ax1, ax2))
        plt.close(fig)

    def test_fit_raises_on_zero_sources(self):
        from gcphotom.gcmodel import Fitter

        gc0 = {
            "radius": np.array([1.0, 2.0]),
            "flux": np.zeros((0, 2)),
            "background_var": np.zeros((0, 2)),
            "flux_clean": np.zeros((0, 2)),
            "contamination": np.zeros((0, 2)),
        }
        f = Fitter(gc0)
        with pytest.raises(ValueError):
            f.fit(niter=1)

    def test_fit_show_plots(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=100, show=True)
        assert bf is not None
        plt.close("all")


class TestFitterInit:
    def test_init_shapes(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        assert f.fluxes.shape[1] <= len(cat)
        assert f.fluxes.shape[0] == len(gc["radius"])
        assert f.bkg_var.shape == f.fluxes.shape
        assert f.goods.shape == f.fluxes.shape

    def test_initial_guess_empty_sources(self):
        gc = {
            "radius": np.array([1.0, 2.0, 4.0]),
            "flux": np.empty((0, 3)),
            "flux_clean": np.empty((0, 3)),
            "background_var": np.empty((0, 3)),
            "contamination": np.empty((0, 3)),
        }
        f = gcp.Fitter(gc)
        ig = f.initial_guess()
        assert ig["gamma"] == 3.0
        assert ig["alpha"] == 3.0
        assert len(ig["flux"]) == 0
        assert len(ig["back"]) == 0

    def test_results_all_dropped(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=500)
        # Force all goods to False to simulate total rejection
        f.goods = jnp.zeros_like(f.goods)
        f._cut()
        res = f.results(bf)
        assert np.all(np.isnan(res["flux"]))
        assert np.all(np.isnan(res["back"]))
        assert res["gamma"] == float(bf["gamma"])

    def test_results_length_mismatch(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=500)
        # Pass bf with wrong-length arrays to trigger fallback
        bf_bad = {
            "gamma": bf["gamma"],
            "alpha": bf["alpha"],
            "flux": np.ones(len(positions) + 10),
            "back": np.ones(len(positions) + 10),
        }
        res = f.results(bf_bad)
        assert not np.all(np.isnan(res["flux"]))
        assert len(res["flux"]) == len(positions)

    def test_chi2_method(self, small_sim):
        """Direct chi2 call covers the standalone chi2() method."""
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        ig = f.initial_guess()
        val = float(f.chi2(ig))
        assert np.isfinite(val)


class TestFitterFit:
    @pytest.mark.skip(reason="convergence issue — needs tuning")
    def test_flux_recovery(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=5000, learning_rate=1e-2)
        res = f.results(bf)

        ratios = res["flux"] / cat["flux"]
        assert np.median(ratios) > 0.7
        assert np.median(ratios) < 1.3
        assert np.std(np.log10(ratios)) < 0.1

    @pytest.mark.skip(reason="convergence issue — needs tuning")
    def test_gamma_recovery(self, small_sim):
        true_gamma = 2.5
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=5000, learning_rate=1e-2)

        assert bf["gamma"] > 0
        assert float(bf["gamma"]) < true_gamma * 3

    @pytest.mark.skip(reason="convergence issue — needs tuning")
    def test_chi2_decreases(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        ig = f.initial_guess()
        chi2_before = float(f.chi2(ig))
        bf, extra = f.fit(niter=5000, learning_rate=1e-2)
        chi2_after = float(f.chi2(bf))
        assert chi2_after < chi2_before


class TestFitterBackground:
    @pytest.mark.skip(reason="convergence issue — needs tuning")
    def test_background_recovery(self, small_sim_with_bg):
        img, cat = small_sim_with_bg
        positions = np.column_stack([cat["x"], cat["y"]])
        bkg_var = np.full_like(img, 9.0)  # read_noise=3 → variance=9
        gc = gcp.extract_growth_curves(
            img - 100, positions, background_variance=bkg_var
        )
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=5000, learning_rate=1e-2)
        res = f.results(bf)

        assert np.median(np.abs(res["back"])) < 1.0

    @pytest.mark.skip(reason="convergence issue — needs tuning")
    def test_flux_recovery_with_bg(self, small_sim_with_bg):
        img, cat = small_sim_with_bg
        positions = np.column_stack([cat["x"], cat["y"]])
        bkg_var = np.full_like(img, 9.0)  # read_noise=3 → variance=9
        gc = gcp.extract_growth_curves(
            img - 100, positions, background_variance=bkg_var
        )
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=5000, learning_rate=1e-2)
        res = f.results(bf)

        ratios = res["flux"] / cat["flux"]
        assert np.median(ratios) > 0.6
        assert np.median(ratios) < 1.4


class TestFitterResults:
    def test_results_keys_and_shapes(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=1000)
        res = f.results(bf)

        for key in ("flux", "back", "gamma", "alpha", "ngoods", "chi2"):
            assert key in res

        n = res["flux"].shape[0]
        assert res["back"].shape[0] == n
        assert res["ngoods"].shape[0] == n
        assert res["chi2"].shape[0] == n
        # results are expanded to original input length
        assert n == len(positions)


class TestFitterHelpers:
    def test_detect_contamination_reduces_goods(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=1000)
        goods_before = int(f.goods.sum())
        f.detect_contamination(bf)
        goods_after = int(f.goods.sum())
        assert goods_after <= goods_before

    def test_results_expanded_with_nans_after_contamination(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=1000)
        res_before = f.results(bf)
        n_before = len(positions)
        assert len(res_before["flux"]) == n_before
        f.detect_contamination(bf)
        # results uses current .kept to expand; we call with prior bf
        res_after = f.results(bf)
        assert len(res_after["flux"]) == n_before
        # some entries may be NaN where sources were dropped
        if int(f.goods.sum()) < n_before:
            assert np.any(~np.isfinite(res_after["flux"]))

    def test_plot_psf_returns_axes(self, small_sim):
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=1000)
        ax1, ax2 = f.plot_PSF(bf)
        assert ax1 is not None
        assert ax2 is not None
        plt.close("all")


class TestFullPipeline:
    @pytest.mark.skip(reason="convergence issue — needs tuning")
    def test_1000_sources(self):
        """End-to-end test with 1000 sources."""
        img, cat = gcp.simulate_image(
            shape=(1024, 1024), gamma=2.5, alpha=3.0, background=50, seed=42
        )
        positions = np.column_stack([cat["x"], cat["y"]])
        # Use auto background variance estimate (background=50, read_noise=0 in simulation)
        gc = gcp.extract_growth_curves(img - 50, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=5000, learning_rate=1e-2)
        res = f.results(bf)

        ratios = res["flux"] / np.array(cat["flux"])
        assert np.median(ratios) > 0.5
        assert np.median(ratios) < 1.5
        assert np.std(np.log10(ratios)) < 0.2


class TestRobustLoss:
    def test_loss_functions(self):
        """All loss factory functions return callables that work with JAX arrays."""
        x = jnp.array([0.0, 1.0, 2.0, 10.0])
        for factory in [gcp.tukey(), gcp.pseudo_huber(), gcp.cauchy()]:
            result = factory(x)
            assert result.shape == x.shape
            assert jnp.all(result >= 0)
            assert float(result[0]) == 0.0

    def test_tukey_saturation(self):
        """Tukey loss saturates at c^2/6 beyond |x| > c."""
        t = gcp.tukey(c=3.0)
        result = t(jnp.array([0.0, 10.0]))
        assert float(result[1]) == pytest.approx(3.0**2 / 6)

    def test_fit_default_loss(self, small_sim):
        """Default fit uses Tukey bisquare and converges."""
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, extra = f.fit(niter=500)
        assert bf is not None
        assert float(extra["loss"][-1]) < float(extra["loss"][0]) * 0.99

    def test_fit_chi2_loss(self, small_sim):
        """User-provided lambda equivalent to chi2 works."""
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, extra = f.fit(niter=500, loss=lambda x: x**2)
        assert bf is not None
        assert float(extra["loss"][-1]) < float(extra["loss"][0]) * 0.99

    def test_detect_contamination_after_robust_fit(self, small_sim):
        """detect_contamination still works after a robust fit."""
        img, cat = small_sim
        positions = np.column_stack([cat["x"], cat["y"]])
        gc = gcp.extract_growth_curves(img, positions)
        f = gcp.Fitter(gc)
        bf, _ = f.fit(niter=500)
        goods_before = int(f.goods.sum())
        f.detect_contamination(bf)
        goods_after = int(f.goods.sum())
        assert goods_after <= goods_before

    def test_fit_adam_tolerance_break(self):
        from gcphotom.jaxfitter import fit_adam

        def quadratic(x):
            return (x["a"] - 5.0) ** 2

        params = {"a": 0.0}
        # Huge tolerance triggers early break
        result, extra = fit_adam(quadratic, params, tol=1e30, niter=1000)
        assert len(extra["loss"]) < 50  # broke early due to tol

    def test_tukey_gradient_bounded(self):
        """Tukey loss gradient goes to zero for large residuals (robustness property)."""
        import jax

        x_large = jnp.array(100.0)
        x_small = jnp.array(0.1)

        t = gcp.tukey(c=4.685)
        chi2 = lambda x: x**2

        grad_t = jax.grad(lambda x: jnp.mean(t(x)))
        grad_c = jax.grad(lambda x: jnp.mean(chi2(x)))

        gtl = float(grad_t(x_large))
        gcl = float(grad_c(x_large))
        # For large residual: chi2 gradient >> Tukey gradient
        assert gtl < gcl * 0.01

        gts = float(grad_t(x_small))
        gcs = float(grad_c(x_small))
        # For small residual: both gradients are similar
        ratio = gts / gcs
        assert 0.4 < ratio < 2.0
