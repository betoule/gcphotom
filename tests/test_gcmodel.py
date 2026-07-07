"""Tests for gcmodel.Fitter."""

import jax.numpy as jnp
import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gcphotom as gcp


@pytest.fixture(scope="module")
def small_sim_gc():
    """Simulate + extract growth curves once per module; all tests share this."""
    shape = (256, 256)
    cat = gcp.make_realistic_source_catalog(20, shape=shape, seed=42)
    for i in range(len(cat)):
        cat["x"][i] = 40 + (i % 5) * 45
        cat["y"][i] = 40 + (i // 5) * 55
    img, cat = gcp.simulate_image(
        shape, cat, gamma=2.5, alpha=3.0, background=0, seed=42
    )
    positions = np.column_stack([cat["x"], cat["y"]])
    gc = gcp.extract_growth_curves(img, positions, show_progress=False)
    return img, cat, gc


@pytest.fixture(scope="module")
def small_sim_fitted(small_sim_gc):
    """Fit once per module; all read-only tests share this fitter."""
    _, _, gc = small_sim_gc
    f = gcp.Fitter(gc)
    bf, extra = f.fit(show_progress=False, niter=100, compute_uncertainty=True)
    result = f.results(bf)
    return f, bf, extra, result


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

    def test_moffat_center_positive(self):
        assert gcp.gcmodel.moffat(0.0, 2.0, 3.0) > 0

    def test_imoffat_finite(self):
        r = gcp.gcmodel.imoffat(0.1, 2.0, 3.0)
        assert np.isfinite(r) and r >= 0

    def test_residuals_mask_and_plot(self, small_sim_fitted):
        f, bf, _, _ = small_sim_fitted
        r = f.residuals(bf, mask=True)
        assert r.shape[0] > 0
        r2 = f.residuals(bf, mask=False)
        assert r2.shape == r.shape
        wr = f.weighted_residuals(bf, mask=True)
        assert wr.shape[0] > 0
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
            f.fit(show_progress=False, niter=1)

    def test_fit_show_plots(self, small_sim_gc):
        _, _, gc = small_sim_gc
        f = gcp.Fitter(gc)
        bf, _ = f.fit(show_progress=False, niter=50, show=True)
        assert bf is not None
        plt.close("all")


class TestFitterInit:
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
        assert len(ig["flux"]) == 0
        assert len(ig["back"]) == 0

    def test_results_all_dropped(self, small_sim_gc):
        _, _, gc = small_sim_gc
        f = gcp.Fitter(gc)
        bf, _ = f.fit(show_progress=False, niter=50)
        f.goods = jnp.zeros_like(f.goods)
        f._cut()
        res = f.results(bf)
        assert np.all(np.isnan(res["flux"]))
        assert np.all(np.isnan(res["back"]))

    def test_rescale_params_includes_uncertainty(self, small_sim_fitted):
        f, bf, _, _ = small_sim_fitted
        rp = f.rescale_params(bf)
        assert "std_errors" in rp

    def test_expand_to_original_length(self, small_sim_gc):
        _, _, gc = small_sim_gc
        f = gcp.Fitter(gc)
        bf, _ = f.fit(show_progress=False, niter=50)
        n_orig = f._orig_n
        n_cur = f.fluxes.shape[1]
        arr = np.arange(n_cur, dtype=float)
        expanded = f.expand_to_original(arr)
        assert len(expanded) == n_orig

    def test_goodness_chi2_finite(self, small_sim_fitted):
        f, bf, _, _ = small_sim_fitted
        g = f.goodness(bf)
        assert np.all(np.isfinite(g["chi2"]))

    def test_chi2_method(self, small_sim_gc):
        _, _, gc = small_sim_gc
        f = gcp.Fitter(gc)
        ig = f.initial_guess()
        val = float(f.chi2(ig))
        assert np.isfinite(val)


class TestFitterHelpers:
    def test_detect_contamination_reduces_goods(self, small_sim_gc):
        _, _, gc = small_sim_gc
        f = gcp.Fitter(gc)
        bf, _ = f.fit(show_progress=False, niter=50)
        goods_before = int(f.goods.sum())
        f.detect_contamination(bf)
        goods_after = int(f.goods.sum())
        assert goods_after <= goods_before

    def test_results_expanded_with_nans_after_contamination(self, small_sim_gc):
        _, _, gc = small_sim_gc
        f = gcp.Fitter(gc)
        bf, _ = f.fit(show_progress=False, niter=50)
        n_before = f._orig_n
        f.detect_contamination(bf)
        bf2, _ = f.fit(show_progress=False, niter=50)
        res_after = f.results(bf2)
        assert len(res_after["flux"]) == n_before
        if n_before > f.fluxes.shape[1]:
            assert np.any(np.isnan(res_after["flux"]))

    def test_plot_psf_returns_axes(self, small_sim_fitted):
        f, bf, _, _ = small_sim_fitted
        ax1, ax2 = f.plot_PSF(bf)
        assert ax1 is not None
        assert ax2 is not None
        plt.close("all")

    def test_plot_psf_with_true_params(self, small_sim_fitted):
        f, bf, _, _ = small_sim_fitted
        ax1, ax2 = f.plot_PSF(bf, gamma_true=2.5, alpha_true=3.0)
        assert ax1 is not None
        assert ax2 is not None
        plt.close("all")


class TestRobustLoss:
    def test_loss_functions(self):
        x = jnp.array([0.0, 1.0, 2.0, 10.0])
        for factory in [gcp.tukey(), gcp.pseudo_huber(), gcp.cauchy()]:
            result = factory(x)
            assert jnp.all(result >= 0)
            assert float(result[0]) == 0.0

    def test_tukey_saturation(self):
        t = gcp.tukey(c=3.0)
        result = t(jnp.array([0.0, 10.0]))
        assert float(result[1]) == pytest.approx(3.0**2 / 6)

    @pytest.mark.parametrize("loss", [None, lambda x: x**2])
    def test_fit_loss_decreases(self, loss, small_sim_gc):
        _, _, gc = small_sim_gc
        f = gcp.Fitter(gc)
        fit_kwargs = {"show_progress": False, "niter": 50}
        if loss is not None:
            fit_kwargs["loss"] = loss
        bf, extra = f.fit(**fit_kwargs)
        assert bf is not None
        assert float(extra["loss"][-1]) < float(extra["loss"][0]) * 0.99

    def test_fit_adam_tolerance_break(self):
        from gcphotom.jaxfitter import fit_adam

        def quadratic(x):
            return (x["a"] - 5.0) ** 2

        params = {"a": 0.0}
        result, extra = fit_adam(
            quadratic, params, tol=1e30, niter=1000, show_progress=False
        )
        assert len(extra["loss"]) < 50

    def test_tukey_gradient_bounded(self):
        import jax

        x_large = jnp.array(100.0)
        x_small = jnp.array(0.1)

        t = gcp.tukey(c=4.685)
        chi2 = lambda x: x**2

        grad_t = jax.grad(lambda x: jnp.mean(t(x)))
        grad_c = jax.grad(lambda x: jnp.mean(chi2(x)))

        gtl = float(grad_t(x_large))
        gcl = float(grad_c(x_large))
        assert gtl < gcl * 0.01

        gts = float(grad_t(x_small))
        gcs = float(grad_c(x_small))
        ratio = gts / gcs
        assert 0.4 < ratio < 2.0


class TestUncertainty:
    def test_std_errors_finite_positive(self, small_sim_fitted):
        *_, result = small_sim_fitted
        se = result["std_errors"]
        for key in ("flux", "back", "gamma", "alpha"):
            assert np.all(np.isfinite(np.asarray(se[key])))
            assert np.all(np.asarray(se[key]) > 0)

    def test_uncertainty_after_contamination(self, small_sim_gc):
        _, _, gc = small_sim_gc
        f = gcp.Fitter(gc)
        bf, _ = f.fit(show_progress=False, niter=50)
        f.detect_contamination(bf)
        bf2, _ = f.fit(show_progress=False, niter=50, compute_uncertainty=True)
        result = f.results(bf2)
        assert np.all(np.isfinite(np.asarray(result["std_errors"]["gamma"])))


class TestParameterUncertainty:
    def test_linear_model(self):
        from gcphotom.jaxfitter import parameter_uncertainty

        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([0.1, 2.1, 3.9, 6.2, 7.8])
        sigma = jnp.full(5, 0.5)

        params = {"a": jnp.array(2.0), "b": jnp.array(0.0)}

        def wr_fn(p):
            return (y - (p["a"] * x + p["b"])) / sigma

        cov, se = parameter_uncertainty(wr_fn, params)

        w = 1.0 / sigma**2
        A = jnp.column_stack([x, jnp.ones_like(x)])
        expected = jnp.linalg.inv((A * w[:, None]).T @ A)
        wr = wr_fn(params)
        med = jnp.median(wr)
        nmad_val = 1.4826 * jnp.median(jnp.abs(wr - med))
        expected *= float(nmad_val**2)

        np.testing.assert_allclose(cov, expected, atol=1e-6)
        assert float(se["a"]) > 0
        assert float(se["b"]) > 0

    def test_insufficient_data(self):
        from gcphotom.jaxfitter import parameter_uncertainty

        x = jnp.array([1.0, 2.0])
        y = jnp.array([1.0, 2.0])
        sigma = jnp.ones(2)

        params = {"a": jnp.array(1.0), "b": jnp.array(0.0)}

        def wr_fn(p):
            return (y - (p["a"] * x + p["b"])) / sigma

        cov, se = parameter_uncertainty(wr_fn, params)
        assert jnp.all(jnp.isnan(cov))
        assert jnp.isnan(se["a"])
        assert jnp.isnan(se["b"])
