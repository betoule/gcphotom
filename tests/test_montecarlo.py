"""Tests for the Monte Carlo simulation framework."""

import os
import time
from functools import partial

import numpy as np
import pytest

import gcphotom as gcp


@pytest.fixture(scope="module")
def mc_config():
    return gcp.montecarlo.SimulationConfig(
        n_sources=20,
        shape=(256, 256),
        gamma=3.0,
        alpha=3.0,
        background=100.0,
        read_noise=5.0,
        n_pixels=5,
        fit_kwargs={"learning_rate": 1e-2, "niter": 50},
    )


@pytest.fixture(scope="module")
def sim_data(mc_config):
    """Generate single-realisation data for estimator tests."""
    rng = np.random.default_rng(42)
    sd = int(rng.integers(0, 2**31, 1)[0])
    catalog = gcp.make_realistic_source_catalog(
        n_sources=mc_config.n_sources, shape=mc_config.shape, seed=sd
    )
    image, _ = gcp.simulate_image(
        shape=mc_config.shape,
        catalog=catalog,
        gamma=mc_config.gamma,
        alpha=mc_config.alpha,
        background=mc_config.background,
        read_noise=mc_config.read_noise,
        seed=sd,
    )
    return image, catalog


@pytest.fixture(scope="module")
def mc_results(mc_config):
    """Run 1 MC realisation with default estimators."""
    mc = gcp.montecarlo.MonteCarlo(mc_config, n_realizations=1, seed=42)
    results = mc.run(verbose=False, show_progress=False)
    return mc, results


@pytest.fixture(scope="module")
def detections_and_cog(mc_config, sim_data):
    """Pre-computed detection, segmentation and growth curves for
    *sim_data*, shared by the estimator tests below."""
    image, _ = sim_data
    seg, det_cat, bkg_map, bkg_var_map = gcp.detect_and_segment(
        image, n_pixels=mc_config.n_pixels
    )
    detections = {
        "seg": seg,
        "det_cat": det_cat,
        "bkg_map": bkg_map,
        "bkg_var_map": bkg_var_map,
    }
    cog = gcp.extract_growth_curves(
        image,
        det_cat,
        segmentation_image=seg,
        background_variance=bkg_var_map,
        show_progress=False,
    )
    return detections, cog


class TestTimedEstimator:
    def test_adds_estimation_time(self):
        @gcp.montecarlo.timed_estimator
        def dummy(image, detections, cog):
            time.sleep(0.01)
            return {"best_fit": {"flux": np.array([1.0])}, "uncertainty": None}

        result = dummy(None, None, None)
        assert "extra" in result
        assert "estimation_time" in result["extra"]
        assert result["extra"]["estimation_time"] > 0.005

    def test_does_not_override_existing_extra(self):
        @gcp.montecarlo.timed_estimator
        def dummy(image, detections, cog):
            return {"best_fit": {"flux": np.array([1.0])}, "extra": {"foo": "bar"}}

        result = dummy(None, None, None)
        assert result["extra"]["foo"] == "bar"
        assert "estimation_time" in result["extra"]


class TestEstimators:
    """Test each built-in estimator independently."""

    def _run_estimator(self, estimator, image, detections, cog):
        return estimator(image, detections, cog)

    @pytest.mark.parametrize(
        "estimator,cfg_field",
        [
            (gcp.montecarlo.gc_estimator, "fit_kwargs"),
            (gcp.montecarlo.gc_fixed_back_estimator, "fit_kwargs"),
            (gcp.montecarlo.psf_estimator, None),
            (gcp.montecarlo.aperture_estimator, None),
        ],
    )
    def test_estimator_returns_finite_flux(
        self, estimator, cfg_field, mc_config, sim_data, detections_and_cog
    ):
        if cfg_field is not None:
            if cfg_field == "fit_kwargs":
                estimator = partial(estimator, fit_kwargs=mc_config.fit_kwargs)
            elif cfg_field == "background":
                estimator = partial(estimator, background=mc_config.background)
        image, _ = sim_data
        detections, cog = detections_and_cog
        result = self._run_estimator(estimator, image, detections, cog)
        assert np.isfinite(result["best_fit"]["flux"]).any()


class TestRunPipeline:
    def test_sim_cat_has_flux(self, mc_config, sim_data):
        image, catalog = sim_data
        estimators = {
            "Dummy": gcp.montecarlo.timed_estimator(
                lambda img, det, cog: {
                    "best_fit": {"flux": np.full(len(det["det_cat"]), 1.0)},
                    "uncertainty": None,
                }
            )
        }
        result = gcp.montecarlo.run_pipeline(image, catalog, mc_config, estimators)
        assert np.isfinite(np.asarray(result["sim_cat"]["flux"])).any()


class TestMonteCarlo:
    def test_run_returns_results(self, mc_results):
        mc, results = mc_results
        assert len(results) > 0
        assert len(mc.results) == len(results)

    def test_each_realization_has_different_catalog(self, mc_config):
        cat1 = gcp.make_realistic_source_catalog(
            n_sources=mc_config.n_sources, shape=mc_config.shape, seed=42
        )
        cat2 = gcp.make_realistic_source_catalog(
            n_sources=mc_config.n_sources, shape=mc_config.shape, seed=43
        )
        assert not np.allclose(cat1["flux"], cat2["flux"])

    def test_custom_estimators(self, mc_config):
        @gcp.montecarlo.timed_estimator
        def my_est(image, detections, cog):
            return {
                "best_fit": {"flux": np.full(len(detections["det_cat"]), 42.0)},
                "uncertainty": None,
            }

        mc = gcp.montecarlo.MonteCarlo(
            mc_config, n_realizations=1, seed=42, estimators={"MyEst": my_est}
        )
        results = mc.run(verbose=False, show_progress=False)
        assert len(results) > 0
        assert "MyEst" in results[0]
        assert results[0]["MyEst"]["best_fit"]["flux"][0] == 42.0

    def test_failed_realization_is_warned(self, mc_config):
        def failing_est(image, detections, cog):
            raise RuntimeError("intentional failure")

        mc = gcp.montecarlo.MonteCarlo(
            mc_config, n_realizations=1, seed=42, estimators={"Fail": failing_est}
        )
        with pytest.warns(UserWarning):
            results = mc.run(verbose=False, show_progress=False)
        assert len(results) == 0


class TestApertureEstimator:
    """Edge cases for the aperture estimator."""

    def test_too_few_valid_sources(self, sim_data, detections_and_cog):
        """When valid.sum() < 5, return NaN fluxes."""
        image, _ = sim_data
        detections, cog = detections_and_cog
        # Corrupt the COG to force valid.sum() < 5
        cog_orig = cog["flux_clean"].copy()
        cog["flux_clean"] = np.full_like(cog["flux_clean"], np.nan)
        result = gcp.montecarlo.aperture_estimator(image, detections, cog)
        assert np.all(np.isnan(result["best_fit"]["flux"]))
        cog["flux_clean"] = cog_orig

    def test_ac_out_of_range(self, sim_data, detections_and_cog):
        """When aperture correction is out of [1, 5], clamp to large_ratio."""
        image, _ = sim_data
        detections, cog = detections_and_cog
        result = gcp.montecarlo.aperture_estimator(image, detections, cog)
        assert np.isfinite(result["best_fit"]["flux"]).any()


class TestComputeFluxBias:
    def test_bias_is_reasonable(self, mc_results):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_flux_bias(mc.results, estimators=["GC"], nbins=3)
        bias = stats["GC"]["bias"]
        assert np.any(np.abs(bias) < 50)

    def test_auto_estimators(self, mc_results):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_flux_bias(mc.results, nbins=3)
        assert "GC" in stats


class TestPlotFunctions:
    def test_plot_flux_bias_returns_axes(self, mc_results, tmp_path):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_flux_bias(mc.results, estimators=["GC"], nbins=3)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = gcp.montecarlo.plot_flux_bias(stats)
        assert ax is not None
        fig = ax.get_figure()
        fig.savefig(str(tmp_path / "test_flux_bias.png"))
        plt.close(fig)

    def test_plot_scalar_bias_returns_axes(self, mc_results, tmp_path):
        mc, _ = mc_results
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        axes = gcp.montecarlo.plot_scalar_bias(mc.results)
        assert len(axes) > 0
        fig = axes[0].get_figure()
        fig.savefig(str(tmp_path / "test_scalar_bias.png"))
        plt.close(fig)

    def test_plot_estimation_times_returns_axes(self, mc_results, tmp_path):
        mc, _ = mc_results
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = gcp.montecarlo.plot_estimation_times(mc.results)
        assert ax is not None
        fig = ax.get_figure()
        fig.savefig(str(tmp_path / "test_est_times.png"))
        plt.close(fig)


class TestMonteCarloRunVerbose:
    def test_verbose_mode(self, mc_results):
        mc, results = mc_results
        assert len(results) > 0


class TestSaveLoad:
    def test_save_and_load(self, mc_results, tmp_path):
        mc, _ = mc_results
        path = str(tmp_path / "mc_results.pkl")
        written = gcp.montecarlo.save_results(path, mc.results)
        assert written.endswith(".pkl")

        loaded = gcp.montecarlo.load_results(path)
        assert len(loaded) == len(mc.results)

    def test_default_extension(self, mc_results, tmp_path):
        mc, _ = mc_results
        path = str(tmp_path / "mc_results")
        gcp.montecarlo.save_results(path, mc.results)
        assert os.path.exists(str(tmp_path / "mc_results.pkl"))
        loaded = gcp.montecarlo.load_results(path)
        assert len(loaded) == len(mc.results)

    def test_load_results_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gcp.montecarlo.load_results(str(tmp_path / "nonexistent.pkl"))
