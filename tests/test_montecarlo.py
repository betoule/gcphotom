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
        fit_kwargs={"learning_rate": 1e-2, "niter": 100},
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


class TestSimulationConfig:
    def test_defaults(self):
        cfg = gcp.montecarlo.SimulationConfig()
        assert cfg.n_sources == 1000
        assert cfg.gamma == 3.0
        assert cfg.alpha == 3.0
        assert cfg.background == 100.0
        assert cfg.read_noise == 5.0

    def test_custom(self):
        cfg = gcp.montecarlo.SimulationConfig(
            n_sources=500,
            gamma=2.5,
            alpha=2.0,
            background=50.0,
            read_noise=3.0,
            fit_kwargs={"learning_rate": 1e-3, "niter": 500},
        )
        assert cfg.n_sources == 500
        assert cfg.gamma == 2.5
        assert cfg.alpha == 2.0
        assert cfg.fit_kwargs["learning_rate"] == 1e-3


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

    def _run_estimator(self, estimator, mc_config, sim_data):
        image, catalog = sim_data
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
        return estimator(image, detections, cog)

    def test_gc_estimator(self, mc_config, sim_data):
        estimator = partial(
            gcp.montecarlo.gc_estimator, fit_kwargs=mc_config.fit_kwargs
        )
        result = self._run_estimator(estimator, mc_config, sim_data)
        assert "best_fit" in result
        assert "flux" in result["best_fit"]
        assert "uncertainty" in result
        assert "extra" in result
        assert "estimation_time" in result["extra"]
        assert np.ndim(result["best_fit"]["flux"]) == 1
        assert np.isfinite(result["best_fit"]["flux"]).any()

    def test_gc_fixed_back_estimator(self, mc_config, sim_data):
        estimator = partial(
            gcp.montecarlo.gc_fixed_back_estimator, fit_kwargs=mc_config.fit_kwargs
        )
        result = self._run_estimator(estimator, mc_config, sim_data)
        assert "flux" in result["best_fit"]
        assert np.isfinite(result["best_fit"]["flux"]).any()

    def test_psf_estimator(self, mc_config, sim_data):
        estimator = partial(
            gcp.montecarlo.psf_estimator, background=mc_config.background
        )
        result = self._run_estimator(estimator, mc_config, sim_data)
        assert "flux" in result["best_fit"]
        assert result["uncertainty"] is None

    def test_aperture_estimator(self, mc_config, sim_data):
        estimator = partial(
            gcp.montecarlo.aperture_estimator, fit_kwargs=mc_config.fit_kwargs
        )
        result = self._run_estimator(estimator, mc_config, sim_data)
        assert "flux" in result["best_fit"]
        assert result["uncertainty"] is None

    def test_default_estimators_builds_dict(self, mc_config):
        ests = gcp.montecarlo.default_estimators(mc_config)
        for name in ("GC", "GC (fixed back)", "PSF", "Aperture + AC"):
            assert name in ests
            assert callable(ests[name])


class TestRunPipeline:
    def test_returns_expected_keys(self, mc_config, sim_data):
        image, catalog = sim_data
        estimators = gcp.montecarlo.default_estimators(mc_config)
        result = gcp.montecarlo.run_pipeline(image, catalog, mc_config, estimators)
        assert "sim_cat" in result
        assert "det_cat" in result
        assert "params" in result
        for name in estimators:
            assert name in result
            assert "best_fit" in result[name]
            assert "extra" in result[name]

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

    def test_flux_vectors_match_det_cat_length(self, mc_config, sim_data):
        image, catalog = sim_data
        ests = gcp.montecarlo.default_estimators(mc_config)
        result = gcp.montecarlo.run_pipeline(image, catalog, mc_config, ests)
        n_det = len(result["det_cat"])
        for name in ests:
            flux = result[name]["best_fit"]["flux"]
            assert len(flux) == n_det, f"{name}: {len(flux)} != {n_det}"


class TestMonteCarlo:
    def test_run_returns_results(self, mc_results):
        mc, results = mc_results
        assert len(results) > 0
        assert len(mc.results) == len(results)

    def test_each_realization_has_different_catalog(self, mc_config):
        mc = gcp.montecarlo.MonteCarlo(mc_config, n_realizations=2, seed=42)
        results = mc.run(verbose=False, show_progress=False)
        fluxes = [np.asarray(r["GC"]["best_fit"]["flux"]) for r in results]
        for i in range(len(fluxes) - 1):
            min_len = min(len(fluxes[i]), len(fluxes[i + 1]))
            assert not np.allclose(fluxes[i][:min_len], fluxes[i + 1][:min_len])

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


class TestComputeFluxBias:
    def test_returns_expected_keys(self, mc_results):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_flux_bias(
            mc.results, estimators=["GC", "PSF"], nbins=3
        )
        assert "GC" in stats
        assert "PSF" in stats
        for s in stats.values():
            assert "xbins" in s
            assert "bias" in s
            assert "bias_err" in s

    def test_auto_detect_estimators(self, mc_results):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_flux_bias(mc.results, nbins=3)
        for name in ("GC", "GC (fixed back)", "PSF", "Aperture + AC"):
            assert name in stats

    def test_bias_is_reasonable(self, mc_results):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_flux_bias(mc.results, estimators=["GC"], nbins=3)
        bias = stats["GC"]["bias"]
        assert np.any(np.abs(bias) < 50)


class TestPlotFluxBias:
    def test_returns_axes(self, mc_results, tmp_path):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_flux_bias(mc.results, estimators=["GC"], nbins=3)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = gcp.montecarlo.plot_flux_bias(stats)
        assert ax is not None
        fig = ax.get_figure()
        fig.savefig(str(tmp_path / "test_plot.png"))
        plt.close(fig)


class TestSaveLoad:
    def test_save_and_load(self, mc_results, tmp_path):
        mc, _ = mc_results
        path = str(tmp_path / "mc_results.pkl")
        written = gcp.montecarlo.save_results(path, mc.results)
        assert written.endswith(".pkl")

        loaded = gcp.montecarlo.load_results(path)
        assert len(loaded) == len(mc.results)
        assert set(loaded[0].keys()) == set(mc.results[0].keys())

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

    def test_loaded_results_compute_flux_bias(self, mc_results, tmp_path):
        mc, _ = mc_results
        path = str(tmp_path / "mc_flux.pkl")
        gcp.montecarlo.save_results(path, mc.results)
        loaded = gcp.montecarlo.load_results(path)

        stats = gcp.montecarlo.compute_flux_bias(loaded, estimators=["GC"], nbins=3)
        assert "GC" in stats
        assert "xbins" in stats["GC"]
