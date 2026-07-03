"""Tests for the Monte Carlo bias/coverage framework."""

import numpy as np
import pytest

import gcphotom as gcp


@pytest.fixture(scope="module")
def mc_config():
    """Simulation config for tests — small and fast."""
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
def mc_results(mc_config):
    """Run 1 MC realization and return results."""
    mc = gcp.montecarlo.MonteCarlo(mc_config, n_realizations=1, seed=42)
    summary = mc.run(verbose=False, show_progress=False)
    return mc, summary


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
        assert cfg.background == 50.0
        assert cfg.fit_kwargs["learning_rate"] == 1e-3


class TestDefaultPipeline:
    def test_returns_expected_keys(self, mc_results):
        mc, _ = mc_results
        res = mc.results[0]
        for key in (
            "fitted",
            "fitted_no_back",
            "fitter",
            "input_cat",
            "psf",
            "aperture",
            "catalog",
        ):
            assert key in res

    def test_fitted_flux_is_finite(self, mc_results):
        mc, _ = mc_results
        for res in mc.results:
            flux = np.asarray(res["fitted"]["flux"])
            assert np.all(np.isfinite(flux[~np.isnan(flux)]))


class TestMonteCarlo:
    def test_run_returns_results(self, mc_results):
        mc, summary = mc_results
        assert summary.realized > 0
        assert summary.total == 1
        assert len(mc.results) == summary.realized

    def test_each_realization_has_different_catalog(self, mc_config):
        mc = gcp.montecarlo.MonteCarlo(mc_config, n_realizations=2, seed=42)
        mc.run(verbose=False, show_progress=False)
        catalogs = [r["catalog"] for r in mc.results]
        for i in range(len(catalogs) - 1):
            assert not np.allclose(catalogs[i]["flux"], catalogs[i + 1]["flux"])

        fluxes = [np.asarray(r["fitted"]["flux"]) for r in mc.results]
        for i in range(len(fluxes) - 1):
            min_len = min(len(fluxes[i]), len(fluxes[i + 1]))
            assert not np.allclose(fluxes[i][:min_len], fluxes[i + 1][:min_len])

    def test_custom_pipeline(self, mc_config):
        def simple_pipeline(image, catalog, cfg):
            seg, det_cat, _, bkg_var = gcp.detect_and_segment(
                image, n_pixels=cfg.n_pixels
            )
            cog = gcp.extract_growth_curves(
                image,
                det_cat,
                segmentation_image=seg,
                background_variance=bkg_var,
                show_progress=False,
            )
            fitter = gcp.Fitter(cog)
            bf, _ = fitter.fit(**cfg.fit_kwargs, show_progress=False)
            fitted = fitter.results(bf)
            input_cat = gcp.cross_match(det_cat, catalog)
            return {
                "fitted": fitted,
                "fitter": fitter,
                "input_cat": input_cat,
                "catalog": catalog,
            }

        mc = gcp.montecarlo.MonteCarlo(
            mc_config, n_realizations=1, seed=42, pipeline=simple_pipeline
        )
        summary = mc.run(verbose=False, show_progress=False)
        assert summary.realized > 0

    def test_failed_realization_is_warned(self, mc_config):
        def failing_pipeline(image, catalog, cfg):
            raise RuntimeError("intentional failure")

        mc = gcp.montecarlo.MonteCarlo(
            mc_config, n_realizations=1, seed=42, pipeline=failing_pipeline
        )
        with pytest.warns(UserWarning):
            summary = mc.run(verbose=False, show_progress=False)
        assert summary.realized == 0


class TestCollectData:
    def test_collect_flux_data(self, mc_results):
        mc, _ = mc_results
        x, truth, estimate, se = gcp.montecarlo._collect_data(
            mc.results,
            get_x=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_truth=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_estimate=lambda r: np.asarray(r["fitted"]["flux"]),
            get_std=lambda r: np.asarray(r["fitted"]["std_errors"]["flux"]),
        )

        assert len(x) > 0
        assert len(x) == len(truth) == len(estimate) == len(se)
        assert np.all(np.isfinite(x))

    def test_no_std_errors(self, mc_results):
        mc, _ = mc_results
        x, truth, estimate, se = gcp.montecarlo._collect_data(
            mc.results,
            get_x=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_truth=lambda r: np.asarray(r["input_cat"]["flux"]),
            get_estimate=lambda r: np.asarray(r["aperture"]["flux"]),
            get_std=None,
        )

        assert se is None
        assert len(x) == len(truth) == len(estimate)


class TestComputeBiasCoverage:
    def test_default_estimators(self, mc_results, mc_config):
        mc, _ = mc_results
        estimators = gcp.montecarlo.build_default_estimators(mc_config)
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results, estimators=estimators, nbins=3, sigma_levels=(1.0,)
        )

        assert "GC (est. back)" in stats
        assert "GC (fixed back)" in stats
        s = stats["GC (est. back)"]
        for key in ("xbins", "bias", "bias_err", "rms", "coverage_1.0sigma"):
            assert key in s

    def test_nuisance_parameters(self, mc_results, mc_config):
        mc, _ = mc_results
        estimators = gcp.montecarlo.build_default_estimators(mc_config)
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results, estimators=estimators, nbins=3, sigma_levels=(1.0,)
        )

        assert "Background" in stats
        assert "Gamma" in stats
        assert "Alpha" in stats

    def test_bias_and_coverage_are_reasonable(self, mc_results, mc_config):
        mc, _ = mc_results
        estimators = gcp.montecarlo.build_default_estimators(mc_config)
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results, estimators=estimators, nbins=3, sigma_levels=(1.0,)
        )
        bias = stats["GC (est. back)"]["bias"]
        cov = stats["GC (est. back)"]["coverage_1.0sigma"]
        rms = stats["GC (est. back)"]["rms"]
        assert np.any(np.abs(bias) < 50)
        assert np.all((cov >= 0) & (cov <= 1))
        assert np.all(rms >= 0)


class TestPlotBiasCoverage:
    def test_returns_axes(self, mc_results, mc_config, tmp_path):
        mc, _ = mc_results
        estimators = gcp.montecarlo.build_default_estimators(mc_config)
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results, estimators=estimators, nbins=3, sigma_levels=(1.0,)
        )

        import matplotlib.pyplot as plt

        ax_bias, ax_cov = gcp.montecarlo.plot_bias_coverage(stats, sigma_level=1.0)
        assert ax_bias is not None
        assert ax_cov is not None

        fig = ax_bias.get_figure()
        fig.savefig(str(tmp_path / "test_plot.png"))
        plt.close(fig)
