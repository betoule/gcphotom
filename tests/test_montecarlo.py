"""Tests for the Monte Carlo bias/coverage framework."""

import os

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
    """Run 1 MC realization and return results list."""
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
        assert cfg.background == 50.0
        assert cfg.fit_kwargs["learning_rate"] == 1e-3


class TestDefaultPipeline:
    def test_returns_expected_keys(self, mc_results):
        mc, _ = mc_results
        res = mc.results[0]
        assert "params" in res
        for key in (
            "GC (est. back)",
            "GC (fixed back)",
            "PSF photometry",
            "Aperture + AC",
            "Background",
            "Gamma",
            "Alpha",
        ):
            assert key in res

    def test_estimator_entry_structure(self, mc_results):
        mc, _ = mc_results
        res = mc.results[0]
        for key in (
            "GC (est. back)",
            "GC (fixed back)",
            "PSF photometry",
            "Aperture + AC",
            "Background",
        ):
            entry = res[key]
            assert "best_fit" in entry
            assert "uncertainty" in entry
            assert "extra" in entry
            assert "truth" in entry["extra"]
            assert "bias_scale" in entry["extra"]
            assert entry["extra"]["bias_scale"] == 100.0

        for key in ("Gamma", "Alpha"):
            entry = res[key]
            assert "best_fit" in entry
            assert "uncertainty" in entry
            assert "extra" in entry
            assert "truth" in entry["extra"]
            assert np.ndim(entry["best_fit"]) == 0

    def test_fitted_flux_is_finite(self, mc_results):
        mc, _ = mc_results
        for res in mc.results:
            flux = np.asarray(res["GC (est. back)"]["best_fit"])
            assert np.all(np.isfinite(flux[~np.isnan(flux)]))


class TestMonteCarlo:
    def test_run_returns_results(self, mc_results):
        mc, results = mc_results
        assert len(results) > 0
        assert len(mc.results) == len(results)

    def test_each_realization_has_different_catalog(self, mc_config):
        mc = gcp.montecarlo.MonteCarlo(mc_config, n_realizations=2, seed=42)
        results = mc.run(verbose=False, show_progress=False)
        # Each result should have different best-fit fluxes
        fluxes = [np.asarray(r["GC (est. back)"]["best_fit"]) for r in results]
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
            flux_truth = np.asarray(input_cat["flux"])
            return {
                "params": cfg,
                "MyFlux": {
                    "best_fit": np.asarray(fitted["flux"]),
                    "uncertainty": None,
                    "extra": {"truth": flux_truth, "bias_scale": 100.0},
                },
                "MyGamma": {
                    "best_fit": fitted["gamma"],
                    "uncertainty": fitted.get("std_errors", {}).get("gamma"),
                    "extra": {"truth": cfg.gamma, "bias_scale": 100.0},
                },
            }

        mc = gcp.montecarlo.MonteCarlo(
            mc_config, n_realizations=1, seed=42, pipeline=simple_pipeline
        )
        results = mc.run(verbose=False, show_progress=False)
        assert len(results) > 0
        assert "MyFlux" in results[0]
        assert "MyGamma" in results[0]

    def test_failed_realization_is_warned(self, mc_config):
        def failing_pipeline(image, catalog, cfg):
            raise RuntimeError("intentional failure")

        mc = gcp.montecarlo.MonteCarlo(
            mc_config, n_realizations=1, seed=42, pipeline=failing_pipeline
        )
        with pytest.warns(UserWarning):
            results = mc.run(verbose=False, show_progress=False)
        assert len(results) == 0


class TestComputeBiasCoverage:
    def test_flux_estimators(self, mc_results):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["GC (est. back)", "GC (fixed back)"],
            nbins=3,
            sigma_levels=(1.0,),
        )

        assert "GC (est. back)" in stats
        assert "GC (fixed back)" in stats
        s = stats["GC (est. back)"]
        for key in ("xbins", "bias", "bias_err", "rms", "coverage_1.0sigma"):
            assert key in s

    def test_background_estimator(self, mc_results):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["Background"],
            nbins=3,
            sigma_levels=(1.0,),
        )

        assert "Background" in stats
        s = stats["Background"]
        for key in ("xbins", "bias", "bias_err", "rms", "coverage_1.0sigma"):
            assert key in s

    def test_bias_and_coverage_are_reasonable(self, mc_results):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["GC (est. back)"],
            nbins=3,
            sigma_levels=(1.0,),
        )
        bias = stats["GC (est. back)"]["bias"]
        cov = stats["GC (est. back)"]["coverage_1.0sigma"]
        rms = stats["GC (est. back)"]["rms"]
        assert np.any(np.abs(bias) < 50)
        assert np.all((cov >= 0) & (cov <= 1))
        assert np.all(rms >= 0)

    def test_auto_detect_estimators(self, mc_results):
        mc, _ = mc_results
        # Without specifying estimators, should auto-detect per-source ones
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results, nbins=3, sigma_levels=(1.0,)
        )
        # Should contain flux & background estimators, but not Gamma/Alpha
        for name in ("GC (est. back)", "GC (fixed back)", "Background"):
            assert name in stats, f"Missing {name}"


class TestPlotBiasCoverage:
    def test_returns_axes(self, mc_results, tmp_path):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["GC (est. back)"],
            nbins=3,
            sigma_levels=(1.0,),
        )

        import matplotlib.pyplot as plt

        ax_bias, ax_cov = gcp.montecarlo.plot_bias_coverage(stats, sigma_level=1.0)
        assert ax_bias is not None
        assert ax_cov is not None

        fig = ax_bias.get_figure()
        fig.savefig(str(tmp_path / "test_plot.png"))
        plt.close(fig)


class TestNuisanceStats:
    def test_compute_nuisance_stats(self, mc_results):
        mc, _ = mc_results
        nstats = gcp.montecarlo.compute_nuisance_stats(
            mc.results, estimators=["Gamma", "Alpha"]
        )

        assert "Gamma" in nstats
        assert "Alpha" in nstats
        for label in ("Gamma", "Alpha"):
            s = nstats[label]
            for key in (
                "truth",
                "estimates",
                "bias",
                "mean_bias",
                "std_bias",
                "rms_bias",
            ):
                assert key in s
            assert np.isfinite(s["mean_bias"])

    def test_auto_detect_nuisance(self, mc_results):
        mc, _ = mc_results
        nstats = gcp.montecarlo.compute_nuisance_stats(mc.results)
        assert "Gamma" in nstats
        assert "Alpha" in nstats

    def test_plot_nuisance_summary(self, mc_results, tmp_path):
        mc, _ = mc_results
        nstats = gcp.montecarlo.compute_nuisance_stats(
            mc.results, estimators=["Gamma", "Alpha"]
        )

        import matplotlib.pyplot as plt

        fig = gcp.montecarlo.plot_nuisance_summary(nstats)
        assert fig is not None
        fig.savefig(str(tmp_path / "test_nuisance_plot.png"))
        plt.close(fig)

    def test_plot_background_bias(self, mc_results, tmp_path):
        mc, _ = mc_results
        bg_stats = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["Background"],
            nbins=3,
            sigma_levels=(1.0,),
        )

        import matplotlib.pyplot as plt

        ax = gcp.montecarlo.plot_background_bias(bg_stats)
        assert ax is not None
        fig = ax.get_figure()
        fig.savefig(str(tmp_path / "test_background_plot.png"))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Save / load round-trip tests
# ---------------------------------------------------------------------------


class TestSaveLoadResults:
    """Round-trip save/load of raw MC results via pickle."""

    def test_save_and_load(self, mc_results, tmp_path):
        mc, _ = mc_results
        path = str(tmp_path / "mc_results.pkl")
        written = gcp.montecarlo.save_results(path, mc.results)
        assert written.endswith(".pkl")

        loaded = gcp.montecarlo.load_results(path)
        assert len(loaded) == len(mc.results)
        assert set(loaded[0].keys()) == set(mc.results[0].keys())

    def test_save_and_load_default_extension(self, mc_results, tmp_path):
        mc, _ = mc_results
        path = str(tmp_path / "mc_results")
        gcp.montecarlo.save_results(path, mc.results)
        assert os.path.exists(str(tmp_path / "mc_results.pkl"))

        loaded = gcp.montecarlo.load_results(path)
        assert len(loaded) == len(mc.results)

    def test_load_results_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gcp.montecarlo.load_results(str(tmp_path / "nonexistent.pkl"))

    def test_loaded_results_compute_bias_coverage(self, mc_results, tmp_path):
        mc, _ = mc_results
        path = str(tmp_path / "mc_flux.pkl")
        gcp.montecarlo.save_results(path, mc.results)
        loaded = gcp.montecarlo.load_results(path)

        stats = gcp.montecarlo.compute_bias_coverage(
            loaded, estimators=["GC (est. back)"], nbins=3, sigma_levels=(1.0,)
        )
        assert "GC (est. back)" in stats
        assert "xbins" in stats["GC (est. back)"]

    def test_loaded_results_compute_nuisance(self, mc_results, tmp_path):
        mc, _ = mc_results
        path = str(tmp_path / "mc_nuisance.pkl")
        gcp.montecarlo.save_results(path, mc.results)
        loaded = gcp.montecarlo.load_results(path)

        nstats = gcp.montecarlo.compute_nuisance_stats(
            loaded, estimators=["Gamma", "Alpha"]
        )
        assert "Gamma" in nstats
        assert np.isfinite(nstats["Gamma"]["mean_bias"])

    def test_consistency_original_vs_loaded(self, mc_results, tmp_path):
        mc, _ = mc_results
        nbins = 3
        sigma_levels = (1.0, 2.0)

        original = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["GC (est. back)"],
            nbins=nbins,
            sigma_levels=sigma_levels,
        )

        path = str(tmp_path / "mc_flux_cons.pkl")
        gcp.montecarlo.save_results(path, mc.results)
        loaded = gcp.montecarlo.load_results(path)
        recomputed = gcp.montecarlo.compute_bias_coverage(
            loaded,
            estimators=["GC (est. back)"],
            nbins=nbins,
            sigma_levels=sigma_levels,
        )

        for label in original:
            for key in original[label]:
                assert np.allclose(
                    recomputed[label][key], original[label][key], equal_nan=True
                )


class TestSaveLoadStats:
    """Round-trip save/load of computed stats dicts via pickle."""

    def test_save_and_load_bias_coverage(self, mc_results, tmp_path):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["GC (est. back)"],
            nbins=3,
            sigma_levels=(1.0, 2.0),
        )

        path = str(tmp_path / "bias_stats.pkl")
        written = gcp.montecarlo.save_stats(path, stats)
        assert written.endswith(".pkl")

        loaded = gcp.montecarlo.load_stats(path)
        assert set(loaded.keys()) == set(stats.keys())
        for label in stats:
            for key in stats[label]:
                assert np.allclose(
                    loaded[label][key], stats[label][key], equal_nan=True
                )

    def test_save_and_load_nuisance_stats(self, mc_results, tmp_path):
        mc, _ = mc_results
        nstats = gcp.montecarlo.compute_nuisance_stats(
            mc.results, estimators=["Gamma", "Alpha"]
        )

        path = str(tmp_path / "nuisance_stats.pkl")
        gcp.montecarlo.save_stats(path, nstats)

        loaded = gcp.montecarlo.load_stats(path)
        assert set(loaded.keys()) == set(nstats.keys())
        for label in nstats:
            for key in nstats[label]:
                if key in ("mean_bias", "std_bias", "rms_bias"):
                    assert loaded[label][key] == nstats[label][key]
                else:
                    assert np.allclose(
                        loaded[label][key], nstats[label][key], equal_nan=True
                    )

    def test_load_stats_default_extension(self, mc_results, tmp_path):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["GC (est. back)"],
            nbins=3,
            sigma_levels=(1.0,),
        )

        path = str(tmp_path / "noext")
        gcp.montecarlo.save_stats(path, stats)
        loaded = gcp.montecarlo.load_stats(path)
        assert set(loaded.keys()) == set(stats.keys())

    def test_saved_stats_can_be_plotted(self, mc_results, tmp_path):
        mc, _ = mc_results
        stats = gcp.montecarlo.compute_bias_coverage(
            mc.results,
            estimators=["GC (est. back)"],
            nbins=3,
            sigma_levels=(1.0,),
        )

        path = str(tmp_path / "plot_test.pkl")
        gcp.montecarlo.save_stats(path, stats)
        loaded = gcp.montecarlo.load_stats(path)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax_bias, ax_cov = gcp.montecarlo.plot_bias_coverage(loaded, sigma_level=1.0)
        assert ax_bias is not None
        plt.close(ax_bias.get_figure())

    def test_load_stats_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gcp.montecarlo.load_stats(str(tmp_path / "nonexistent.pkl"))

    def test_save_stats_empty(self, tmp_path):
        path = str(tmp_path / "empty.pkl")
        gcp.montecarlo.save_stats(path, {})
        loaded = gcp.montecarlo.load_stats(path)
        assert loaded == {}
