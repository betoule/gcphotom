import numpy as np
import pytest
from gcphotom.simulator import (
    make_moffat_psf,
    make_source_catalog,
    simulate_image,
)


class TestMakeSourceCatalog:
    def test_catalog_length(self):
        cat = make_source_catalog(100, shape=(256, 256), seed=42)
        assert len(cat) == 100

    def test_positions_within_bounds(self):
        shape = (256, 256)
        margin = 20
        cat = make_source_catalog(50, shape=shape, margin=margin, seed=42)
        assert np.all(cat["x"] >= margin)
        assert np.all(cat["x"] < shape[1] - margin)
        assert np.all(cat["y"] >= margin)
        assert np.all(cat["y"] < shape[0] - margin)

    def test_no_position_overlap(self):
        cat = make_source_catalog(50, shape=(256, 256), min_sep=5, seed=42)
        for i in range(len(cat)):
            for j in range(i + 1, len(cat)):
                dx = cat["x"][i] - cat["x"][j]
                dy = cat["y"][i] - cat["y"][j]
                dist = np.sqrt(dx**2 + dy**2)
                assert dist >= 5

    def test_flux_range(self):
        cat = make_source_catalog(100, shape=(256, 256), seed=42)
        assert np.all(cat["flux"] >= 100)
        assert np.all(cat["flux"] <= 1e6)

    def test_flux_log_uniform(self):
        from scipy.stats import kstest

        cat = make_source_catalog(5000, shape=(2048, 2048), seed=42)
        log_flux = np.log10(cat["flux"])
        _, pvalue = kstest(log_flux, "uniform", args=(2, 4))
        assert pvalue > 0.01


class TestMakeMoffatPSF:
    def test_model_type(self):
        from astropy.modeling.models import Moffat2D

        psf = make_moffat_psf(2.5, 3.0)
        assert isinstance(psf, Moffat2D)

    def test_model_parameters(self):
        psf = make_moffat_psf(2.5, 3.0)
        assert psf.gamma == 2.5
        assert psf.alpha == 3.0
        assert psf.amplitude == 1

    def test_psf_integral(self):
        alpha, beta = 2.5, 3.0
        psf = make_moffat_psf(alpha, beta)
        ny, nx = 101, 101
        yy, xx = np.mgrid[-ny // 2 : ny // 2 + 1, -nx // 2 : nx // 2 + 1]
        values = psf(xx, yy)
        integral = values.sum()
        expected = psf(0, 0) * alpha**2 * np.pi / (beta - 1)
        assert 0.8 < integral / expected < 1.2


class TestSimulateImage:
    def test_image_shape(self):
        cat = make_source_catalog(10, shape=(128, 128), seed=42)
        img = simulate_image((128, 128), cat, alpha=2.5, beta=3.0, seed=42)
        assert img.shape == (128, 128)

    def test_no_nan(self):
        cat = make_source_catalog(10, shape=(128, 128), seed=42)
        img = simulate_image((128, 128), cat, alpha=2.5, beta=3.0, seed=42)
        assert np.all(np.isfinite(img))

    def test_background_level(self):
        cat = make_source_catalog(1, shape=(256, 256), seed=42)
        cat["x"][0] = 128
        cat["y"][0] = 128
        img = simulate_image(
            (256, 256), cat, alpha=2.5, beta=3.0, background=500, seed=42
        )
        edge_mask = np.ones((256, 256), dtype=bool)
        edge_mask[50:206, 50:206] = False
        bg_region = img[edge_mask]
        assert np.abs(np.median(bg_region) - 500) < 50

    def test_noise_statistics(self):
        cat = make_source_catalog(1, shape=(256, 256), seed=42)
        cat["x"][0] = 128
        cat["y"][0] = 128
        img = simulate_image(
            (256, 256),
            cat,
            alpha=2.5,
            beta=3.0,
            background=100,
            read_noise=5,
            seed=42,
        )
        edge_mask = np.ones((256, 256), dtype=bool)
        edge_mask[50:206, 50:206] = False
        bg_region = img[edge_mask]
        expected_std = np.sqrt(100 + 5**2)
        assert np.abs(np.std(bg_region) - expected_std) < 10

    def test_flux_conservation(self):
        alpha, beta = 2.5, 3.0
        cat = make_source_catalog(5, shape=(512, 512), min_sep=15, seed=42)
        for i in range(len(cat)):
            cat["x"][i] = 100 + i * 80
            cat["y"][i] = 256
        img = simulate_image(
            (512, 512), cat, alpha=alpha, beta=beta, background=0, seed=42
        )
        total_injected = cat["flux"].sum()
        total_image = img.sum()
        ratio = total_image / total_injected
        assert 0.8 < ratio < 1.3

    def test_full_simulation(self):
        cat = make_source_catalog(100, shape=(256, 256), seed=42)
        img = simulate_image(
            (256, 256),
            cat,
            alpha=2.5,
            beta=3.0,
            background=200,
            read_noise=5,
            seed=42,
        )
        assert img.shape == (256, 256)
        assert np.all(np.isfinite(img))
        assert img.min() >= 0
