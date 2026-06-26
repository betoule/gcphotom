import numpy as np

import gcphotom as gcp
from gcphotom.simulator import make_realistic_source_catalog, simulate_image


class TestMakeSourceCatalog:
    def test_positions_within_bounds(self):
        shape = (256, 256)
        margin = 20
        cat = make_realistic_source_catalog(50, shape=shape, margin=margin, seed=42)
        assert np.all(cat["x"] >= margin)
        assert np.all(cat["x"] < shape[1] - margin)
        assert np.all(cat["y"] >= margin)
        assert np.all(cat["y"] < shape[0] - margin)

    def test_flux_range(self):
        cat = make_realistic_source_catalog(100, shape=(256, 256), seed=42)
        assert np.all(cat["flux"] >= 100)
        assert np.all(cat["flux"] <= 1e6)

    def test_flux_log_uniform_mean(self):
        cat = make_realistic_source_catalog(5000, shape=(2048, 2048), seed=42)
        log_flux = np.log10(cat["flux"])
        expected_mean = (np.log10(100) + np.log10(1e6)) / 2
        assert abs(np.mean(log_flux) - expected_mean) < 0.05


class TestSimulateImage:
    def test_image_shape(self):
        cat = make_realistic_source_catalog(10, shape=(128, 128), seed=42)
        img, _ = simulate_image((128, 128), cat, gamma=2.5, alpha=3.0, seed=42)
        assert img.shape == (128, 128)

    def test_auto_generate_catalog(self):
        img, cat = simulate_image(seed=42)
        assert img.ndim == 2
        assert len(cat) > 0

    def test_no_nan(self):
        cat = make_realistic_source_catalog(10, shape=(128, 128), seed=42)
        img, _ = simulate_image((128, 128), cat, gamma=2.5, alpha=3.0, seed=42)
        assert np.all(np.isfinite(img))

    def test_background_level(self):
        cat = make_realistic_source_catalog(1, shape=(256, 256), seed=42)
        cat["x"][0] = 128
        cat["y"][0] = 128
        img, _ = simulate_image(
            (256, 256), cat, gamma=2.5, alpha=3.0, background=500, seed=42
        )
        edge_mask = np.ones((256, 256), dtype=bool)
        edge_mask[50:206, 50:206] = False
        bg_region = img[edge_mask]
        assert np.abs(np.median(bg_region) - 500) < 50

    def test_noise_statistics(self):
        cat = make_realistic_source_catalog(1, shape=(256, 256), seed=42)
        cat["x"][0] = 128
        cat["y"][0] = 128
        img, _ = simulate_image(
            (256, 256),
            cat,
            gamma=2.5,
            alpha=3.0,
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
        gamma, alpha = 2.5, 3.0
        cat = make_realistic_source_catalog(5, shape=(512, 512), seed=42)
        for i in range(len(cat)):
            cat["x"][i] = 100 + i * 80
            cat["y"][i] = 256
        img, _ = simulate_image(
            (512, 512), cat, gamma=gamma, alpha=alpha, background=0, seed=42
        )
        total_injected = cat["flux"].sum()
        total_image = img.sum()
        ratio = total_image / total_injected
        assert 0.8 < ratio < 1.3

    def test_make_test_source_catalog(self):
        cat = gcp.make_test_source_catalog(n_sources_side=3, shape=(64, 64))
        assert len(cat) == 9
        assert "x" in cat.colnames and "y" in cat.colnames and "flux" in cat.colnames
