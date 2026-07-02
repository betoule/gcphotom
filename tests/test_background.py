import numpy as np
import pytest
from gcphotom.background import estimate_background
from gcphotom.simulator import make_realistic_source_catalog, simulate_image


class TestEstimateBackground:
    def test_output_shapes(self):
        image = np.random.randn(256, 256) + 100.0
        bkg, var = estimate_background(image)
        assert bkg.shape == image.shape
        assert var.shape == image.shape

    def test_constant_background_recovery(self):
        true_bkg = 123.4
        image = np.full((512, 512), true_bkg)
        rng = np.random.default_rng(42)
        image += rng.normal(0, 2.0, image.shape)
        bkg, var = estimate_background(image)
        np.testing.assert_allclose(bkg.mean(), true_bkg, atol=0.5)
        np.testing.assert_allclose(var.mean(), 4.0, atol=1.0)

    def test_gradient_recovery(self):
        ny, nx = 512, 512
        y, x = np.mgrid[:ny, :nx]
        gradient = 0.01 * x + 0.005 * y
        image = gradient + np.random.default_rng(42).normal(0, 1.0, (ny, nx))
        bkg, _ = estimate_background(image, box_size=(64, 64))
        np.testing.assert_allclose(bkg[0, 0], gradient[0, 0], atol=1.0)
        np.testing.assert_allclose(bkg[-1, -1], gradient[-1, -1], atol=1.0)

    def test_with_sources(self):
        shape = (512, 512)
        cat = make_realistic_source_catalog(50, shape=shape, seed=42)
        image, _ = simulate_image(
            shape, cat, gamma=3.0, alpha=3.0, background=200, read_noise=3, seed=42
        )
        bkg, var = estimate_background(image)
        np.testing.assert_allclose(bkg.mean(), 200, atol=5)
        assert np.all(var > 0)

    def test_with_mask(self):
        shape = (256, 256)
        image = np.full(shape, 100.0)
        rng = np.random.default_rng(42)
        image += rng.normal(0, 2.0, shape)
        mask = np.zeros(shape, dtype=bool)
        mask[100:120, 100:120] = True
        bkg, _ = estimate_background(image, mask=mask)
        assert bkg.shape == shape
        np.testing.assert_allclose(bkg.mean(), 100.0, atol=0.5)

    def test_variance_is_positive(self):
        image = np.random.randn(128, 128) + 50.0
        _, var = estimate_background(image)
        assert np.all(var > 0)
