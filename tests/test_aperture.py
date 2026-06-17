import numpy as np
import pytest
from gcphotom.aperture import (
    estimate_error,
    extract_growth_curves,
    _extract_single_growth_curve,
)
from gcphotom.simulator import make_source_catalog, simulate_image


@pytest.fixture
def simple_image():
    shape = (128, 128)
    cat = make_source_catalog(1, shape=shape, seed=42)
    cat["x"][0] = 64
    cat["y"][0] = 64
    img, returned_cat = simulate_image(
        shape, cat, gamma=2.5, alpha=3.0, background=0, seed=42
    )
    return img, returned_cat


class TestEstimateError:
    def test_shape(self):
        img = np.ones((64, 64)) * 100
        err = estimate_error(img, background=50, read_noise=3)
        assert err.shape == (64, 64)

    def test_known_values(self):
        img = np.ones((10, 10)) * 100
        err = estimate_error(img, background=50, read_noise=0)
        expected = np.sqrt(50)
        assert np.allclose(err, expected)

    def test_read_noise_dominant(self):
        img = np.ones((10, 10)) * 10
        err = estimate_error(img, background=10, read_noise=5)
        expected = np.sqrt(0 + 5**2)
        assert np.allclose(err, expected)

    def test_no_negative_signal(self):
        img = np.ones((10, 10)) * 10
        err = estimate_error(img, background=50, read_noise=3)
        expected = np.sqrt(0 + 3**2)
        assert np.allclose(err, expected)


class TestExtractSingleGrowthCurve:
    def test_output_shapes(self, simple_image):
        img, cat = simple_image
        radii = np.arange(1, 20, 0.5)
        radius, profile, perr = _extract_single_growth_curve(
            img, (cat["x"][0], cat["y"][0]), radii
        )
        assert len(radius) == len(radii)
        assert len(profile) == len(radii)
        assert len(perr) == len(radii)

    def test_monotonic_increase(self, simple_image):
        img, cat = simple_image
        radii = np.arange(1, 15, 0.5)
        _, profile, _ = _extract_single_growth_curve(
            img, (cat["x"][0], cat["y"][0]), radii
        )
        increasing = np.diff(profile) > 0
        assert increasing.sum() > len(increasing) * 0.8

    def test_flux_recovery(self, simple_image):
        img, cat = simple_image
        radii = np.arange(1, 30, 0.5)
        _, profile, _ = _extract_single_growth_curve(
            img, (cat["x"][0], cat["y"][0]), radii
        )
        ratio = profile[-1] / cat["flux"][0]
        assert 0.7 < ratio < 1.3

    def test_with_error(self, simple_image):
        img, cat = simple_image
        radii = np.arange(1, 20, 0.5)
        error = np.ones_like(img) * 2
        _, _, perr = _extract_single_growth_curve(
            img, (cat["x"][0], cat["y"][0]), radii, error=error
        )
        assert np.all(perr > 0)


class TestExtractGrowthCurves:
    def test_multi_source(self):
        shape = (256, 256)
        cat = make_source_catalog(5, shape=shape, seed=42)
        for i in range(len(cat)):
            cat["x"][i] = 50 + i * 40
            cat["y"][i] = 128
        img, _ = simulate_image(shape, cat, gamma=2.5, alpha=3.0, background=0, seed=42)
        positions = np.column_stack([cat["x"], cat["y"]])
        radii = np.arange(1, 20, 0.5)
        result = extract_growth_curves(img, positions, radii)

        assert len(result["radius"]) == len(radii)
        assert result["flux"].shape == (5, len(radii))
        assert result["flux_err"].shape == (5, len(radii))

    def test_with_error_map(self):
        shape = (256, 256)
        cat = make_source_catalog(3, shape=shape, seed=42)
        for i in range(len(cat)):
            cat["x"][i] = 80 + i * 50
            cat["y"][i] = 128
        img, _ = simulate_image(
            shape, cat, gamma=2.5, alpha=3.0, background=100, seed=42
        )
        positions = np.column_stack([cat["x"], cat["y"]])
        radii = np.arange(1, 20, 0.5)
        error = estimate_error(img, 100, 3)
        result = extract_growth_curves(img - 100, positions, radii, error=error)
        assert np.all(result["flux_err"] > 0)

    def test_default_radii(self):
        shape = (128, 128)
        cat = make_source_catalog(1, shape=shape, seed=42)
        cat["x"][0] = 64
        cat["y"][0] = 64
        img, _ = simulate_image(shape, cat, gamma=2.5, alpha=3.0, background=0, seed=42)
        positions = np.column_stack([cat["x"], cat["y"]])
        result = extract_growth_curves(img, positions)

        assert len(result["radius"]) > 2
        assert result["radius"][0] > 0
        assert result["radius"][-1] > result["radius"][0]
