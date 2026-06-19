import importlib.util
import numpy as np
import pytest
from astropy.table import Table
from gcphotom.aperture import (
    cross_match,
    detect_and_segment,
    estimate_error,
    extract_growth_curves,
    _extract_single_growth_curve,
)
from gcphotom.simulator import make_realistic_source_catalog, simulate_image


@pytest.fixture
def simple_image():
    shape = (128, 128)
    cat = make_realistic_source_catalog(1, shape=shape, seed=42)
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
        cat = make_realistic_source_catalog(5, shape=shape, seed=42)
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
        cat = make_realistic_source_catalog(3, shape=shape, seed=42)
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
        cat = make_realistic_source_catalog(1, shape=shape, seed=42)
        cat["x"][0] = 64
        cat["y"][0] = 64
        img, _ = simulate_image(shape, cat, gamma=2.5, alpha=3.0, background=0, seed=42)
        positions = np.column_stack([cat["x"], cat["y"]])
        result = extract_growth_curves(img, positions)

        assert len(result["radius"]) > 2
        assert result["radius"][0] > 0
        assert result["radius"][-1] > result["radius"][0]


@pytest.fixture
def controlled_catalog():
    """Create a catalog with known bright, well-separated sources."""

    def _make(positions, flux=1e5, shape=(256, 256), background=100, seed=42):
        cat = Table()
        cat["x"] = np.array([p[0] for p in positions])
        cat["y"] = np.array([p[1] for p in positions])
        cat["flux"] = np.full(len(positions), flux)
        img, _ = simulate_image(
            shape, cat, gamma=2.5, alpha=3.0, background=background, seed=seed
        )
        return img

    return _make


class TestDetectAndSegment:
    def test_detects_all_well_separated(self, controlled_catalog):
        positions = [(50, 50), (100, 50), (150, 50), (50, 150), (150, 150)]
        img = controlled_catalog(positions)
        seg, cat = detect_and_segment(img, background=100)
        assert len(cat) == len(positions)

    def test_positions_close_to_truth(self, controlled_catalog):
        input_positions = np.array([(50, 50), (100, 100), (150, 150)])
        img = controlled_catalog(input_positions)
        seg, cat = detect_and_segment(img, background=100)
        positions = np.column_stack([cat.x_centroid, cat.y_centroid])
        dists = np.sqrt(np.sum((positions - input_positions) ** 2, axis=1))
        assert np.all(dists < 1.0)

class TestExtractGrowthCurvesWithSegmentation:
    def test_returns_contamination_keys(self, controlled_catalog):
        img = controlled_catalog([(100, 100)])
        seg, cat = detect_and_segment(img, background=100)
        sub = img - 100
        result = extract_growth_curves(
            sub,
            np.column_stack([cat.x_centroid, cat.y_centroid]),
            segmentation_image=seg,
        )
        assert "contamination" in result
        assert "flux_clean" in result

    def test_without_segmentation_no_contamination(self, controlled_catalog):
        img = controlled_catalog([(100, 100)])
        sub = img - 100
        seg, cat = detect_and_segment(img, background=100)
        result = extract_growth_curves(sub, np.column_stack([cat.x_centroid, cat.y_centroid]))
        assert "contamination" not in result
        assert "flux_clean" not in result

    def test_isolated_source_low_contamination(self, controlled_catalog):
        img = controlled_catalog([(128, 128)])
        seg, cat = detect_and_segment(img, background=100)
        sub = img - 100
        result = extract_growth_curves(
            sub,
            np.column_stack([cat.x_centroid, cat.y_centroid]),
            segmentation_image=seg,
        )
        # Contamination for isolated source is flux in PSF wings outside segment.
        # It should be a small fraction of total flux.
        assert np.all(result["contamination"] / result["flux"] < 0.05)

    def test_overlapping_pair_has_contamination(self, controlled_catalog):
        img = controlled_catalog([(100, 100), (110, 110)])
        seg, cat = detect_and_segment(img, background=100)
        sub = img - 100
        radii = np.arange(3, 20, 1)
        result = extract_growth_curves(
            sub,
            np.column_stack([cat.x_centroid, cat.y_centroid]),
            radii=radii,
            segmentation_image=seg,
        )
        assert len(cat) == 2
        assert np.any(result["contamination"] > 0)

    def test_end_to_end_simulated(self, controlled_catalog):
        img = controlled_catalog([(60, 60), (128, 128), (196, 60)])
        seg, cat = detect_and_segment(img, background=100)
        sub = img - 100
        result = extract_growth_curves(
            sub,
            np.column_stack([cat.x_centroid, cat.y_centroid]),
            segmentation_image=seg,
        )
        assert result["contamination"].shape[0] == len(cat)
        assert np.all(result["contamination"] >= 0)
        assert np.all(result["flux_clean"] >= 0)


class TestCrossMatch:
    def test_all_matched_for_well_separated(self):
        input_pos = np.array([[50, 50], [100, 100], [150, 150]])
        detected = np.array([[50.3, 50.2], [100.1, 99.9], [149.8, 150.1]])
        result = cross_match(input_pos, detected, tolerance=5.0)
        assert np.all(result["match_indices"] >= 0)
        assert np.all(result["match_distances"] < 5.0)

    def test_unmatched_beyond_tolerance(self):
        input_pos = np.array([[50, 50]])
        detected = np.array([[200, 200]])
        result = cross_match(input_pos, detected, tolerance=5.0)
        assert result["match_indices"][0] == -1
        assert np.isinf(result["match_distances"][0])

    def test_close_pair_both_matched(self):
        input_pos = np.array([[100, 100], [106, 106]])
        detected = np.array([[100.1, 100.1], [105.9, 105.9]])
        result = cross_match(input_pos, detected, tolerance=5.0)
        assert np.all(result["match_indices"] >= 0)
        assert len(np.unique(result["match_indices"])) == 2
