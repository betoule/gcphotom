import numpy as np
import pytest
from astropy.table import Table
from gcphotom.aperture import (
    cross_match,
    detect_and_segment,
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
        assert result["background_var"].shape == (5, len(radii))

    def test_with_background_variance(self):
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
        bkg_var = np.full_like(img, 9.0)  # read_noise=3 → variance=9
        result = extract_growth_curves(
            img - 100, positions, radii, background_variance=bkg_var
        )
        assert np.all(result["background_var"] >= 0)

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
        seg, cat, _, _ = detect_and_segment(img, background=100)
        assert len(cat) == len(positions)

    def test_positions_close_to_truth(self, controlled_catalog):
        input_positions = np.array([(50, 50), (100, 100), (150, 150)])
        img = controlled_catalog(input_positions)
        seg, cat, _, _ = detect_and_segment(img, background=100)
        positions = np.column_stack([cat.x_centroid, cat.y_centroid])
        dists = np.sqrt(np.sum((positions - input_positions) ** 2, axis=1))
        assert np.all(dists < 1.0)

    def test_detects_without_explicit_background(self, controlled_catalog):
        positions = [(50, 50), (100, 50), (150, 50)]
        img = controlled_catalog(positions)
        seg, cat, _, _ = detect_and_segment(img)
        assert len(cat) == len(positions)


class TestExtractGrowthCurvesWithSegmentation:
    def test_returns_contamination_keys(self, controlled_catalog):
        img = controlled_catalog([(100, 100)])
        seg, cat, _, _ = detect_and_segment(img, background=100)
        sub = img - 100
        result = extract_growth_curves(
            sub,
            np.column_stack([cat.x_centroid, cat.y_centroid]),
            segmentation_image=seg,
        )
        assert "contamination" in result
        assert "flux_clean" in result

    def test_without_segmentation_clean_equals_total(self, controlled_catalog):
        img = controlled_catalog([(100, 100)])
        sub = img - 100
        seg, cat, _, _ = detect_and_segment(img, background=100)
        result = extract_growth_curves(
            sub, np.column_stack([cat.x_centroid, cat.y_centroid])
        )
        assert "flux_clean" in result
        assert "contamination" in result
        np.testing.assert_allclose(result["flux_clean"], result["flux"])
        np.testing.assert_allclose(result["contamination"], 0)

    def test_isolated_source_low_contamination(self, controlled_catalog):
        img = controlled_catalog([(128, 128)])
        seg, cat, _, _ = detect_and_segment(img, background=100)
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
        seg, cat, _, _ = detect_and_segment(img, background=100)
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
        seg, cat, _, _ = detect_and_segment(img, background=100)
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


class TestExtractGrowthCurvesCatalogInput:
    def test_extract_accepts_table(self, controlled_catalog):
        img = controlled_catalog([(60, 60), (120, 120)])
        seg, cat, _, _ = detect_and_segment(img, background=100)
        # build a Table from the catalog
        tab = Table({"x": cat.x_centroid, "y": cat.y_centroid})
        result = extract_growth_curves(img, tab, segmentation_image=seg)
        assert result["flux"].shape[0] == len(cat)

    def test_extract_accepts_source_catalog(self, controlled_catalog):
        img = controlled_catalog([(70, 70)])
        seg, cat, _, _ = detect_and_segment(img, background=100)
        result = extract_growth_curves(img, cat, segmentation_image=seg)
        assert result["flux"].shape[0] == 1

    def test_auto_background_variance(self, controlled_catalog):
        img = controlled_catalog([(80, 80)])
        seg, cat, _, _ = detect_and_segment(img, background=100)
        # do not pass background_variance -> auto inside
        result = extract_growth_curves(img, cat, segmentation_image=seg)
        assert np.all(result["background_var"] >= 0)

    def test_extract_accepts_positions_list(self, controlled_catalog):
        img = controlled_catalog([(55, 55)])
        seg, cat, _, _ = detect_and_segment(img, background=100)
        poss = [[float(cat.x_centroid[0]), float(cat.y_centroid[0])]]
        result = extract_growth_curves(img, poss, segmentation_image=seg)
        assert result["flux"].shape[0] == 1

    def test_extract_rejects_bad_sources(self, controlled_catalog):
        img = controlled_catalog([(90, 90)])
        with pytest.raises(TypeError):
            extract_growth_curves(img, "not-positions")

    def test_extract_rejects_1d_array(self, controlled_catalog):
        img = controlled_catalog([(88, 88)])
        with pytest.raises(TypeError):
            extract_growth_curves(img, np.array([1.0, 2.0]))

    def test_extract_table_without_xy_raises(self, controlled_catalog):
        img = controlled_catalog([(95, 95)])
        bad = Table({"a": [1], "b": [2]})
        with pytest.raises(TypeError):
            extract_growth_curves(img, bad)

    def test_deblend_splits_close_pair(self, controlled_catalog):
        # two sources ~6 px apart form a single blob without deblending
        img = controlled_catalog([(60, 60), (66, 60)], flux=1e5)
        seg_no, cat_no, _, _ = detect_and_segment(img, background=100, deblend=False)
        assert len(cat_no) == 1
        seg_yes, cat_yes, _, _ = detect_and_segment(img, background=100, deblend=True)
        assert len(cat_yes) == 2

    def test_returns_2d_background_map(self, controlled_catalog):
        img = controlled_catalog([(100, 100)])
        _, _, bkg_map, bkg_var_map = detect_and_segment(img, background=100)
        assert bkg_map.shape == img.shape
        assert bkg_var_map.shape == img.shape
        np.testing.assert_allclose(bkg_map, 100, atol=0.01)
        assert np.all(bkg_var_map > 0)

    def test_2d_background_auto_estimate(self, controlled_catalog):
        img = controlled_catalog([(100, 100)])
        _, _, bkg_map, bkg_var_map = detect_and_segment(img)
        assert bkg_map.shape == img.shape
        assert bkg_var_map.shape == img.shape
        np.testing.assert_allclose(bkg_map.mean(), 100, atol=5)
        assert np.all(bkg_var_map > 0)

    def test_extract_with_2d_variance(self, controlled_catalog):
        img = controlled_catalog([(100, 100)])
        seg, cat, _, _ = detect_and_segment(img, background=100)
        bkg_var = np.full_like(img, 25.0)
        result = extract_growth_curves(
            img - 100,
            np.column_stack([cat.x_centroid, cat.y_centroid]),
            background_variance=bkg_var,
        )
        assert result["background_var"].shape[0] == len(cat)
        assert np.all(result["background_var"] > 0)
