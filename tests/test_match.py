"""Tests for gcphotom.match efficient cross-matching."""

import numpy as np

import gcphotom as gcp


class TestMatchPixel:
    def test_exact_match(self):
        ref = {"x": np.array([0.0, 10.0]), "y": np.array([0.0, 0.0])}
        cat = {"x": np.array([0.0, 10.0]), "y": np.array([0.0, 0.0])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=1.0)
        np.testing.assert_array_equal(idx, [0, 1])

    def test_within_tolerance(self):
        ref = {"x": np.array([50.0, 100.0]), "y": np.array([50.0, 100.0])}
        cat = {"x": np.array([50.3, 100.1]), "y": np.array([50.2, 99.9])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=5.0)
        np.testing.assert_array_equal(idx, [0, 1])

    def test_beyond_tolerance(self):
        ref = {"x": np.array([0.0]), "y": np.array([0.0])}
        cat = {"x": np.array([100.0]), "y": np.array([100.0])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=5.0)
        np.testing.assert_array_equal(idx, [-1])

    def test_many_to_one_allowed(self):
        ref = {"x": np.array([0.0]), "y": np.array([0.0])}
        cat = {"x": np.array([0.0, 0.1]), "y": np.array([0.0, 0.0])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=1.0)
        np.testing.assert_array_equal(idx, [0, 0])

    def test_empty_query(self):
        ref = {"x": np.array([0.0]), "y": np.array([0.0])}
        cat = {"x": np.array([]), "y": np.array([])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=1.0)
        assert len(idx) == 0

    def test_empty_ref(self):
        ref = {"x": np.array([]), "y": np.array([])}
        cat = {"x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=1.0)
        np.testing.assert_array_equal(idx, [-1, -1])

    def test_zero_tolerance_exact(self):
        ref = {"x": np.array([5.0]), "y": np.array([5.0])}
        cat = {"x": np.array([5.0]), "y": np.array([5.0])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=0.0)
        np.testing.assert_array_equal(idx, [0])

    def test_zero_tolerance_miss(self):
        ref = {"x": np.array([5.0]), "y": np.array([5.0])}
        cat = {"x": np.array([5.0001]), "y": np.array([5.0])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=0.0)
        np.testing.assert_array_equal(idx, [-1])


class TestMatchDistances:
    def test_distances_optional_default_false(self):
        ref = {"x": np.array([0.0]), "y": np.array([0.0])}
        cat = {"x": np.array([0.0]), "y": np.array([0.0])}
        res = gcp.match.match(ref, cat, project=False, xy=True, radius=1.0)
        assert isinstance(res, np.ndarray)  # only indices

    def test_distances_exact_when_requested(self):
        ref = {"x": np.array([0.0, 100.0]), "y": np.array([0.0, 0.0])}
        cat = {"x": np.array([0.0, 100.3, 999.0]), "y": np.array([0.0, 0.0, 0.0])}
        idx, dist = gcp.match.match(
            ref, cat, project=False, xy=True, radius=5.0, compute_distances=True
        )
        np.testing.assert_array_equal(idx, [0, 1, -1])
        assert dist[0] == 0.0
        assert np.isclose(dist[1], 0.3)
        assert np.isinf(dist[2])


class TestMatchSky:
    def test_sky_one_arcsec_match(self):
        # 1 arcsec separation, tolerance 2 arcsec. Radius in radians for projected.
        ref = {"ra": np.array([0.0]), "dec": np.array([0.0])}
        cat = {"ra": np.array([0.0]), "dec": np.array([1.0 / 3600.0])}
        rad = 2.0 / 3600.0 * np.pi / 180.0
        idx = gcp.match.match(ref, cat, project=True, radius=rad)
        np.testing.assert_array_equal(idx, [0])

    def test_sky_beyond_tolerance(self):
        ref = {"ra": np.array([0.0]), "dec": np.array([0.0])}
        cat = {"ra": np.array([0.0]), "dec": np.array([10.0 / 3600.0])}
        rad = 1.0 / 3600.0 * np.pi / 180.0
        idx = gcp.match.match(ref, cat, project=True, radius=rad)
        np.testing.assert_array_equal(idx, [-1])


class TestCrossMatchWrapper:
    def test_wrapper_basic(self):
        inp = np.array([[50.0, 50.0], [100.0, 100.0]])
        det = np.array([[50.1, 50.2], [99.9, 100.1]])
        res = gcp.cross_match(inp, det, tolerance=1.0)
        assert np.all(res["match_indices"] >= 0)
        assert np.all(res["match_distances"] < 1.0)

    def test_wrapper_unmatched(self):
        inp = np.array([[0.0, 0.0]])
        det = np.array([[100.0, 100.0]])
        res = gcp.cross_match(inp, det, tolerance=1.0)
        assert res["match_indices"][0] == -1
        assert np.isinf(res["match_distances"][0])

    def test_wrapper_empty(self):
        res = gcp.cross_match(np.empty((0, 2)), np.array([[0.0, 0.0]]))
        assert len(res["match_indices"]) == 0

    def test_wrapper_many_to_one(self):
        inp = np.array([[0.0, 0.0], [0.01, 0.0]])
        det = np.array([[0.0, 0.0]])
        res = gcp.cross_match(inp, det, tolerance=1.0)
        np.testing.assert_array_equal(res["match_indices"], [0, 0])
