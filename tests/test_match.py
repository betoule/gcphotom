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

    def test_sky_project_path(self):
        # exercise project=True path inside match (used by cross_match catalog when non-xy)
        ref = {"ra": np.array([0.0, 10.0]), "dec": np.array([0.0, 0.0])}
        cat = {"ra": np.array([0.0]), "dec": np.array([0.0])}
        idx = gcp.match.match(ref, cat, project=True, radius=0.1)
        np.testing.assert_array_equal(idx, [0])

    def test_match_empty_ref_xy(self):
        ref = {"x": np.array([]), "y": np.array([])}
        cat = {"x": np.array([0.0]), "y": np.array([0.0])}
        idx = gcp.match.match(ref, cat, project=False, xy=True, radius=1.0)
        np.testing.assert_array_equal(idx, [-1])

    def test_match_spherical_haversine(self):
        ref = {"ra": np.array([0.0]), "dec": np.array([0.0])}
        cat = {"ra": np.array([0.0]), "dec": np.array([0.0])}
        idx = gcp.match.match(ref, cat, project=False, xy=False, radius=0.1)
        np.testing.assert_array_equal(idx, [0])


class TestCrossMatchCatalogs:
    def test_catalog_length_preserved_with_nans(self):
        from astropy.table import Table

        det = Table({"x": [50.1, 100.1], "y": [50.2, 100.1]})
        sim = Table({"x": [50.0, 200.0], "y": [50.0, 200.0], "flux": [100.0, 200.0]})
        res = gcp.cross_match(det, sim, tolerance=5.0)
        assert len(res) == len(det)
        assert "flux" in res.colnames
        assert np.isfinite(res["flux"][0])
        assert not np.isfinite(res["flux"][1])

    def test_catalog_order_follows_det(self):
        from astropy.table import Table

        det = Table({"x": [100.0, 50.0], "y": [100.0, 50.0]})
        sim = Table({"x": [50.0, 100.0], "y": [50.0, 100.0], "id": [1, 2]})
        res = gcp.cross_match(det, sim, tolerance=1.0)
        np.testing.assert_array_equal(res["id"], [2, 1])

    def test_catalog_empty_det(self):
        from astropy.table import Table

        det = Table({"x": [], "y": []})
        sim = Table({"x": [0.0], "y": [0.0], "flux": [10.0]})
        res = gcp.cross_match(det, sim)
        assert len(res) == 0
        assert "flux" in res.colnames

    def test_catalog_no_matches(self):
        from astropy.table import Table

        det = Table({"x": [0.0], "y": [0.0]})
        sim = Table({"x": [100.0], "y": [100.0], "flux": [99.0]})
        res = gcp.cross_match(det, sim, tolerance=1.0)
        assert len(res) == 1
        assert not np.isfinite(res["flux"][0])
