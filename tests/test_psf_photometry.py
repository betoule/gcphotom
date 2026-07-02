import numpy as np
import pytest

import gcphotom as gcp
from gcphotom.psf_photometry import _as_xy, psf_photometry


class TestAsXY:
    def test_source_catalog(self):
        image, _ = gcp.simulate_image(n_sources=10, seed=42)
        _, cat, _ = gcp.detect_and_segment(image, n_pixels=5)
        x, y = _as_xy(cat)
        assert len(x) > 0
        assert np.allclose(x, cat.x_centroid)
        assert np.allclose(y, cat.y_centroid)

    def test_table(self):
        from astropy.table import Table

        tab = Table({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        x, y = _as_xy(tab)
        np.testing.assert_array_equal(x, [1.0, 2.0])
        np.testing.assert_array_equal(y, [3.0, 4.0])

    def test_array(self):
        arr = np.array([[1.0, 3.0], [2.0, 4.0]])
        x, y = _as_xy(arr)
        np.testing.assert_array_equal(x, [1.0, 2.0])
        np.testing.assert_array_equal(y, [3.0, 4.0])


class TestPSFPhotometry:
    @pytest.fixture
    def image_and_sources(self):
        image, sim_cat = gcp.simulate_image(n_sources=100, seed=42)
        seg, det_cat, _ = gcp.detect_and_segment(image, n_pixels=5)
        return image, sim_cat, det_cat

    def test_returns_results_and_epsf(self, image_and_sources):
        image, _, det_cat = image_and_sources
        results, epsf_res = psf_photometry(image, det_cat, nstars=10)
        assert results is not None
        assert epsf_res is not None
        assert hasattr(epsf_res, "epsf")

    def test_results_columns(self, image_and_sources):
        image, _, det_cat = image_and_sources
        results, _ = psf_photometry(image, det_cat, nstars=10)
        for col in ("flux_fit", "flux_err", "x_fit", "y_fit"):
            assert col in results.colnames

    def test_flux_recovery(self, image_and_sources):
        image, sim_cat, det_cat = image_and_sources
        results, _ = psf_photometry(image, det_cat, nstars=10)
        matched = gcp.cross_match(results, sim_cat)
        ratio = results["flux_fit"] / matched["flux"]
        assert np.nanmedian(np.abs(ratio - 1)) < 0.1

    def test_too_few_isolated_stars(self):
        image, _ = gcp.simulate_image(n_sources=2, seed=42)
        _, det_cat, _ = gcp.detect_and_segment(image, n_pixels=5)
        with pytest.raises(ValueError, match="isolated"):
            psf_photometry(image, det_cat, nstars=10)
