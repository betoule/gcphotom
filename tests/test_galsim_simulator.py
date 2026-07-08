"""Tests for the GalSim-based image simulation module."""

import numpy as np
from astropy.table import Table
import pytest

import gcphotom as gcp


class TestSimulateImageGalsim:
    def test_empty_catalog(self):
        """An empty catalog produces a zero image of the right shape."""
        shape = (64, 64)
        cat = Table({"x": [], "y": [], "flux": []}, dtype=[float, float, float])
        img, returned = gcp.simulate_image_galsim(
            shape, cat, background=0, read_noise=0, method="auto"
        )
        assert img.shape == shape
        assert img.sum() == 0
        assert returned is cat

    def test_fft_flux_conservation(self):
        """Noiseless FFT of a single source recovers the input flux."""
        shape = (129, 129)
        flux = 50000.0
        cat = Table({"x": [64.0], "y": [64.0], "flux": [flux]})
        img, _ = gcp.simulate_image_galsim(
            shape, cat, gamma=3.0, alpha=3.0, background=0, read_noise=0, method="auto"
        )
        recovered = img.sum()
        assert recovered == pytest.approx(flux, rel=1e-3)

    def test_coordinate_shift(self):
        """A source at a sub-pixel position has the correct centroid."""
        shape = (129, 129)
        x_src, y_src = 63.2, 64.7
        cat = Table({"x": [x_src], "y": [y_src], "flux": [50000.0]})
        img, _ = gcp.simulate_image_galsim(
            shape, cat, gamma=3.0, alpha=3.0, background=0, read_noise=0, method="auto"
        )
        total = img.sum()
        xc = np.sum(np.arange(shape[1]) * img.sum(axis=0)) / total
        yc = np.sum(np.arange(shape[0]) * img.sum(axis=1)) / total
        assert xc == pytest.approx(x_src, abs=0.1)
        assert yc == pytest.approx(y_src, abs=0.1)

    def test_photon_shooting_shape(self):
        """Photon shooting produces a valid image with positive flux."""
        shape = (129, 129)
        cat = Table({"x": [64.0], "y": [64.0], "flux": [50000.0]})
        img, _ = gcp.simulate_image_galsim(
            shape, cat, gamma=3.0, alpha=3.0, background=0, read_noise=0, method="phot"
        )
        assert img.shape == shape
        assert img.dtype == np.float64
        assert img.sum() > 0

    def test_auto_with_noise(self):
        """Auto mode with auto-generated catalog and Poisson+read noise."""
        shape = (64, 64)
        img, cat = gcp.simulate_image_galsim(
            shape, n_sources=5, background=100, read_noise=3, seed=42, method="auto"
        )
        assert img.shape == shape
        assert len(cat) == 5

    def test_phot_with_background(self):
        """Phot mode with background and read noise."""
        shape = (64, 64)
        cat = Table({"x": [32.0], "y": [32.0], "flux": [50000.0]})
        img, _ = gcp.simulate_image_galsim(
            shape, cat, background=50, read_noise=2, seed=42, method="phot"
        )
        assert img.shape == shape
        assert img.sum() > 0

    def test_phot_zero_flux(self):
        """Phot mode with all sources having zero flux."""
        shape = (64, 64)
        cat = Table({"x": [32.0], "y": [32.0], "flux": [0.0]})
        img, _ = gcp.simulate_image_galsim(
            shape, cat, background=0, read_noise=0, method="phot"
        )
        assert img.sum() == 0

    def test_auto_zero_flux(self):
        """Auto mode with all sources having zero flux."""
        shape = (64, 64)
        cat = Table({"x": [32.0], "y": [32.0], "flux": [0.0]})
        img, _ = gcp.simulate_image_galsim(
            shape, cat, background=0, read_noise=0, method="auto"
        )
        assert img.sum() == 0

    def test_chromatic_rendering(self):
        """Chromatic photon shooting produces a finite image."""
        shape = (64, 64)
        cat = Table({"x": [32.0], "y": [32.0], "flux": [50000.0], "bp_rp": [0.0]})
        img, _ = gcp.simulate_image_galsim(
            shape,
            cat,
            background=0,
            read_noise=0,
            method="phot",
            max_phot_sources=10,
            chromatic=True,
            bandpass="r",
        )
        assert img.shape == shape
        assert img.sum() > 0
        assert np.all(np.isfinite(img))

    def test_chromatic_with_sensor(self):
        """Chromatic + sensor rendering produces a finite image."""
        shape = (64, 64)
        cat = Table({"x": [32.0], "y": [32.0], "flux": [50000.0], "bp_rp": [0.5]})
        img, _ = gcp.simulate_image_galsim(
            shape,
            cat,
            background=0,
            read_noise=0,
            method="phot",
            max_phot_sources=10,
            chromatic=True,
            bandpass="r",
            sensor=True,
            bf_strength=0.5,
            diffusion_factor=0.5,
        )
        assert img.shape == shape
        assert img.sum() > 0
        assert np.all(np.isfinite(img))

    def test_chromatic_auto_switches_to_phot(self):
        """chromatic=True forces method='phot' even if method='auto'."""
        shape = (64, 64)
        cat = Table({"x": [32.0], "y": [32.0], "flux": [50000.0], "bp_rp": [1.0]})
        img, _ = gcp.simulate_image_galsim(
            shape,
            cat,
            background=0,
            read_noise=0,
            method="auto",
            chromatic=True,
            bandpass="g",
        )
        assert img.shape == shape
        assert img.sum() > 0
