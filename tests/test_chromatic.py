"""Tests for chromatic PSF, SED, and sensor utilities."""

import numpy as np
import pytest

import gcphotom as gcp


class TestTophatBandpass:
    def test_known_bandpasses(self):
        for name in ("g", "r", "i", "z"):
            bp = gcp.tophat_bandpass(name)
            assert bp.blue_limit < bp.red_limit
            assert bp.effective_wavelength > 0

    def test_unknown_bandpass(self):
        with pytest.raises(ValueError, match="Unknown bandpass"):
            gcp.tophat_bandpass("u")


class TestSedFromColor:
    def test_sed_is_callable(self):
        sed = gcp.sed_from_color(0.0)
        flux = sed(500.0)
        assert np.isfinite(flux)
        assert flux > 0

    def test_sed_variation_with_color(self):
        """Blue stars (negative bp_rp) are brighter in blue bands."""
        bp_b = gcp.tophat_bandpass("g")
        bp_r = gcp.tophat_bandpass("r")
        blue_sed = gcp.sed_from_color(-0.3)
        red_sed = gcp.sed_from_color(2.0)

        blue_b = blue_sed.calculateFlux(bp_b)
        blue_r = blue_sed.calculateFlux(bp_r)
        red_b = red_sed.calculateFlux(bp_b)
        red_r = red_sed.calculateFlux(bp_r)

        # Blue star has higher g/r ratio than red star
        blue_ratio = blue_b / blue_r
        red_ratio = red_b / red_r
        assert blue_ratio > red_ratio

    def test_normalization(self):
        """SED normalized with withFlux gives correct flux in bandpass."""
        bp = gcp.tophat_bandpass("r")
        sed = gcp.sed_from_color(0.0)
        sed_norm = sed.withFlux(50000.0, bandpass=bp)
        assert sed_norm.calculateFlux(bp) == pytest.approx(50000.0, rel=1e-10)


class TestBuildChromaticPsf:
    def test_default_psf(self):
        psf = gcp.build_chromatic_psf()
        assert hasattr(psf, "drawImage")

    def test_atmosphere_only(self):
        psf = gcp.build_chromatic_psf(atmosphere=True, optics=False)
        assert hasattr(psf, "drawImage")

    def test_optics_only(self):
        psf = gcp.build_chromatic_psf(atmosphere=False, optics=True)
        assert hasattr(psf, "drawImage")

    def test_no_components(self):
        """Fallback to base Moffat when both atmosphere and optics are off."""
        psf = gcp.build_chromatic_psf(atmosphere=False, optics=False)
        assert hasattr(psf, "drawImage")


class TestBuildSensor:
    def test_no_effects_returns_none(self):
        assert gcp.build_sensor(bf_strength=0.0, diffusion_factor=0.0) is None

    def test_bf_only(self):
        sensor = gcp.build_sensor(bf_strength=1.0, diffusion_factor=0.0)
        assert sensor is not None

    def test_diffusion_only(self):
        sensor = gcp.build_sensor(bf_strength=0.0, diffusion_factor=1.0)
        assert sensor is not None

    def test_both(self):
        sensor = gcp.build_sensor(bf_strength=0.5, diffusion_factor=0.3)
        assert sensor is not None
        assert "SiliconSensor" in type(sensor).__name__
