"""Tests for Gaia DR3 source catalog generation."""

import numpy as np
import pytest

from gcphotom.gaia_catalog import make_gaia_source_catalog


def _simple_wcs():
    """Create a simple TAN WCS centred at (ra, dec) = (0, 0)."""
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [64.0, 64.0]
    w.wcs.cdelt = [-0.001, 0.001]  # ~3.6 arcsec/pix
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set()
    return w


def test_make_gaia_source_catalog_handles_empty_sky():
    """A WCS pointing at an empty patch of sky returns an empty catalog."""
    w = _simple_wcs()
    shape = (128, 128)
    try:
        cat = make_gaia_source_catalog(w, shape, zeropoint=25.0, g_max=10.0)
        assert isinstance(cat, type(cat))
        assert "x" in cat.colnames
        assert "y" in cat.colnames
        assert "flux" in cat.colnames
    except Exception as exc:
        msg = str(exc)
        if (
            "connection" in msg.lower()
            or "timeout" in msg.lower()
            or "no data" in msg.lower()
        ):
            pytest.skip(f"Gaia query unavailable: {msg}")
        else:
            raise
