"""Gaia DR3 source catalog generation for image simulation."""

import gaiahealpixcache
import numpy as np
from astropy.table import Table


def make_gaia_source_catalog(
    wcs,
    shape,
    zeropoint,
    g_min=None,
    g_max=20.0,
    margin_arcmin=5.0,
    seed=None,
):
    """Generate a source catalog from Gaia DR3 for a given image footprint.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        World Coordinate System describing the image geometry.  Must be a
        celestial projection (e.g. ``RA---TAN``, ``DEC--TAN``) with the
        standard ``crpix``, ``crval``, ``cdelt`` attributes set.
    shape : tuple of int
        Image shape ``(ny, nx)`` in pixels.
    zeropoint : float
        Magnitude zeropoint: the G-band magnitude that produces 1 ADU.
        Source flux is computed as ``10 ** (-0.4 * (g_mag - zeropoint))``.
    g_min : float or None
        Faintest G magnitude to include (inclusive).  ``None`` means no
        faint limit (all magnitudes up to *g_max*).
    g_max : float or None
        Brightest G magnitude to include (inclusive).  ``None`` means no
        bright limit.
    margin_arcmin : float
        Extra sky margin in arcminutes added on each side of the image
        footprint when querying the Gaia archive.  This ensures that
        sources just outside the image whose PSF wings might contribute
        flux are also included.
    seed : int or None
        Ignored (Gaia catalog is deterministic).  Accepted for
        compatibility with the ``MonteCarlo.catalog_fn`` interface.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Table with columns ``x``, ``y`` (0-indexed pixel coordinates) and
        ``flux`` (ADU), ready to pass to :func:`simulate_image`.
    """
    ny, nx = shape

    corners_pix = np.array(
        [[0.0, 0.0], [0.0, float(ny)], [float(nx), 0.0], [float(nx), float(ny)]]
    )
    corners_world = wcs.wcs_pix2world(corners_pix, 0)
    ra_corners = corners_world[:, 0]
    dec_corners = corners_world[:, 1]

    margin_deg = margin_arcmin / 60.0
    ra_min = float(ra_corners.min()) - margin_deg
    ra_max = float(ra_corners.max()) + margin_deg
    dec_min = float(dec_corners.min()) - margin_deg
    dec_max = float(dec_corners.max()) + margin_deg

    sources = gaiahealpixcache.query_rectangular(
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        product="bright_sources",
    )

    mag = sources["phot_g_mean_mag"]
    valid = np.isfinite(mag)
    if g_min is not None:
        valid &= mag >= g_min
    if g_max is not None:
        valid &= mag <= g_max

    sources = sources[valid]
    mag = sources["phot_g_mean_mag"]

    sky = np.column_stack([sources["ra"], sources["dec"]])
    pix = wcs.wcs_world2pix(sky, 0)
    xs = pix[:, 0]
    ys = pix[:, 1]

    margin_pix = 2
    in_image = (
        (xs >= -margin_pix)
        & (xs < nx + margin_pix)
        & (ys >= -margin_pix)
        & (ys < ny + margin_pix)
    )
    xs = xs[in_image]
    ys = ys[in_image]
    mag = mag[in_image]

    flux = 10.0 ** (-0.4 * (mag - zeropoint))

    catalog = Table()
    catalog["x"] = xs
    catalog["y"] = ys
    catalog["flux"] = flux

    return catalog
