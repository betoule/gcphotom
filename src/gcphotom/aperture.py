import numpy as np
from photutils.profiles import CurveOfGrowth
from photutils.segmentation import (
    detect_sources,
    detect_threshold,
    deblend_sources,
    SourceCatalog,
)

from .background import estimate_background
from .match import cross_match as _cross_match  # re-export for backward compat


def _extract_single_growth_curve(image, position, radii, error=None, mask=None):
    """Extract a circular growth curve for a single source.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Image data (should be background-subtracted).
    position : tuple of float
        ``(x, y)`` pixel coordinate of the source center.
    radii : 1D `~numpy.ndarray`
        Aperture radii in pixels.
    error : 2D `~numpy.ndarray` or None
        Per-pixel 1-sigma error.
    mask : 2D `~numpy.ndarray` of bool or None
        Boolean mask where ``True`` indicates excluded pixels.
        Passed directly to ``CurveOfGrowth``.

    Returns
    -------
    radius : 1D `~numpy.ndarray`
        Aperture radii.
    profile : 1D `~numpy.ndarray`
        Cumulative flux at each radius.
    profile_error : 1D `~numpy.ndarray`
        Flux uncertainty at each radius.
    """
    cog = CurveOfGrowth(image, position, radii, error=error, mask=mask)
    perr = (
        cog.profile_error if len(cog.profile_error) > 0 else np.zeros_like(cog.profile)
    )
    return cog.radius, cog.profile, perr


def _as_positions(sources):
    """Convert sources (array, Table, or SourceCatalog) to (N, 2) float array."""
    if isinstance(sources, np.ndarray):
        arr = np.asarray(sources, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
    # astropy Table-like with x/y columns
    if hasattr(sources, "colnames"):
        try:
            x = np.asarray(sources["x"], dtype=float)
            y = np.asarray(sources["y"], dtype=float)
            return np.column_stack([x, y])
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    # photutils SourceCatalog or similar
    if hasattr(sources, "x_centroid") and hasattr(sources, "y_centroid"):
        x = np.asarray(sources.x_centroid, dtype=float)
        y = np.asarray(sources.y_centroid, dtype=float)
        return np.column_stack([x, y])
    # fallback: sequence of positions
    try:
        arr = np.asarray(sources, dtype=float)
    except (ValueError, TypeError):
        raise TypeError(
            "sources must be an (N,2) ndarray, Table with 'x'/'y', or SourceCatalog"
        ) from None
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    raise TypeError(
        "sources must be an (N,2) ndarray, Table with 'x'/'y', or SourceCatalog"
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def extract_growth_curves(
    image, sources, radii=None, background_variance=None, segmentation_image=None
):
    """Extract circular growth curves for multiple sources.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Image data. Background subtraction is optional; a linear background
        term is modeled during fitting.
    sources : 2D `~numpy.ndarray`, `~astropy.table.Table`, or `~photutils.segmentation.SourceCatalog`
        ``(n_sources, 2)`` array of ``(x, y)`` coordinates, or a catalog
        providing ``x``/``y`` columns or ``x_centroid``/``y_centroid``.
    radii : 1D `~numpy.ndarray` or None
        Aperture radii in pixels. Defaults to 10 logarithmically spaced
        values between 3 and 30 pixels.
    background_variance : float, 2D `~numpy.ndarray` or None
        Per-pixel background variance map. If a scalar, a uniform
        variance is assumed. If ``None`` (default), a 2D variance map
        is estimated via mesh-based sigma-clipped statistics using
        :func:`estimate_background`. This variance captures background
        photon noise, read-out noise, and any other spatially stationary
        noise source. Object photon noise is handled separately during
        fitting.
    segmentation_image : `~photutils.segmentation.SegmentationImage` or None
        Segmentation map from :func:`detect_and_segment`. If provided,
        contamination from neighboring sources is estimated.

    Returns
    -------
    result : dict
        Dictionary with keys:

        * ``radius``: 1D array of aperture radii.
        * ``flux``: 2D array ``(n_sources, n_radii)`` of cumulative flux.
        * ``background_var``: 2D array ``(n_sources, n_radii)`` of
          cumulative background variance (background pixel variance
          multiplied by the effective aperture area at each radius).
        * ``flux_clean``: 2D array ``(n_sources, n_radii)`` of flux with
          neighboring sources masked. When ``segmentation_image`` is not
          provided, ``flux_clean`` is identical to ``flux``.
        * ``contamination``: 2D array ``(n_sources, n_radii)`` of
          contaminating flux (``flux - flux_clean``). When no segmentation
          is provided, this is an array of zeros.
    """
    if radii is None:
        radii = np.logspace(np.log10(3), np.log10(30), num=10)

    positions = _as_positions(sources)
    n_sources = len(positions)
    n_radii = len(radii)
    flux = np.zeros((n_sources, n_radii))
    background_var = np.zeros((n_sources, n_radii))
    flux_clean = np.zeros((n_sources, n_radii))
    contamination = np.zeros((n_sources, n_radii))

    if background_variance is None:
        _, bkg_var = estimate_background(image)
        background_variance = bkg_var
    elif np.isscalar(background_variance):
        background_variance = np.full_like(image, background_variance, dtype=float)

    error = np.sqrt(background_variance)

    if segmentation_image is not None:
        seg_data = segmentation_image.data

    for i, pos in enumerate(positions):
        _, profile, profile_err = _extract_single_growth_curve(
            image, pos, radii, error=error
        )
        flux[i] = profile
        background_var[i] = profile_err**2

        if segmentation_image is not None:
            mask = (seg_data != segmentation_image.labels[i]) & (seg_data > 0)
            _, clean_profile, _ = _extract_single_growth_curve(
                image, pos, radii, mask=mask
            )
            flux_clean[i] = clean_profile
            contamination[i] = profile - clean_profile
        else:
            flux_clean[i] = profile

    return {
        "radius": radii,
        "flux": flux,
        "background_var": background_var,
        "flux_clean": flux_clean,
        "contamination": contamination,
    }


# pylint: enable=too-many-arguments,too-many-positional-arguments


# pylint: disable=too-many-arguments,too-many-positional-arguments
def detect_and_segment(
    image,
    background=None,
    n_sigma=3.0,
    n_pixels=10,
    deblend=True,
    bkg_box_size=(50, 50),
):
    """Detect sources and produce a segmentation image.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Image data with background included.
    background : float, 2D `~numpy.ndarray`, or None
        Background level to subtract. If a scalar, a uniform background
        is subtracted. If a 2D array, it is subtracted pixel-wise.
        If ``None`` (default), a 2D background map is estimated via
        :func:`estimate_background`.
    n_sigma : float
        Detection significance threshold (passed to ``detect_threshold``).
    n_pixels : int
        Minimum number of connected pixels for a valid source.
    deblend : bool
        If ``True`` (default), run ``deblend_sources`` to separate overlapping
        sources using photutils' deblender.
    bkg_box_size : tuple of int, optional
        Box size for 2D background estimation. Used only when
        ``background`` is ``None``. Default ``(50, 50)``.

    Returns
    -------
    segmentation_image : `~photutils.segmentation.SegmentationImage`
    catalog : `~photutils.segmentation.SourceCatalog`
    background_map : 2D `~numpy.ndarray`
        The background map that was subtracted. Full-resolution 2D array
        even when a scalar was provided.
    """
    if background is None:
        background, _ = estimate_background(image, box_size=bkg_box_size)

    if np.isscalar(background):
        background_map = np.full_like(image, background, dtype=float)
        scalar_bkg = background
    else:
        background_map = np.asarray(background, dtype=float)
        scalar_bkg = float(np.nanmedian(background_map))

    subtracted = image - background_map
    threshold = detect_threshold(image, n_sigma=n_sigma, background=scalar_bkg)
    seg = detect_sources(subtracted, threshold, n_pixels=n_pixels)

    if deblend:
        seg = deblend_sources(
            subtracted,
            seg,
            n_pixels=n_pixels,
            n_levels=32,
            contrast=0.001,
            progress_bar=False,
        )

    catalog = SourceCatalog(subtracted, seg)

    return seg, catalog, background_map


# pylint: enable=too-many-arguments,too-many-positional-arguments


def cross_match(input_positions, detected_positions, tolerance=5.0):
    """Match input positions to detected positions by nearest neighbor.

    Efficient grid-binned implementation (see :mod:`gcphotom.match`).

    Parameters
    ----------
    input_positions : 2D `~numpy.ndarray`
        ``(n_input, 2)`` array of ``(x, y)`` coordinates.
    detected_positions : 2D `~numpy.ndarray`
        ``(n_detected, 2)`` array of ``(x, y)`` coordinates.
    tolerance : float
        Maximum distance in pixels for a valid match.

    Returns
    -------
    result : dict
        Dictionary with keys:

        * ``match_indices``: 1D array of integers. For each input position,
          the index in ``detected_positions`` (or ``-1`` if unmatched).
        * ``match_distances``: 1D array of floats. Distance in pixels
          (or ``inf`` if unmatched).
    """
    return _cross_match(input_positions, detected_positions, tolerance=tolerance)
