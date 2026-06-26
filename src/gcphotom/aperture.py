import numpy as np
from photutils.profiles import CurveOfGrowth
from photutils.segmentation import (
    detect_sources,
    detect_threshold,
    deblend_sources,
    SourceCatalog,
)

from .match import cross_match as _cross_match  # re-export for backward compat


def estimate_error(image, background, read_noise):
    """Compute per-pixel error estimate.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Image data.
    background : float
        Estimated background level.
    read_noise : float
        Read noise standard deviation.

    Returns
    -------
    error : 2D `~numpy.ndarray`
        Per-pixel 1-sigma error.
    """
    signal = np.maximum(image - background, 0)
    return np.sqrt(signal + read_noise**2)


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


# pylint: disable=too-many-arguments,too-many-positional-arguments
def extract_growth_curves(
    image, positions, radii=None, error=None, segmentation_image=None
):
    """Extract circular growth curves for multiple sources.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Image data (should be background-subtracted).
    positions : 2D `~numpy.ndarray`
        ``(n_sources, 2)`` array of ``(x, y)`` coordinates.
    radii : 1D `~numpy.ndarray` or None
        Aperture radii in pixels. Defaults to 10 logarithmically spaced
        values between 0.5 and 30 pixels.
    error : 2D `~numpy.ndarray` or None
        Per-pixel 1-sigma error.
    segmentation_image : `~photutils.segmentation.SegmentationImage` or None
        Segmentation map from :func:`detect_and_segment`. If provided,
        contamination from neighboring sources is estimated.

    Returns
    -------
    result : dict
        Dictionary with keys:

        * ``radius``: 1D array of aperture radii.
        * ``flux``: 2D array ``(n_sources, n_radii)`` of cumulative flux.
        * ``flux_err``: 2D array ``(n_sources, n_radii)`` of flux
          uncertainties.
        * ``flux_clean``: 2D array ``(n_sources, n_radii)`` of flux with
          neighboring sources masked. Present only if ``segmentation_image``
          is provided.
        * ``contamination``: 2D array ``(n_sources, n_radii)`` of
          contaminating flux (``flux - flux_clean``). Present only if
          ``segmentation_image`` is provided.
    """
    if radii is None:
        radii = np.logspace(np.log10(3), np.log10(30), num=10)

    n_sources = len(positions)
    n_radii = len(radii)
    flux = np.zeros((n_sources, n_radii))
    flux_err = np.zeros((n_sources, n_radii))

    if segmentation_image is not None:
        flux_clean = np.zeros((n_sources, n_radii))
        seg_data = segmentation_image.data

    for i, pos in enumerate(positions):
        _, profile, profile_err = _extract_single_growth_curve(
            image, pos, radii, error=error
        )
        flux[i] = profile
        flux_err[i] = profile_err

        if segmentation_image is not None:
            mask = (seg_data != segmentation_image.labels[i]) & (seg_data > 0)
            _, clean_profile, _ = _extract_single_growth_curve(
                image, pos, radii, mask=mask
            )
            flux_clean[i] = clean_profile

    result = {
        "radius": radii,
        "flux": flux,
        "flux_err": flux_err,
    }
    if segmentation_image is not None:
        result["flux_clean"] = flux_clean
        result["contamination"] = flux - flux_clean

    return result


# pylint: enable=too-many-arguments,too-many-positional-arguments


def detect_and_segment(image, background, n_sigma=3.0, n_pixels=10, deblend=True):
    """Detect sources and produce a segmentation image.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Image data with background included.
    background : float
        Background level to subtract.
    n_sigma : float
        Detection significance threshold (passed to ``detect_threshold``).
    n_pixels : int
        Minimum number of connected pixels for a valid source.
    deblend : bool
        If ``True``, run ``deblend_sources`` to separate overlapping sources.

    Returns
    -------
    segmentation_image: `~photutils.segmentation.SegmentationImage`.
    catalog: `~photutils.segmentation.SourceCatalog`.
    """
    subtracted = image - background
    threshold = detect_threshold(image, n_sigma=n_sigma, background=background)
    seg = detect_sources(subtracted, threshold, n_pixels=n_pixels)

    if deblend:
        try:
            seg = deblend_sources(
                subtracted,
                seg,
                n_pixels=n_pixels,
                n_levels=32,
                contrast=0.001,
                progress_bar=False,
            )
        except (ModuleNotFoundError, ImportError):
            pass  # skimage not available; skip deblending

    catalog = SourceCatalog(subtracted, seg)

    return seg, catalog


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
