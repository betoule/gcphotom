"""Standard PSF photometry using photutils' empirical PSF workflow."""

import numpy as np

from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import EPSFBuilder, PSFPhotometry, SourceGrouper, extract_stars

from .background import estimate_background


def _as_xy(sources):
    """Convert sources to x, y arrays.

    Accepts a SourceCatalog (with ``x_centroid``/``y_centroid``),
    an Astropy Table (with ``x``/``y``), or a 2D array of shape ``(N, 2)``.
    """
    if isinstance(sources, np.ndarray):
        arr = np.asarray(sources, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 0], arr[:, 1]
    if hasattr(sources, "x_centroid") and hasattr(sources, "y_centroid"):
        return np.asarray(sources.x_centroid, dtype=float), np.asarray(
            sources.y_centroid, dtype=float
        )
    if hasattr(sources, "__getitem__"):
        return np.asarray(sources["x"], dtype=float), np.asarray(
            sources["y"], dtype=float
        )
    raise TypeError(
        "sources must be a SourceCatalog, Table with 'x'/'y', or (N,2) ndarray"
    )


def _select_epsf_stars(
    x,
    y,
    flux,
    nstars=20,
    min_separation=15,
):
    """Select the brightest isolated stars for ePSF building.

    Parameters
    ----------
    x, y : array_like
        Source positions.
    flux : array_like
        Source fluxes (any aperture measure).
    nstars : int
        Number of stars to select.
    min_separation : float
        Minimum separation in pixels for isolation.

    Returns
    -------
    indices : ndarray
        Indices into the source arrays of selected stars.
    """
    grouper = SourceGrouper(min_separation=min_separation)
    groups = grouper(x, y)
    group_sizes = {}
    for gid in groups:
        group_sizes[gid] = group_sizes.get(gid, 0) + 1
    isolated = np.array([group_sizes[g] == 1 for g in groups])

    sorted_isolated = np.argsort(-flux)[isolated[np.argsort(-flux)]]
    return sorted_isolated[:nstars]


def psf_photometry(
    image,
    sources,
    nstars=20,
    min_separation=15,
    fit_shape=25,
    extract_size=51,
    oversampling=1,
    aperture_radius=10.0,
    epsf_maxiters=10,
    background=None,
):
    """Standard PSF photometry with an empirical PSF built from the image.

    Subtracts the background (estimated or provided) before building the
    ePSF and running photometry.  Builds an ePSF from the brightest
    isolated stars, then fits all detected sources using
    ``photutils.psf.PSFPhotometry``.

    Parameters
    ----------
    image : 2D ndarray
        Input image (background included).
    sources : SourceCatalog, Table, or ndarray
        Detected sources.  A ``SourceCatalog`` (with ``x_centroid``/
        ``y_centroid``/``segment_flux``), an Astropy ``Table`` with
        ``x``/``y``/``flux`` columns, or an ``(N, 2)`` array of
        positions.  If flux information is not available, all sources
        are considered for ePSF selection.
    nstars : int
        Number of bright isolated stars to use for ePSF building.
    min_separation : float
        Minimum separation in pixels for a star to be considered
        "isolated" for ePSF building.
    fit_shape : int
        Fitting box size in pixels (passed to ``PSFPhotometry``).
        Should be large enough to capture most of the PSF flux.
    extract_size : int
        Cutout size in pixels for ePSF star extraction.
        Should be large enough to capture the full PSF wings.
    oversampling : int
        ePSF oversampling factor (passed to ``EPSFBuilder``).
    aperture_radius : float
        Aperture radius in pixels for initial flux estimation in
        ``PSFPhotometry``.
    epsf_maxiters : int
        Maximum ePSF building iterations.
    background : float, 2D ndarray or None
        Background level to subtract.  If a scalar, a uniform background
        is subtracted.  If a 2D array, it is subtracted pixel-wise.
        If ``None`` (default), a 2D background map is estimated and its
        median is used for thresholding.

    Returns
    -------
    results : Table
        Photometry results table from ``PSFPhotometry``.  Key columns
        include ``x_fit``, ``y_fit``, ``flux_fit``, and ``flux_err``.
    epsf_result : EPSFBuildResult
        ePSF building result object containing the fitted ``epsf``
        model and convergence diagnostics.
    """
    x, y = _as_xy(sources)
    n_sources = len(x)

    # Subtract background
    if background is None:
        background, _ = estimate_background(image)
    if np.isscalar(background):
        image_sub = image - background
    else:
        image_sub = image - np.asarray(background, dtype=float)

    # Extract flux for star selection
    if hasattr(sources, "segment_flux"):
        flux = np.asarray(sources.segment_flux, dtype=float)
    elif hasattr(sources, "__getitem__") and "flux" in sources.colnames:
        flux = np.asarray(sources["flux"], dtype=float)
    else:
        flux = np.ones(n_sources)

    # Select bright isolated stars for ePSF
    epsf_indices = _select_epsf_stars(
        x, y, flux, nstars=nstars, min_separation=min_separation
    )

    if len(epsf_indices) < 3:
        raise ValueError(
            f"Only {len(epsf_indices)} isolated star(s) found "
            f"(need at least 3 for ePSF building). "
            f"Try reducing min_separation={min_separation} or nstars={nstars}."
        )

    # Build ePSF
    epsf_tab = Table()
    epsf_tab["x"] = x[epsf_indices]
    epsf_tab["y"] = y[epsf_indices]

    nd = NDData(image_sub)
    stars = extract_stars(nd, epsf_tab, size=extract_size)

    epsf_builder = EPSFBuilder(
        oversampling=oversampling,
        maxiters=epsf_maxiters,
        progress_bar=False,
    )
    epsf_result = epsf_builder(stars)
    epsf = epsf_result.epsf

    # Run PSF photometry on all sources
    init_params = Table()
    init_params["x_0"] = x
    init_params["y_0"] = y

    psf_phot = PSFPhotometry(
        epsf,
        fit_shape=fit_shape,
        aperture_radius=aperture_radius,
        grouper=SourceGrouper(min_separation=min_separation),
        progress_bar=False,
    )
    results = psf_phot(image_sub, init_params=init_params)

    return results, epsf_result
