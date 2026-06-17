import numpy as np
from photutils.profiles import CurveOfGrowth


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


def _extract_single_growth_curve(image, position, radii, error=None):
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

    Returns
    -------
    radius : 1D `~numpy.ndarray`
        Aperture radii.
    profile : 1D `~numpy.ndarray`
        Cumulative flux at each radius.
    profile_error : 1D `~numpy.ndarray`
        Flux uncertainty at each radius.
    """
    cog = CurveOfGrowth(image, position, radii, error=error)
    perr = (
        cog.profile_error if len(cog.profile_error) > 0 else np.zeros_like(cog.profile)
    )
    return cog.radius, cog.profile, perr


def extract_growth_curves(image, positions, radii=None, error=None):
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

    Returns
    -------
    result : dict
        Dictionary with keys:

        * ``radius``: 1D array of aperture radii.
        * ``flux``: 2D array ``(n_sources, n_radii)`` of cumulative flux.
        * ``flux_err``: 2D array ``(n_sources, n_radii)`` of flux
          uncertainties.
    """
    if radii is None:
        radii = np.logspace(np.log10(3), np.log10(30), num=10)

    n_sources = len(positions)
    n_radii = len(radii)
    flux = np.zeros((n_sources, n_radii))
    flux_err = np.zeros((n_sources, n_radii))

    for i, pos in enumerate(positions):
        _, profile, profile_err = _extract_single_growth_curve(
            image, pos, radii, error=error
        )
        flux[i] = profile
        flux_err[i] = profile_err

    return {
        "radius": radii,
        "flux": flux,
        "flux_err": flux_err,
    }
