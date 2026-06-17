import numpy as np
from astropy.modeling.models import Moffat2D
from astropy.table import Table
from photutils.datasets import apply_poisson_noise, make_model_image


def make_source_catalog(n_sources, shape=(1024, 1024), margin=20, min_sep=5, seed=None):
    """Generate a realistic source catalog.

    Parameters
    ----------
    n_sources : int
        Target number of sources.
    shape : tuple of int
        Image shape (ny, nx).
    margin : int
        Pixel margin from image edge.
    min_sep : int
        Minimum separation between sources in pixels.
    seed : int or None
        Random seed.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Table with columns ``x``, ``y``, ``flux``.
    """
    rng = np.random.default_rng(seed)

    lo_x, hi_x = margin, shape[1] - margin
    lo_y, hi_y = margin, shape[0] - margin

    xs, ys = [], []
    placed = 0
    attempts = 0
    max_attempts = n_sources * 20

    while placed < n_sources and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(lo_x, hi_x)
        y = rng.uniform(lo_y, hi_y)

        if placed == 0:
            xs.append(x)
            ys.append(y)
            placed += 1
            continue

        dx = np.array(xs) - x
        dy = np.array(ys) - y
        dist = np.sqrt(dx**2 + dy**2)

        if dist.min() >= min_sep:
            xs.append(x)
            ys.append(y)
            placed += 1

    catalog = Table()
    catalog["x"] = xs
    catalog["y"] = ys

    log_fmin, log_fmax = np.log10(100), np.log10(1e6)
    log_flux = rng.uniform(log_fmin, log_fmax, len(catalog))
    catalog["flux"] = 10**log_flux

    return catalog


def make_moffat_psf(alpha, beta):
    """Build a Moffat2D model with the given shape parameters.

    Parameters
    ----------
    alpha : float
        Moffat scale parameter (gamma) in pixels.
    beta : float
        Moffat shape parameter.

    Returns
    -------
    model : `~astropy.modeling.models.Moffat2D`
        Moffat model with amplitude=1, gamma=alpha, beta=beta,
        centered at (0, 0).
    """
    return Moffat2D(amplitude=1, gamma=alpha, alpha=beta, x_0=0, y_0=0)


def simulate_image(
    shape,
    catalog,
    alpha,
    beta,
    background=0.0,
    read_noise=0.0,
    seed=None,
):
    """Simulate a 2D image with Moffat PSF sources and noise.

    Parameters
    ----------
    shape : tuple of int
        Image shape (ny, nx).
    catalog : `~astropy.table.Table`
        Source catalog with columns ``x``, ``y``, ``flux``.
    alpha : float
        Moffat scale parameter in pixels.
    beta : float
        Moffat shape parameter.
    background : float
        Constant background level in ADU.
    read_noise : float
        Gaussian read noise standard deviation in ADU.
    seed : int or None
        Random seed.

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Simulated image.
    """
    psf = make_moffat_psf(alpha, beta)

    params = Table()
    total_flux = catalog["flux"]
    amplitude = total_flux * (beta - 1) / (alpha**2 * np.pi)
    params["amplitude"] = amplitude
    params["x_0"] = catalog["x"]
    params["y_0"] = catalog["y"]
    params["gamma"] = alpha
    params["alpha"] = beta

    model_shape = max(21, int(6 * alpha))
    image = make_model_image(
        shape,
        psf,
        params,
        model_shape=(model_shape, model_shape),
        discretize_method="oversample",
        discretize_oversample=10,
    )

    image = image + background

    if np.any(image >= 0):
        image = apply_poisson_noise(image, seed=seed)

    if read_noise > 0:
        rng = np.random.default_rng(seed)
        image = image + rng.normal(0, read_noise, shape)

    return image
