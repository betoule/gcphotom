import numpy as np
from astropy.modeling.models import Moffat2D
from astropy.table import Table
from photutils.datasets import apply_poisson_noise, make_model_image


def make_realistic_source_catalog(
    n_sources=1000, shape=(1024, 1024), margin=20, seed=None
):
    """Generate a realistic source catalog.

    Parameters
    ----------
    n_sources : int
        Target number of sources.
    shape : tuple of int
        Image shape (ny, nx).
    margin : int
        Pixel margin from image edge.
    seed : int or None
        Random seed.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Table with columns ``x``, ``y``, ``flux``.
    """
    rng = np.random.default_rng(seed)

    xs = rng.uniform(margin, shape[1] - margin, n_sources)
    ys = rng.uniform(margin, shape[0] - margin, n_sources)

    catalog = Table()
    catalog["x"] = xs
    catalog["y"] = ys

    log_fmin, log_fmax = np.log10(100), np.log10(1e6)
    log_flux = rng.uniform(log_fmin, log_fmax, n_sources)
    catalog["flux"] = 10**log_flux

    return catalog


def make_test_source_catalog(n_sources_side=4, shape=(128, 128), fmin=100, fmax=1e6):
    """Generate a source catalog for test purposes.

    The sources are regularly spaced on the focal plane.

    Parameters
    ----------
    n_sources_side : int
        The total number of sources will be n_sources_side**2.
    shape : tuple of int
        Image shape (ny, nx).
    fmin : float
        Flux in ADU of the faintest source (upper left)
    fmax : float
        Flux in ADU of the brightest source (bottom right)

    Returns
    -------
    catalog : `~astropy.table.Table`
        Table with columns ``x``, ``y``, ``flux``.
    """
    xs = np.linspace(0, shape[0], n_sources_side + 2)[1:-1]
    ys = np.linspace(0, shape[1], n_sources_side + 2)[1:-1]

    xs, ys = np.meshgrid(xs, ys)

    catalog = Table()
    catalog["x"] = xs.flatten()
    catalog["y"] = ys.flatten()

    log_fmin, log_fmax = np.log10(fmin), np.log10(fmax)
    catalog["flux"] = np.logspace(log_fmin, log_fmax, len(xs.flatten()))

    return catalog


def simulate_image(
    shape=(1024, 1024),
    catalog=None,
    n_sources=1000,
    gamma=3,
    alpha=3,
    background=100,
    read_noise=5,
    seed=None,
):
    """Simulate a 2D image with Moffat PSF sources and noise.

    Parameters
    ----------
    shape : tuple of int
        Image shape (ny, nx).
    catalog : `~astropy.table.Table` or None
        Source catalog with columns ``x``, ``y``, ``flux``.
        If ``None``, a catalog is generated via ``make_source_catalog``.
    n_sources : int
        Number of sources when ``catalog`` is ``None``. Ignored if
        ``catalog`` is provided.
    gamma : float
        Moffat scale parameter in pixels.
    alpha : float
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
    catalog : `~astropy.table.Table`
        Source catalog with injected truth values.
    """
    if catalog is None:
        catalog = make_realistic_source_catalog(
            n_sources=n_sources, shape=shape, seed=seed
        )

    psf = Moffat2D(amplitude=1, gamma=gamma, alpha=alpha, x_0=0, y_0=0)

    params = Table()
    params["amplitude"] = catalog["flux"] * (alpha - 1) / (gamma**2 * np.pi)
    params["x_0"] = catalog["x"]
    params["y_0"] = catalog["y"]
    params["gamma"] = gamma
    params["alpha"] = alpha

    model_shape = max(21, int(6 * gamma))
    image = make_model_image(
        shape,
        psf,
        params,
        model_shape=(model_shape, model_shape),
        discretize_method="oversample",
        discretize_oversample=10,
    )

    image = image + background
    image = apply_poisson_noise(image, seed=seed)

    if read_noise > 0:
        rng = np.random.default_rng(seed)
        image = image + rng.normal(0, read_noise, shape)

    return image, catalog
