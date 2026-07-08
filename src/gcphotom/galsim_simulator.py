"""GalSim-based image simulation."""

# pylint: disable=no-value-for-parameter
# (false positives from GalSim C++ extension — bandpass defaults to None)

import numpy as np
from tqdm.auto import tqdm

import galsim as gs

from .simulator import make_realistic_source_catalog


def _fits_to_galsim_coords(x, y, nx, ny):
    """Convert 0-indexed FITS pixel to GalSim world coordinates."""
    return float(x) - nx / 2.0 + 0.5, float(y) - ny / 2.0 + 0.5


def simulate_image_galsim(
    shape=(1024, 1024),
    catalog=None,
    n_sources=1000,
    gamma=3,
    alpha=3,
    background=100,
    read_noise=5,
    seed=None,
    *,
    method="auto",
    max_phot_sources=100,
):
    """Simulate an image using GalSim.

    Parameters
    ----------
    shape : tuple of int
        Image shape ``(ny, nx)``.
    catalog : `~astropy.table.Table` or None
        Source catalog with columns ``x``, ``y``, ``flux``.
        If ``None``, a catalog is generated via
        :func:`~gcphotom.make_realistic_source_catalog`.
    n_sources : int
        Number of sources when ``catalog`` is ``None``.
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
    method : str
        ``"auto"`` (FFT convolution, then Poisson noise) or
        ``"phot"`` (photon shooting with intrinsic Poisson noise).
    max_phot_sources : int
        Maximum number of sources per batch when photon shooting.
        Only relevant when ``method="phot"``.

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

    ny, nx = shape
    gs_rng = gs.BaseDeviate(seed)
    np_rng = np.random.default_rng(seed)

    if method == "phot":
        image = None
        n_batches = (len(catalog) + max_phot_sources - 1) // max_phot_sources
        for start in tqdm(
            range(0, len(catalog), max_phot_sources),
            total=n_batches,
            desc="Photon shooting",
            unit="batch",
            leave=False,
        ):
            batch = catalog[start : start + max_phot_sources]
            profiles = []
            for row in batch:
                flux = float(row["flux"])
                if flux <= 0:
                    continue
                dx, dy = _fits_to_galsim_coords(row["x"], row["y"], nx, ny)
                moffat = gs.Moffat(beta=alpha, scale_radius=gamma, flux=flux)
                profiles.append(moffat.shift(dx=dx, dy=dy))
            if not profiles:
                continue
            scene = gs.Add(profiles)
            batch_img = scene.drawImage(
                method="phot", nx=nx, ny=ny, scale=1.0, dtype=np.float32, rng=gs_rng
            )
            image = batch_img if image is None else image + batch_img

        if image is None:
            img = np.zeros(shape, dtype=np.float64)
        else:
            img = np.asarray(image.array, dtype=np.float64)

        if background > 0:
            img += np_rng.poisson(background, size=shape).astype(np.float64)

    else:  # method == "auto"
        profiles = []
        for row in catalog:
            flux = float(row["flux"])
            if flux <= 0:
                continue
            dx, dy = _fits_to_galsim_coords(row["x"], row["y"], nx, ny)
            moffat = gs.Moffat(beta=alpha, scale_radius=gamma, flux=flux)
            profiles.append(moffat.shift(dx=dx, dy=dy))

        if profiles:
            scene = gs.Add(profiles)
            image = scene.drawImage(
                method="auto", nx=nx, ny=ny, scale=1.0, dtype=np.float32
            )
            img = np.asarray(image.array, dtype=np.float64)
        else:
            img = np.zeros(shape, dtype=np.float64)

        img += background

        if background > 0 or read_noise > 0:
            img = np_rng.poisson(np.maximum(img, 0)).astype(np.float64)

    if read_noise > 0:
        img += np_rng.normal(0, read_noise, shape)

    return img, catalog
