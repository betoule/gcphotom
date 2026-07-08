"""GalSim-based image simulation."""

# pylint: disable=no-value-for-parameter
# (false positives from GalSim C++ extension — bandpass defaults to None)

import numpy as np
from tqdm.auto import tqdm

import galsim as gs

from .chromatic import (
    build_chromatic_psf,
    build_sensor,
    sed_from_color,
    tophat_bandpass,
)
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
    # Chromatic / sensor options
    chromatic=False,
    bandpass="r",
    sensor=False,
    bf_strength=0.0,
    diffusion_factor=0.0,
    zenith_angle=30.0,
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
    chromatic : bool
        Enable chromatic rendering (PSF + SED).  Requires ``method="phot"``
        and a ``bp_rp`` column in the catalog.
    bandpass : str
        Bandpass name for chromatic mode: ``"g"``, ``"r"``, ``"i"``, ``"z"``.
    sensor : bool
        Enable SiliconSensor (brighter-fatter + charge diffusion).
        Requires ``method="phot"`` and ``chromatic=True``.
    bf_strength : float
        Brighter-fatter strength (0 = off, 1 = LSST nominal).
    diffusion_factor : float
        Charge diffusion factor (0 = off, 1 = LSST nominal).
    zenith_angle : float
        Zenith angle in degrees for chromatic DCR.

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

    if chromatic or sensor:
        method = "phot"

    if method == "phot":
        image = None
        n_batches = (len(catalog) + max_phot_sources - 1) // max_phot_sources

        # Build shared chromatic / sensor resources once.
        psf = None
        bp = None
        sens = None
        if chromatic:
            psf = build_chromatic_psf(
                gamma=gamma,
                alpha=alpha,
                zenith_angle=zenith_angle,
                pixel_scale=1.0,
            )
            bp = tophat_bandpass(bandpass)
        if sensor:
            sens = build_sensor(
                bf_strength=bf_strength,
                diffusion_factor=diffusion_factor,
                seed=seed,
            )

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

                if chromatic:
                    sed = sed_from_color(float(row.get("bp_rp", 0.0)))
                    sed_norm = sed.withFlux(flux, bandpass=bp)
                    star = gs.DeltaFunction() * sed_norm
                    obj = gs.Convolve([psf, star]).shift(dx=dx, dy=dy)
                    profiles.append(obj)
                else:
                    moffat = gs.Moffat(beta=alpha, scale_radius=gamma, flux=flux)
                    profiles.append(moffat.shift(dx=dx, dy=dy))

            if not profiles:
                continue
            scene = gs.Add(profiles)

            draw_kwargs = {
                "method": "phot",
                "nx": nx,
                "ny": ny,
                "scale": 1.0,
                "dtype": np.float32,
                "rng": gs_rng,
            }
            if bp is not None:
                draw_kwargs["bandpass"] = bp
            if sens is not None:
                draw_kwargs["sensor"] = sens

            batch_img = scene.drawImage(**draw_kwargs)
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
