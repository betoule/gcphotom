"""Chromatic PSF, SEDs, and sensor effects for GalSim image simulation."""

import galsim as gs
import numpy as np

# hc/k_B in nm·K  (for blackbody SED)
_HC_K = 14387768.78

# Top-hat bandpass edges in nm
_TOPHAT_BANDPASSES: dict[str, tuple[float, float]] = {
    "g": (400.0, 550.0),
    "r": (550.0, 700.0),
    "i": (700.0, 820.0),
    "z": (820.0, 920.0),
}


def tophat_bandpass(name: str) -> gs.Bandpass:
    """Build a top-hat (square) bandpass with AB zeropoint.

    Parameters
    ----------
    name : str
        One of ``"g"``, ``"r"``, ``"i"``, ``"z"``.

    Returns
    -------
    `galsim.Bandpass`
    """
    if name not in _TOPHAT_BANDPASSES:
        raise ValueError(
            f"Unknown bandpass '{name}'. Choose from {list(_TOPHAT_BANDPASSES)}"
        )
    blue, red = _TOPHAT_BANDPASSES[name]
    waves = np.linspace(200, 1200, 1001)
    tp = gs.LookupTable(waves, np.where((waves >= blue) & (waves <= red), 1.0, 0.0))
    return gs.Bandpass(tp, wave_type="nm").withZeropoint("AB")


def sed_from_color(bp_rp: float) -> gs.SED:
    """Build a blackbody SED from BP-RP colour.

    The SED is **unnormalized** — use ``sed.withFlux(flux, bandpass=...)``
    at render time.

    Parameters
    ----------
    bp_rp : float
        Gaia BP-RP colour index.

    Returns
    -------
    `galsim.SED`
    """
    # Approximate Teff from BP-RP for main-sequence stars.
    # Reasonable for bp_rp in [-0.3, 3.0].
    teff = 2000.0 + 10000.0 * np.exp(-bp_rp / 1.2)
    teff = np.clip(teff, 2000.0, 50000.0)

    def _bb(wave_nm):
        w = np.atleast_1d(np.asarray(wave_nm, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 / (w**4 * (np.exp(_HC_K / (w * teff)) - 1))
        result[~np.isfinite(result)] = 0.0
        if np.ndim(wave_nm) == 0:
            return float(result[0])
        return result

    return gs.SED(_bb, wave_type="nm", flux_type="fphotons")


def build_chromatic_psf(
    gamma=3.0,
    alpha=3.0,
    base_wavelength=500.0,
    pixel_scale=0.2,
    atmosphere=True,
    zenith_angle=30.0,
    parallactic_angle=0.0,
    optics=True,
    telescope_diameter=8.36,
    obscuration=1.2 / 8.36,
    aberrations=None,
):
    """Build a combined chromatic PSF.

    Parameters
    ----------
    gamma : float
        Moffat scale radius (pixels) at *base_wavelength*.
    alpha : float
        Moffat shape parameter.
    base_wavelength : float
        Reference wavelength in nm.
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    atmosphere : bool
        Include chromatic atmospheric PSF.
    zenith_angle : float
        Zenith angle in degrees (for DCR).
    parallactic_angle : float
        Parallactic angle in degrees.
    optics : bool
        Include chromatic optical PSF.
    telescope_diameter : float
        Telescope primary mirror diameter in metres.
    obscuration : float
        Central obscuration ratio.
    aberrations : list or None
        Zernike coefficients in waves at *base_wavelength*.

    Returns
    -------
    `galsim.ChromaticObject`
    """
    if aberrations is None:
        aberrations = [0.0] * 4

    # Base Moffat PSF at reference wavelength (in arcsec).
    # Convert gamma from pixels to arcsec.
    gamma_arcsec = gamma * pixel_scale
    base = gs.Moffat(beta=alpha, scale_radius=gamma_arcsec, flux=1.0)

    components = []

    if atmosphere:
        atm = gs.ChromaticAtmosphere(
            base,
            base_wavelength=base_wavelength,
            scale_unit=gs.arcsec,
            zenith_angle=zenith_angle * gs.degrees,
            parallactic_angle=parallactic_angle * gs.degrees,
        )
        components.append(atm)

    if optics:
        opt = gs.ChromaticOpticalPSF(
            lam=base_wavelength,
            diam=telescope_diameter,
            obscuration=obscuration,
            aberrations=aberrations,
            scale_unit=gs.arcsec,
        )
        components.append(opt)

    if not components:
        # Fallback: monochromatic Moffat wrapped as ChromaticObject
        return base

    psf = gs.Convolve(components) if len(components) > 1 else components[0]

    # Pre-compute PSF at a grid of wavelengths for faster rendering.
    waves = np.linspace(300, 1100, 20)
    psf = psf.interpolate(waves)

    return psf


def build_sensor(
    bf_strength: float = 0.0,
    diffusion_factor: float = 0.0,
    sensor_name: str = "lsst_itl_50_8",
    seed: int | None = None,
):
    """Build a SiliconSensor with configurable effects.

    Parameters
    ----------
    bf_strength : float
        Brighter-fatter strength (0 = off, 1 = LSST nominal).
    diffusion_factor : float
        Charge diffusion factor (0 = off, 1 = LSST nominal).
    sensor_name : str
        Sensor model name (e.g. ``"lsst_itl_50_8"``, ``"lsst_e2v_50_8"``).
    seed : int or None
        Random seed.

    Returns
    -------
    `galsim.SiliconSensor` or None
        ``None`` when both effects are off.
    """
    if bf_strength == 0.0 and diffusion_factor == 0.0:
        return None
    # GalSim divides by strength internally, so ensure it is never zero.
    effective_strength = max(bf_strength, 1e-10)
    return gs.SiliconSensor(
        sensor_name,
        strength=effective_strength,
        diffusion_factor=diffusion_factor,
        rng=gs.BaseDeviate(seed),
    )
