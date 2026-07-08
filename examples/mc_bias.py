"""Monte Carlo flux bias analysis.

Usage::

    # Run a new simulation
    mc_bias.py run [OPTIONS]

    # Re-plot saved results
    mc_bias.py show PATH [OPTIONS]
"""

from enum import Enum
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import typer

import gcphotom as gcp

app = typer.Typer(help="Monte Carlo flux bias analysis.")


class Simulator(str, Enum):
    astropy = "astropy"
    galsim_auto = "galsim-auto"
    galsim_phot = "galsim-phot"


GAIA_FIELDS: dict[str, dict] = {
    "COSMOS": {"ra": 150.0, "dec": 2.2, "comment": "b ≈ +42° (low density)"},
    "GEMINI": {"ra": 95.0, "dec": 35.0, "comment": "b ≈ +15° (mid density)"},
    "CYGNUS": {
        "ra": 300.0,
        "dec": 40.0,
        "comment": "Galactic plane, b ≈ 0° (high density)",
    },
}
GAIA_PIXEL_SCALE = 0.00028  # ~1 arcsec/pixel in degrees
GAIA_ZEROPOINT = 25.0
GAIA_G_MAX = 20.0
GAIA_MARGIN_ARCMIN = 5.0


def _default_output_stem(tag, n_realizations, n_sources_int=None, ny=1024, nx=1024):
    """Build a default output stem from non-default simulation parameters.

    Only parameters that differ from their defaults are included, keeping
    the filename short for typical runs while remaining informative.
    """
    parts = [tag]
    if n_realizations != 100:
        parts.append(f"r{n_realizations}")
    if n_sources_int is not None and n_sources_int != 1000:
        parts.append(f"n{n_sources_int}")
    if ny != 1024 or nx != 1024:
        parts.append(f"{ny}x{nx}")
    return f"mc_{'_'.join(parts)}"


def _plot(results, tag, nbins):
    flux_stats = gcp.montecarlo.compute_flux_bias(results, nbins=nbins)
    gcp.montecarlo.plot_flux_bias(flux_stats)
    plt.savefig(f"mc_flux_bias_{tag}.png", dpi=150)
    print(f"Saved mc_flux_bias_{tag}.png")

    gcp.montecarlo.plot_scalar_bias(results)
    plt.savefig(f"mc_scalar_bias_{tag}.png", dpi=150)
    print(f"Saved mc_scalar_bias_{tag}.png")

    gcp.montecarlo.plot_estimation_times(results)
    plt.savefig(f"mc_estimation_times_{tag}.png", dpi=150)
    print(f"Saved mc_estimation_times_{tag}.png")

    plt.show()


@app.command()
def run(
    # Simulator
    simulator: Simulator = typer.Option(
        Simulator.astropy,
        "--simulator",
        help="Image generator backend.",
    ),
    # Monte Carlo
    n_realizations: int = typer.Option(
        100,
        "--n-realizations",
        help="Number of independent realizations.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Master random seed.",
    ),
    # Image geometry
    ny: int = typer.Option(
        1024,
        "--ny",
        help="Image height (pixels).",
    ),
    nx: int = typer.Option(
        1024,
        "--nx",
        help="Image width (pixels).",
    ),
    n_sources: str = typer.Option(
        "1000",
        "--n-sources",
        help="Number of sources (synthetic) or field name for Gaia catalog: "
        "COSMOS, GEMINI, CYGNUS.",
    ),
    gamma: float = typer.Option(
        3.0,
        "--gamma",
        help="Moffat scale radius (pixels).",
    ),
    alpha: float = typer.Option(
        3.0,
        "--alpha",
        help="Moffat shape parameter (beta).",
    ),
    background: float = typer.Option(
        100.0,
        "--background",
        help="Constant background level (ADU).",
    ),
    read_noise: float = typer.Option(
        5.0,
        "--read-noise",
        help="Gaussian read noise sigma (ADU).",
    ),
    max_phot_sources: int = typer.Option(
        100,
        "--max-phot-sources",
        help="Max sources per batch for GalSim photon shooting.",
    ),
    # Chromatic rendering (only with galsim-phot and method="phot")
    chromatic: bool = typer.Option(
        False,
        "--chromatic",
        help="Enable chromatic PSF + SED rendering (galsim-phot only).",
    ),
    bandpass: str = typer.Option(
        "r",
        "--bandpass",
        help="Bandpass for chromatic mode: g, r, i, z.",
    ),
    sensor: bool = typer.Option(
        False,
        "--sensor",
        help="Enable SiliconSensor (brighter-fatter + diffusion).",
    ),
    bf_strength: float = typer.Option(
        0.0,
        "--bf-strength",
        help="Brighter-fatter strength (0=off, 1=LSST nominal).",
    ),
    diffusion_factor: float = typer.Option(
        0.0,
        "--diffusion-factor",
        help="Charge diffusion factor (0=off, 1=LSST nominal).",
    ),
    zenith_angle: float = typer.Option(
        30.0,
        "--zenith-angle",
        help="Zenith angle in degrees for atmospheric chromatic effects.",
    ),
    n_pixels: int = typer.Option(
        5,
        "--n-pixels",
        help="Background mesh size for source detection.",
    ),
    # Fit parameters
    learning_rate: float = typer.Option(
        1e-2,
        "--learning-rate",
        help="Adam learning rate for growth-curve fit.",
    ),
    niter: int = typer.Option(
        2000,
        "--niter",
        help="Number of optimizer iterations.",
    ),
    # Plotting
    nbins: int = typer.Option(
        10,
        "--nbins",
        help="Number of flux bins for bias plot.",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        help="Generate and show bias plots after the run.",
    ),
    # Output
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Save MC results to this file (.pkl appended if missing). "
        "Defaults to an auto-generated name based on non-default parameters.",
    ),
):
    """Run a Monte Carlo simulation (results saved to file by default)."""
    # Parse --n-sources: integer -> synthetic catalog, field name -> Gaia DR3
    gaia_field = None
    n_sources_int = None
    try:
        n_sources_int = int(n_sources)
    except ValueError as exc:
        key = n_sources.upper()
        if key in GAIA_FIELDS:
            gaia_field = key
        else:
            raise typer.BadParameter(
                f"'{n_sources}' is not a valid integer or known Gaia field name. "
                f"Known fields: {', '.join(GAIA_FIELDS)}"
            ) from exc

    if chromatic or sensor:
        if simulator != Simulator.galsim_phot:
            print(
                "Warning: --chromatic/--sensor requires galsim-phot; "
                "switching simulator."
            )
            simulator = Simulator.galsim_phot

        simulate_fn = partial(
            gcp.simulate_image_galsim,
            method="phot",
            max_phot_sources=max_phot_sources,
            chromatic=chromatic,
            bandpass=bandpass,
            sensor=sensor,
            bf_strength=bf_strength,
            diffusion_factor=diffusion_factor,
            zenith_angle=zenith_angle,
        )
    else:
        if simulator == Simulator.astropy:
            simulate_fn = gcp.simulate_image
        elif simulator == Simulator.galsim_auto:
            simulate_fn = partial(gcp.simulate_image_galsim, method="auto")
        else:
            simulate_fn = partial(
                gcp.simulate_image_galsim,
                method="phot",
                max_phot_sources=max_phot_sources,
            )

    print(f"Simulator backend: {simulator.value}")

    if gaia_field:
        field_info = GAIA_FIELDS[gaia_field]
        print(f"Catalog: Gaia DR3, field {gaia_field} ({field_info['comment']})")
        print(f"Image size: {ny}x{nx}")
        print(
            f"Gaia query: G<{GAIA_G_MAX}, ZP={GAIA_ZEROPOINT}, "
            f"margin={GAIA_MARGIN_ARCMIN}'"
        )

        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
        wcs.wcs.cdelt = [-GAIA_PIXEL_SCALE, GAIA_PIXEL_SCALE]
        wcs.wcs.crval = [field_info["ra"], field_info["dec"]]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        catalog_fn = partial(
            gcp.make_gaia_source_catalog,
            wcs=wcs,
            shape=(ny, nx),
            zeropoint=GAIA_ZEROPOINT,
            g_max=GAIA_G_MAX,
            margin_arcmin=GAIA_MARGIN_ARCMIN,
        )

        cfg = gcp.montecarlo.SimulationConfig(
            shape=(ny, nx),
            gamma=gamma,
            alpha=alpha,
            background=background,
            read_noise=read_noise,
            n_pixels=n_pixels,
            fit_kwargs={"learning_rate": learning_rate, "niter": niter},
        )
        mc = gcp.montecarlo.MonteCarlo(
            cfg,
            n_realizations=n_realizations,
            seed=seed,
            simulate_fn=simulate_fn,
            catalog_fn=catalog_fn,
        )
        tag = gaia_field
    else:
        print(f"Catalog: synthetic, {n_sources_int} sources")
        print(f"Moffat: gamma={gamma}, alpha={alpha}")
        print(f"Noise: background={background}, read_noise={read_noise}")

        cfg = gcp.montecarlo.SimulationConfig(
            n_sources=n_sources_int,
            shape=(ny, nx),
            gamma=gamma,
            alpha=alpha,
            background=background,
            read_noise=read_noise,
            n_pixels=n_pixels,
            fit_kwargs={"learning_rate": learning_rate, "niter": niter},
        )
        mc = gcp.montecarlo.MonteCarlo(
            cfg,
            n_realizations=n_realizations,
            seed=seed,
            simulate_fn=simulate_fn,
        )
        tag = simulator.value

    results = mc.run(verbose=True)
    print(f"\nCompleted {len(results)}/{mc.n_realizations} realizations.")

    if not output:
        output = _default_output_stem(
            tag,
            n_realizations,
            n_sources_int=n_sources_int,
            ny=ny,
            nx=nx,
        )
    path = gcp.montecarlo.save_results(output, results)
    print(f"Saved results to {path}")

    if plot:
        _plot(results, tag=tag, nbins=nbins)


@app.command()
def show(
    results_path: str = typer.Argument(..., help="Path to saved results file (.pkl)."),
    nbins: int = typer.Option(
        10,
        "--nbins",
        help="Number of flux bins for bias plot.",
    ),
):
    """Re-plot bias figures from saved results."""
    results = gcp.montecarlo.load_results(results_path)
    print(f"Loaded {len(results)} realizations from {results_path}")
    tag = Path(results_path).stem
    _plot(results, tag=tag, nbins=nbins)


if __name__ == "__main__":
    app()
