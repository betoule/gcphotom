"""Monte Carlo flux bias analysis.

Usage::

    # Run a new simulation
    mc_flux_bias_coverage.py run [OPTIONS]

    # Re-plot saved results
    mc_flux_bias_coverage.py show PATH [OPTIONS]
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
    n_sources: int = typer.Option(
        1000,
        "--n-sources",
        help="Number of sources per realization.",
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
    # Output
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Save MC results to this file (.pkl appended if missing).",
    ),
):
    """Run a Monte Carlo simulation and produce bias plots."""
    if simulator == Simulator.astropy:
        simulate_fn = gcp.simulate_image
    elif simulator == Simulator.galsim_auto:
        simulate_fn = partial(gcp.simulate_image_galsim, method="auto")
    else:
        simulate_fn = partial(
            gcp.simulate_image_galsim, method="phot", max_phot_sources=max_phot_sources
        )

    print(f"Simulator backend: {simulator.value}")
    print(f"Image size: {ny}x{nx},  {n_sources} sources")
    print(f"Moffat: gamma={gamma}, alpha={alpha}")
    print(f"Noise: background={background}, read_noise={read_noise}")

    cfg = gcp.montecarlo.SimulationConfig(
        n_sources=n_sources,
        shape=(ny, nx),
        gamma=gamma,
        alpha=alpha,
        background=background,
        read_noise=read_noise,
        n_pixels=n_pixels,
        fit_kwargs={"learning_rate": learning_rate, "niter": niter},
    )

    mc = gcp.montecarlo.MonteCarlo(
        cfg, n_realizations=n_realizations, seed=seed, simulate_fn=simulate_fn
    )
    results = mc.run(verbose=True)
    print(f"\nCompleted {len(results)}/{mc.n_realizations} realizations.")

    if output:
        path = gcp.montecarlo.save_results(output, results)
        print(f"Saved results to {path}")

    _plot(results, tag=simulator.value, nbins=nbins)


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
