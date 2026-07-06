"""Monte Carlo flux bias analysis.

Runs ~100 realizations with random catalogs and produces a single
plot of flux bias vs simulated flux for several photometry methods.
"""

import matplotlib
import importlib.util as _util

if _util.find_spec("tkinter") is not None:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gcphotom as gcp

# --- Simulation setup ---------------------------------------------------

cfg = gcp.montecarlo.SimulationConfig(
    n_sources=1000,
    shape=(1024, 1024),
    gamma=3.0,
    alpha=3.0,
    background=100.0,
    read_noise=5.0,
    n_pixels=5,
    fit_kwargs={"learning_rate": 1e-2, "niter": 2000},
)

# --- Run Monte Carlo ----------------------------------------------------

mc = gcp.montecarlo.MonteCarlo(cfg, n_realizations=100, seed=42)
results = mc.run(verbose=True)
print(f"\nCompleted {len(results)}/{mc.n_realizations} realizations.")

# --- Flux bias analysis -------------------------------------------------

flux_stats = gcp.montecarlo.compute_flux_bias(results, nbins=10)
gcp.montecarlo.plot_flux_bias(flux_stats)
plt.savefig("mc_flux_bias.png", dpi=150)
print("Saved mc_flux_bias.png")

gcp.montecarlo.plot_estimation_times(results)
plt.savefig("mc_estimation_times.png", dpi=150)
print("Saved mc_estimation_times.png")
plt.show()
