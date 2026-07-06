"""Flux bias analysis on a grid of perfectly separated sources.

Uses make_test_source_catalog to place sources on a regular grid, so every
realisation sees the same source positions and fluxes — only the noise varies.
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
    shape=(256, 256),
    gamma=3.0,
    alpha=3.0,
    background=100.0,
    read_noise=5.0,
    n_pixels=5,
    fit_kwargs={"learning_rate": 1e-2, "niter": 2000},
)

# Deterministic factory — same grid catalog every realisation, same flux
# values.  The *seed* argument is accepted for compatibility with the
# factory interface but is ignored (only the noise should vary).
catalog_fn = lambda seed: gcp.make_test_source_catalog(
    n_sources_side=7,
    shape=cfg.shape,
    fmin=100,
    fmax=1e6,
)

# --- Run Monte Carlo ----------------------------------------------------

mc = gcp.montecarlo.MonteCarlo(
    cfg,
    n_realizations=50,
    seed=42,
    catalog_fn=catalog_fn,
)
results = mc.run(verbose=True)
print(f"\nCompleted {len(results)}/{mc.n_realizations} realizations.")

# --- Flux bias analysis -------------------------------------------------

flux_stats = gcp.montecarlo.compute_flux_bias(results, nbins=10)
gcp.montecarlo.plot_flux_bias(flux_stats)
plt.savefig("mc_grid_bias.png", dpi=150)
print("Saved mc_grid_bias.png")
plt.show()
