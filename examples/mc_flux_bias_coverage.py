"""Monte Carlo bias and coverage analysis for flux estimators.

Runs ~100 realizations of a controlled test setup and produces a
two-panel plot showing the bias and 1-sigma coverage of each flux
estimator as a function of simulated flux.
"""

import matplotlib.pyplot as plt
import numpy as np

import gcphotom as gcp

# --- Simulation setup ---------------------------------------------------

n_sources_side = 6
shape = (256, 256)
fmin, fmax = 200, 5e5
gamma, alpha = 3.0, 3.0
background = 100.0
read_noise = 5.0
n_realizations = 100

catalog = gcp.make_test_source_catalog(
    n_sources_side=n_sources_side, shape=shape, fmin=fmin, fmax=fmax
)

cfg = gcp.montecarlo.SimulationConfig(
    catalog=catalog,
    shape=shape,
    gamma=gamma,
    alpha=alpha,
    background=background,
    read_noise=read_noise,
    n_pixels=5,
    fit_kwargs={"learning_rate": 1e-2, "niter": 2000},
    nbins=10,
)

# --- Run Monte Carlo ----------------------------------------------------

mc = gcp.montecarlo.MonteCarlo(cfg, n_realizations=n_realizations, seed=42)
summary = mc.run(verbose=True)
print(f"\nCompleted {summary.realized}/{summary.total} realizations.")

# --- Compute and plot bias/coverage -------------------------------------

stats = gcp.montecarlo.compute_bias_coverage(
    mc.results,
    nbins=10,
    sigma_levels=(1.0, 2.0, 3.0),
)

gcp.montecarlo.plot_bias_coverage(stats, sigma_level=1.0)
plt.tight_layout()
plt.savefig("mc_bias_coverage.png", dpi=150)
print("Saved mc_bias_coverage.png")
