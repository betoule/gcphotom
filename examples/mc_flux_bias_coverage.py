"""Monte Carlo bias and coverage analysis for flux estimators.

Runs ~100 realizations with random catalogs and produces a two-panel
plot showing the bias (with RMS shaded region) and 1-sigma coverage
of each flux estimator as a function of simulated flux.
"""

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
summary = mc.run(verbose=True)
print(f"\nCompleted {summary.realized}/{summary.total} realizations.")

# --- Compute and plot bias/coverage -------------------------------------

estimators = gcp.montecarlo.build_default_estimators(cfg)
stats = gcp.montecarlo.compute_bias_coverage(
    mc.results,
    estimators=estimators,
    nbins=10,
    sigma_levels=(1.0, 2.0, 3.0),
)

gcp.montecarlo.plot_bias_coverage(stats, sigma_level=1.0)
plt.tight_layout()
plt.savefig("mc_bias_coverage.png", dpi=150)
print("Saved mc_bias_coverage.png")
