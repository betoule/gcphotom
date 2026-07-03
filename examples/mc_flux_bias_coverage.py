"""Monte Carlo bias and coverage analysis.

Runs ~100 realizations with random catalogs and produces separate plots
for flux estimators, background bias, and nuisance parameter recovery.

Parameter types
---------------
- **Flux** — per-source flux bias and coverage vs simulated flux.
- **Background** — per-source background bias vs simulated flux.
- **Nuisance** (gamma, alpha) — global scalar parameters fitted once
  per realization, shown as histograms across realizations.
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

mc = gcp.montecarlo.MonteCarlo(cfg, n_realizations=10, seed=42)
summary = mc.run(verbose=True)
print(f"\nCompleted {summary.realized}/{summary.total} realizations.")

# --- Build all estimators (grouped by type) -----------------------------

estimators = gcp.montecarlo.build_default_estimators(cfg)

# ---------------------------------------------------------------------------
# 1. Flux bias and coverage vs simulated flux
# ---------------------------------------------------------------------------

flux_stats = gcp.montecarlo.compute_bias_coverage(
    mc.results,
    estimators=estimators["flux"],
    nbins=10,
    sigma_levels=(1.0, 2.0, 3.0),
)

gcp.montecarlo.plot_bias_coverage(flux_stats, sigma_level=1.0)
plt.tight_layout()
plt.savefig("mc_flux_bias_coverage.png", dpi=150)
print("Saved mc_flux_bias_coverage.png")
plt.close()

# ---------------------------------------------------------------------------
# 2. Background bias vs simulated flux
# ---------------------------------------------------------------------------

bg_stats = gcp.montecarlo.compute_bias_coverage(
    mc.results,
    estimators=estimators["background"],
    nbins=10,
    sigma_levels=(1.0,),
)

gcp.montecarlo.plot_background_bias(bg_stats)
plt.tight_layout()
plt.savefig("mc_background_bias.png", dpi=150)
print("Saved mc_background_bias.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. Nuisance parameter recovery (gamma, alpha)
# ---------------------------------------------------------------------------

nuisance_stats = gcp.montecarlo.compute_nuisance_stats(
    mc.results, estimators=estimators["nuisance"]
)

gcp.montecarlo.plot_nuisance_summary(nuisance_stats)
plt.tight_layout()
plt.savefig("mc_nuisance_recovery.png", dpi=150)
print("Saved mc_nuisance_recovery.png")
