"""Check the impact of the model_shape fix on parameter bias.

Runs a small MC (10 realizations, 300 sources) with the updated
simulator and prints fitted gamma/alpha vs truth.
"""

import numpy as np
import matplotlib.pyplot as plt

import gcphotom as gcp

cfg = gcp.montecarlo.SimulationConfig(
    n_sources=300,
    shape=(512, 512),
    gamma=3.0,
    alpha=3.0,
    background=100.0,
    read_noise=5.0,
    n_pixels=5,
    fit_kwargs={"learning_rate": 1e-2, "niter": 2000},
)

mc = gcp.montecarlo.MonteCarlo(cfg, n_realizations=10, seed=42)
results = mc.run(verbose=True)
print(f"\nCompleted {len(results)}/{mc.n_realizations} realizations.\n")

# --- Extract fitted gamma/alpha per realization -------------------------
gammas, alphas = [], []
gammas_fb, alphas_fb = [], []
for r in results:
    bf = r.get("GC", {}).get("best_fit", {})
    gammas.append(float(bf.get("gamma", np.nan)))
    alphas.append(float(bf.get("alpha", np.nan)))
    bf_fb = r.get("GC (fixed back)", {}).get("best_fit", {})
    gammas_fb.append(float(bf_fb.get("gamma", np.nan)))
    alphas_fb.append(float(bf_fb.get("alpha", np.nan)))

gammas = np.array(gammas)
alphas = np.array(alphas)
gammas_fb = np.array(gammas_fb)
alphas_fb = np.array(alphas_fb)

print(f"         {'gamma':>8s}  {'alpha':>8s}")
print(f"Truth:   {3.0:>8.3f}  {3.0:>8.3f}")
print(f"GC mean: {np.nanmean(gammas):>8.3f}  {np.nanmean(alphas):>8.3f}")
print(f"GC std:  {np.nanstd(gammas):>8.3f}  {np.nanstd(alphas):>8.3f}")
print(f"GC bias: {np.nanmean(gammas)-3.0:>+8.3f}  {np.nanmean(alphas)-3.0:>+8.3f}")
print()
print(
    f"GC (fix back) mean: {np.nanmean(gammas_fb):>8.3f}  {np.nanmean(alphas_fb):>8.3f}"
)
print(f"GC (fix back) std:  {np.nanstd(gammas_fb):>8.3f}  {np.nanstd(alphas_fb):>8.3f}")
print(
    f"GC (fix back) bias: {np.nanmean(gammas_fb)-3.0:>+8.3f}  {np.nanmean(alphas_fb)-3.0:>+8.3f}"
)

# --- Flux bias -----------------------------------------------------------
flux_stats = gcp.montecarlo.compute_flux_bias(results, nbins=8)
gcp.montecarlo.plot_flux_bias(flux_stats)
plt.savefig("mc_flux_bias_fixed.png", dpi=150)
print("\nSaved mc_flux_bias_fixed.png")

# --- Scalar parameter plot ----------------------------------------------
gcp.montecarlo.plot_scalar_bias(results)
plt.savefig("mc_scalar_bias_fixed.png", dpi=150)
print("Saved mc_scalar_bias_fixed.png")
