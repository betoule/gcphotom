"""Gaia-based simulated image and flux reconstruction.

Queries Gaia DR3 for sources in a given sky footprint, simulates an
image, and compares flux reconstruction across three estimators.
"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

import gcphotom as gcp
from gcphotom.plots import binplot

# --- Sky footprint ----------------------------------------------------------

# Simple tangent-plane WCS centred on (RA, Dec) = (76.377, 52.831)
# with ~1 arcsec/pixel sampling (0.00028 deg/pixel).
wcs = WCS(naxis=2)
wcs.wcs.crpix = [512.0, 512.0]
wcs.wcs.cdelt = [-0.00028, 0.00028]
wcs.wcs.crval = [76.377, 52.831]
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

shape = (1024, 1024)
zeropoint = 25.0  # G=25 -> 1 ADU

# --- Gaia source catalog ---------------------------------------------------

print("Querying Gaia DR3 ...")
catalog = gcp.make_gaia_source_catalog(
    wcs, shape, zeropoint, g_max=20.0, margin_arcmin=5.0
)
print(f"  {len(catalog)} sources in the field")

# --- Simulate image --------------------------------------------------------

cfg = gcp.montecarlo.SimulationConfig(
    shape=shape,
    gamma=3.0,
    alpha=3.0,
    background=100.0,
    read_noise=5.0,
    n_pixels=5,
    fit_kwargs={"learning_rate": 1e-2, "niter": 2000},
)

image, truth = gcp.simulate_image(
    shape=shape,
    catalog=catalog,
    gamma=cfg.gamma,
    alpha=cfg.alpha,
    background=cfg.background,
    read_noise=cfg.read_noise,
    seed=42,
)

# --- Display the image -----------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))
vmin, vmax = np.percentile(image, [5, 99.5])
ax.imshow(image, vmin=vmin, vmax=vmax, origin="lower", cmap="gray")
ax.set_title(f"Simulated image — {len(catalog)} Gaia sources")
fig.tight_layout()
plt.savefig("gaia_simulated_image.png", dpi=150)
print("Saved gaia_simulated_image.png")

# --- Run pipeline -----------------------------------------------------------

estimators = {
    "GC": partial(gcp.montecarlo.gc_estimator, fit_kwargs=cfg.fit_kwargs),
    "PSF": gcp.montecarlo.psf_estimator,
    "Aperture + AC": gcp.montecarlo.aperture_estimator,
}

result = gcp.montecarlo.run_pipeline(image, catalog, cfg, estimators)

# --- Flux reconstruction errors -------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

colors = {"GC": "k", "PSF": "b", "Aperture + AC": "m"}

for name in estimators:
    flux_truth = np.asarray(result["sim_cat"]["flux"])
    flux_est = np.asarray(result[name]["best_fit"]["flux"])
    valid = np.isfinite(flux_truth) & np.isfinite(flux_est) & (flux_truth > 0)
    error = (flux_est[valid] / flux_truth[valid] - 1.0) * 100.0

    xb, yb, yerr = binplot(
        flux_truth[valid],
        error,
        nbins=10,
        logbins=True,
        method="median",
        noplot=True,
    )
    ax.errorbar(xb, yb, yerr=yerr, fmt="o", color=colors[name], label=name, zorder=5)

ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("Simulated flux [ADU]")
ax.set_xscale("log")
ax.set_ylabel("Flux error [%]")
ax.legend(loc="best", frameon=False)
fig.tight_layout()
plt.savefig("gaia_flux_error.png", dpi=150)
print("Saved gaia_flux_error.png")
plt.show()
