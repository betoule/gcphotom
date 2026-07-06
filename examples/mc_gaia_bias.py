"""Gaia-based simulated image and flux reconstruction.

Compares three fields at different stellar densities (COSMOS, Boötes,
Cygnus) to illustrate photometric performance in different conditions.
"""

from functools import partial

import matplotlib
import importlib.util as _util

if _util.find_spec("tkinter") is not None:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

import gcphotom as gcp
from gcphotom.plots import binplot

# --- Simulation parameters -------------------------------------------------

shape = (1024, 1024)
zeropoint = 25.0  # G=25 -> 1 ADU

cfg = gcp.montecarlo.SimulationConfig(
    shape=shape,
    gamma=3.0,
    alpha=3.0,
    background=100.0,
    read_noise=5.0,
    n_pixels=5,
    fit_kwargs={"learning_rate": 1e-2, "niter": 400},
)

estimators = {
    "GC": partial(gcp.montecarlo.gc_estimator, fit_kwargs=cfg.fit_kwargs),
    "PSF": gcp.montecarlo.psf_estimator,
    "Aperture + AC": gcp.montecarlo.aperture_estimator,
}
colors = {"GC": "k", "PSF": "b", "Aperture + AC": "m"}

# --- Three pointings -------------------------------------------------------

fields = [
    {
        "label": "COSMOS",
        "ra": 150.0,
        "dec": 2.2,
        "comment": "b ≈ +42°",
    },
    {
        "label": "Gemini",
        "ra": 95.0,
        "dec": 35.0,
        "comment": "b ≈ +15°",
    },
    {
        "label": "Cygnus",
        "ra": 300.0,
        "dec": 40.0,
        "comment": "Galactic plane, b ≈ 0°",
    },
]

# --- Process each field ----------------------------------------------------

images = []
catalogs = []
results = []

for field in fields:
    print(f"\n{'='*60}")
    print(f"{field['label']}  ({field['comment']})")
    print(f"{'='*60}")

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2.0, shape[0] / 2.0]
    wcs.wcs.cdelt = [-0.00028, 0.00028]
    wcs.wcs.crval = [field["ra"], field["dec"]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    print("  Querying Gaia DR3 ...")
    catalog = gcp.make_gaia_source_catalog(
        wcs, shape, zeropoint, g_max=20.0, margin_arcmin=5.0
    )
    print(f"  {len(catalog)} sources")

    print("  Simulating image ...")
    image, _ = gcp.simulate_image(
        shape=shape,
        catalog=catalog,
        gamma=cfg.gamma,
        alpha=cfg.alpha,
        background=cfg.background,
        read_noise=cfg.read_noise,
        seed=42,
    )
    images.append(image)
    catalogs.append(catalog)

    print("  Running pipeline ...")
    result = gcp.montecarlo.run_pipeline(image, catalog, cfg, estimators)
    results.append(result)


# --- Figure 1: simulated images -------------------------------------------

print("\n--- Saving figures ---")

plt.figure("Simulated images", figsize=(16, 5))

for idx, (field, image, catalog) in enumerate(zip(fields, images, catalogs)):
    ax = plt.subplot(1, 3, idx + 1)
    vmin, vmax = np.percentile(image, [5, 99.5])
    ax.imshow(image, vmin=vmin, vmax=vmax, origin="lower", cmap="gray")
    ax.set_title(f"{field['label']}\n{len(catalog)} sources")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")

plt.suptitle("Simulated images — 1″/pixel, ZP=25, G<20", fontsize=13)
plt.tight_layout()
plt.savefig("gaia_images_comparison.png", dpi=150)
print("  Saved gaia_images_comparison.png")

# --- Figure 2: flux reconstruction errors ----------------------------------

plt.figure("Flux errors", figsize=(16, 5))

for idx, (field, result) in enumerate(zip(fields, results)):
    ax = plt.subplot(1, 3, idx + 1)

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
        ax.errorbar(
            xb, yb, yerr=yerr, fmt="o", color=colors[name], label=name, zorder=5
        )

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Simulated flux [ADU]")
    ax.set_xscale("log")
    ax.set_title(f"{field['label']}  ({len(catalogs[idx])} sources)")
    if idx == 0:
        ax.set_ylabel("Flux error [%]")
        ax.legend(loc="best", frameon=False)

plt.suptitle("Flux reconstruction error — median bias per flux bin", fontsize=13)
plt.tight_layout()
plt.savefig("gaia_flux_error_comparison.png", dpi=150)
print("  Saved gaia_flux_error_comparison.png")
plt.show()

print("\nDone.")
