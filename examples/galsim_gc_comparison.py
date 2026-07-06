"""Compare growth-curve flux reconstruction on GalSim photon-shooting vs FFT images.

Simulates the three Gaia fields with both rendering methods and runs the
full photometry pipeline (detect, segment, growth curves, GC fit) on each.
Plots per-source flux reconstruction error for both methods side by side.
"""

import time
import numpy as np
import matplotlib

# Test backends by trying to load their modules before importing pyplot.
# This avoids the lazy-load failure where use() succeeds at registration
# but fails later when creating a figure.
import importlib.util as _util

_has_tk = _util.find_spec("tkinter") is not None
if _has_tk:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from functools import partial

import gcphotom as gcp
from gcphotom.plots import binplot
import galsim as gs

# --- Parameters ------------------------------------------------------------

shape = (1024, 1024)
ny, nx = shape
zeropoint = 25.0
gamma = 3.0
alpha = 3.0
background = 100.0
read_noise = 5.0

fields = [
    {"label": "COSMOS", "ra": 150.0, "dec": 2.2, "comment": "b ≈ +42°"},
    {"label": "Gemini", "ra": 95.0, "dec": 35.0, "comment": "b ≈ +15°"},
    {"label": "Cygnus", "ra": 300.0, "dec": 40.0, "comment": "Galactic plane, b ≈ 0°"},
]

# --- Estimators ------------------------------------------------------------

cfg = gcp.montecarlo.SimulationConfig(
    shape=shape,
    gamma=gamma,
    alpha=alpha,
    background=background,
    read_noise=read_noise,
    n_pixels=5,
    fit_kwargs={"learning_rate": 1e-2, "niter": 400},
)

estimators = {
    "GC": partial(gcp.montecarlo.gc_estimator, fit_kwargs=cfg.fit_kwargs),
}

rng = gs.BaseDeviate(42)
colors = {"phot": "C0", "FFT": "C1"}
markers = {"phot": "o", "FFT": "s"}


# --- Helpers ---------------------------------------------------------------


def simulate_image_galsim(
    catalog,
    shape,
    gamma,
    alpha,
    background,
    read_noise,
    method="phot",
    rng=None,
    max_phot_sources=2000,
):
    """Simulate an image using GalSim with a constant background and read noise.

    For photon shooting with many sources, sources are batched to avoid
    GalSim's O(n_sources × n_photons) memory allocation in Sum._shoot.
    """
    ny, nx = shape
    kwargs = dict(nx=nx, ny=ny, scale=1.0, dtype=np.float32)
    if method == "phot":
        kwargs["rng"] = rng

    t0 = time.perf_counter()

    if method == "phot" and len(catalog) > max_phot_sources:
        n_batches = (len(catalog) + max_phot_sources - 1) // max_phot_sources
        result = None
        for i in range(n_batches):
            batch = catalog[i * max_phot_sources : (i + 1) * max_phot_sources]
            profiles = []
            for row in batch:
                flux = float(row["flux"])
                if flux <= 0:
                    continue
                moffat = gs.Moffat(beta=alpha, scale_radius=gamma, flux=flux)
                x = float(row["x"]) - nx / 2.0 + 0.5
                y = float(row["y"]) - ny / 2.0 + 0.5
                moffat = moffat.shift(dx=x, dy=y)
                profiles.append(moffat)
            scene = gs.Add(profiles)
            batch_img = scene.drawImage(method=method, **kwargs)
            result = batch_img if result is None else result + batch_img
        image = result
    else:
        profiles = []
        for row in catalog:
            flux = float(row["flux"])
            if flux <= 0:
                continue
            moffat = gs.Moffat(beta=alpha, scale_radius=gamma, flux=flux)
            x = float(row["x"]) - nx / 2.0 + 0.5
            y = float(row["y"]) - ny / 2.0 + 0.5
            moffat = moffat.shift(dx=x, dy=y)
            profiles.append(moffat)
        scene = gs.Add(profiles)
        image = scene.drawImage(method=method, **kwargs)

    dt = time.perf_counter() - t0

    image.array[:] += background
    if read_noise > 0 and rng is not None:
        gnoise = gs.GaussianNoise(rng=rng, sigma=read_noise)
        image.addNoise(gnoise)
    return image.array, dt


# --- Run simulations and pipeline -----------------------------------------

results = {}  # {field_label: {"phot": result_dict, "FFT": result_dict}}

for field in fields:
    print(f"\n{'='*65}")
    print(f"  {field['label']}  ({field['comment']})")
    print(f"{'='*65}")

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
    wcs.wcs.cdelt = [-0.00028, 0.00028]
    wcs.wcs.crval = [field["ra"], field["dec"]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    print("  Querying Gaia DR3 ...", flush=True)
    catalog = gcp.make_gaia_source_catalog(
        wcs, shape, zeropoint, g_max=20.0, margin_arcmin=5.0
    )
    print(f"  {len(catalog)} sources", flush=True)

    field_results = {}
    for method in ("phot", "FFT"):
        print(f"  Simulating ({method}) ...", flush=True)
        img, dt = simulate_image_galsim(
            catalog,
            shape,
            gamma,
            alpha,
            background,
            read_noise,
            method="phot" if method == "phot" else "auto",
            rng=rng,
        )
        print(f"    draw: {dt:.2f}s", flush=True)

        pipe_t0 = time.perf_counter()
        result = gcp.montecarlo.run_pipeline(img, catalog, cfg, estimators)
        pipe_t = time.perf_counter() - pipe_t0
        print(f"    pipeline: {pipe_t:.2f}s", flush=True)

        n_matched = np.sum(np.isfinite(result["sim_cat"]["flux"]))
        print(f"    {n_matched} matched sources", flush=True)
        field_results[method] = result

    results[field["label"]] = field_results


# --- Plot flux reconstruction errors --------------------------------------

print(f"\n{'='*65}")
print("  Plotting flux reconstruction errors")
print(f"{'='*65}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

for idx, field in enumerate(fields):
    ax = axes[idx]
    label = field["label"]

    for method in ("phot", "FFT"):
        result = results[label][method]
        flux_truth = np.asarray(result["sim_cat"]["flux"])
        flux_est = np.asarray(result["GC"]["best_fit"]["flux"])
        valid = np.isfinite(flux_truth) & np.isfinite(flux_est) & (flux_truth > 0)
        error = (flux_est[valid] / flux_truth[valid] - 1.0) * 100.0

        # Bin for trend line
        xb, yb, yerr = binplot(
            flux_truth[valid],
            error,
            nbins=10,
            logbins=True,
            method="median",
            noplot=True,
        )
        ax.errorbar(
            xb,
            yb,
            yerr=yerr,
            fmt=markers[method],
            color=colors[method],
            label=f"{method} (binned)",
            zorder=5,
        )
        # Scatter of individual sources
        ax.scatter(
            flux_truth[valid],
            error,
            s=8,
            alpha=0.3,
            color=colors[method],
            marker=markers[method],
        )

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("True flux [ADU]")
    ax.set_ylabel("Flux error [%]")
    ax.set_title(f"{label}  ({len(results[label]['phot']['sim_cat'])} det.)")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_ylim(-50, 50)

plt.suptitle(
    "Growth-curve flux reconstruction — GalSim photon shooting vs FFT",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("galsim_gc_comparison.png", dpi=150)
print("  Saved galsim_gc_comparison.png")

# Also plot a per-field summary of median absolute error per method
print()
print("  Median |error| per method:")
print(f"  {'Field':<12} {'Phot median |err|':<18} {'FFT median |err|':<18}")
print(f"  {'-'*12} {'-'*18} {'-'*18}")
for field in fields:
    label = field["label"]
    for method in ("phot", "FFT"):
        result = results[label][method]
        flux_truth = np.asarray(result["sim_cat"]["flux"])
        flux_est = np.asarray(result["GC"]["best_fit"]["flux"])
        valid = np.isfinite(flux_truth) & np.isfinite(flux_est) & (flux_truth > 0)
        error = np.abs((flux_est[valid] / flux_truth[valid] - 1.0) * 100.0)
        med = np.nanmedian(error)
        print(f"  {label+' '+method:<12} {med:<18.2f}")

plt.show()
print("\nDone.")
