"""GalSim photon-shooting simulation of three Gaia fields.

Demonstrates photon-shooting image simulation for COSMOS, Gemini, and
Cygnus fields using Gaia DR3 source catalogs.  Reports wall-clock time
per field and saves a comparison figure.

Chromatic and sensor effects are ignored for this first demonstration.
"""

import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.wcs import WCS

import gcphotom as gcp
import galsim as gs

# --- Parameters ------------------------------------------------------------

shape = (512, 512)
zeropoint = 25.0
gamma = 3.0  # Moffat scale radius (pixels)
alpha = 3.0  # Moffat beta (shape parameter)
background = 100.0  # ADU
read_noise = 5.0  # ADU

fields = [
    {"label": "COSMOS", "ra": 150.0, "dec": 2.2, "comment": "b ≈ +42°"},
    {"label": "Gemini", "ra": 95.0, "dec": 35.0, "comment": "b ≈ +15°"},
    {"label": "Cygnus", "ra": 300.0, "dec": 40.0, "comment": "Galactic plane, b ≈ 0°"},
]

rng = gs.BaseDeviate(42)

ny, nx = shape


# --- Helpers ---------------------------------------------------------------


def simulate_image_galsim(
    catalog, shape, gamma, alpha, background, read_noise, method="phot", rng=None
):
    """Simulate an image using GalSim.

    Parameters
    ----------
    catalog : astropy.Table
        Must have 'x', 'y', 'flux' columns (0-indexed FITS coords).
    shape : tuple
        (ny, nx) image shape.
    gamma, alpha : float
        Moffat scale radius and beta.
    background : float
        Constant background level in ADU.
    read_noise : float
        Gaussian read noise sigma in ADU.
    method : str
        GalSim draw method ('phot' or 'auto').
    rng : galsim.BaseDeviate or None
        Random number generator.

    Returns
    -------
    image : np.ndarray
        Simulated image.
    dt : float
        Wall-clock seconds for the draw step (sources only).
    """
    ny, nx = shape

    # Build profiles at image-center-relative coordinates.
    # drawImage places world (0, 0) at the image center pixel.
    # A 0-indexed FITS pixel (x, y) maps to GalSim world coord
    #   wx = x - nx/2 + 0.5, wy = y - ny/2 + 0.5
    # (the +0.5 accounts for GalSim's 1-indexed pixel convention).
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

    # Draw the scene
    t0 = time.perf_counter()
    kwargs = dict(nx=nx, ny=ny, scale=1.0, dtype=np.float32)
    if method == "phot":
        kwargs["rng"] = rng
    image = scene.drawImage(method=method, **kwargs)
    dt = time.perf_counter() - t0

    # Add constant background
    image.array[:] += background

    # Add Gaussian read noise
    if read_noise > 0 and rng is not None:
        gnoise = gs.GaussianNoise(rng=rng, sigma=read_noise)
        image.addNoise(gnoise)

    return image.array, dt


# --- Run simulations -------------------------------------------------------

results = []

for field in fields:
    print(f"\n{'='*65}")
    print(f"  {field['label']}  ({field['comment']})")
    print(f"{'='*65}")

    # --- WCS for this field ------------------------------------------------
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2.0, ny / 2.0]
    wcs.wcs.cdelt = [-0.00028, 0.00028]
    wcs.wcs.crval = [field["ra"], field["dec"]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # --- Gaia catalog ------------------------------------------------------
    print("  Querying Gaia DR3 ...", flush=True)
    catalog = gcp.make_gaia_source_catalog(
        wcs, shape, zeropoint, g_max=20.0, margin_arcmin=5.0
    )
    n_src = len(catalog)
    total_flux = catalog["flux"].sum()
    print(f"  {n_src} sources, total flux: {total_flux:.1f} ADU", flush=True)

    if n_src == 0:
        print("  No sources — skipping")
        continue

    # --- GalSim photon shooting --------------------------------------------
    print("  Photon shooting ...", flush=True)
    img_phot, dt_phot = simulate_image_galsim(
        catalog,
        shape,
        gamma,
        alpha,
        background,
        read_noise,
        method="phot",
        rng=rng,
    )
    print(f"  Wall time: {dt_phot:.2f} s", flush=True)

    # --- GalSim FFT (for comparison) ---------------------------------------
    print("  FFT ...", flush=True)
    img_fft, dt_fft = simulate_image_galsim(
        catalog,
        shape,
        gamma,
        alpha,
        background,
        read_noise,
        method="auto",
        rng=rng,
    )
    print(f"  Wall time: {dt_fft:.2f} s", flush=True)

    results.append(
        {
            "field": field,
            "catalog": catalog,
            "img_phot": img_phot,
            "img_fft": img_fft,
            "dt_phot": dt_phot,
            "dt_fft": dt_fft,
            "n_src": n_src,
            "total_flux": total_flux,
        }
    )


# --- Report ----------------------------------------------------------------

print(f"\n{'='*65}")
print("  Timing summary")
print(f"{'='*65}")
print(
    f"  {'Field':<12} {'Sources':<8} {'Total flux':<14} {'Phot (s)':<12} {'FFT (s)':<12}"
)
print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*12} {'-'*12}")
for r in results:
    print(
        f"  {r['field']['label']:<12} {r['n_src']:<8} {r['total_flux']:<14.1f} {r['dt_phot']:<12.2f} {r['dt_fft']:<12.2f}"
    )


# --- Figures ---------------------------------------------------------------

print(f"\n  Saving figures ...")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

for idx, r in enumerate(results):
    # Photon shooting images
    ax = axes[0, idx]
    vmin, vmax = np.percentile(r["img_phot"], [5, 99.8])
    ax.imshow(r["img_phot"], vmin=vmin, vmax=vmax, origin="lower", cmap="gray")
    ax.set_title(
        f"{r['field']['label']} — Photon shooting\n{r['n_src']} sources, {r['dt_phot']:.1f}s"
    )
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")

    # FFT images
    ax = axes[1, idx]
    vmin, vmax = np.percentile(r["img_fft"], [5, 99.8])
    ax.imshow(r["img_fft"], vmin=vmin, vmax=vmax, origin="lower", cmap="gray")
    ax.set_title(
        f"{r['field']['label']} — FFT\n{r['n_src']} sources, {r['dt_fft']:.1f}s"
    )
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")

plt.suptitle("GalSim simulation — Moffat PSF, ZP=25, G<20, 512×512", fontsize=14)
plt.tight_layout()
plt.savefig("galsim_gaia_comparison.png", dpi=150)
print(f"  Saved galsim_gaia_comparison.png")

# Difference image
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))
for idx, r in enumerate(results):
    diff = r["img_phot"] - r["img_fft"]
    ax = axes2[idx]
    vmin, vmax = np.percentile(abs(diff), [5, 99.9])
    vmax = max(vmax, 1.0)
    im = ax.imshow(diff, vmin=-vmax, vmax=vmax, origin="lower", cmap="RdBu")
    ax.set_title(f"{r['field']['label']}\nphot - FFT (rms={np.std(diff):.2f})")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Difference: photon shooting − FFT", fontsize=14)
plt.tight_layout()
plt.savefig("galsim_gaia_diff.png", dpi=150)
print(f"  Saved galsim_gaia_diff.png")

print(f"\nDone.")
