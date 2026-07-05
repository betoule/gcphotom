"""Examine PSF fit residuals vs aperture radius on a simulated image.

The hypothesis is that pixelization effects at small apertures cause the
bias seen in reconstructed alpha, gamma, and flux in the MC study. This
script runs the growth-curve fitter and plots weighted residuals vs
aperture radius, also splitting by source brightness to reveal flux-
dependent systematics.
"""

import numpy as np
import matplotlib.pyplot as plt

import gcphotom as gcp

# --- Simulation ----------------------------------------------------------
shape = (512, 512)
cat = gcp.make_realistic_source_catalog(500, shape=shape, seed=42)

# Bright, isolated sources on a clean grid for the main diagnostic
# (also add fainter ones to check flux dependence)
bright = cat["flux"] > 2000
n_bright = bright.sum()
print(f"Sources: {len(cat)} total, {n_bright} with flux > 2000")

img, sim_cat = gcp.simulate_image(
    shape, cat, gamma=3.0, alpha=3.0, background=100.0, read_noise=5.0, seed=42
)

# --- Detection + growth-curve extraction --------------------------------
seg, det_cat, bkg_map, bkg_var = gcp.detect_and_segment(img, n_pixels=5)
cog = gcp.extract_growth_curves(
    img,
    det_cat,
    segmentation_image=seg,
    background_variance=bkg_var,
    show_progress=True,
)

# --- Growth-curve fit ---------------------------------------------------
fitter = gcp.Fitter(cog)
bf, extra = fitter.fit(show_progress=True, niter=3000, learning_rate=1e-2)
fitter.detect_contamination(bf)
bf, extra = fitter.fit(
    show_progress=True, niter=3000, learning_rate=1e-2, compute_uncertainty=True
)
result = fitter.results(bf)

print(f"\nFitted gamma = {result['gamma']:.3f}  (true = 3.0)")
print(f"Fitted alpha = {result['alpha']:.3f}  (true = 3.0)")

# --- Main diagnostic plot: model ratio + weighted residuals ------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]}
)
fitter.plot_PSF(bf, axes=(ax1, ax2), gamma_true=3.0, alpha_true=3.0)

# Overlay the fitted and true parameter values as text
ax1.text(
    0.05,
    0.05,
    f"True: γ=3.0, α=3.0\nFitted: γ={result['gamma']:.2f}, α={result['alpha']:.2f}",
    transform=ax1.transAxes,
    fontsize=8,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
)

fig.suptitle("Growth-curve fit: PSF profile and model ratio", fontsize=12, y=0.98)
fig.tight_layout()
fig.savefig("psf_residual_main.png", dpi=150)
print("\nSaved psf_residual_main.png")

# --- Flux-split diagnostic ----------------------------------------------
# See if residuals at small radii depend on source brightness.
wr_all = np.asarray(fitter.weighted_residuals(bf, mask=True))
flux_est = result["flux"]
kept = fitter.kept[: len(flux_est)] if len(flux_est) < len(fitter.kept) else fitter.kept
goods = np.asarray(fitter.goods)
radii = np.asarray(fitter.radii)

# Map kept sources back to the original flux ordering
flux_full = np.full(fitter._orig_n, np.nan)
flux_full[fitter.kept] = flux_est

# Build masked arrays for three brightness tiers
flux_sorted = np.sort(flux_full[~np.isnan(flux_full)])
if len(flux_sorted) >= 6:
    lo = flux_sorted[len(flux_sorted) // 3]
    hi = flux_sorted[2 * len(flux_sorted) // 3]
else:
    lo = hi = np.median(flux_sorted)

tiers = {
    "Faint": flux_full < lo,
    "Medium": (flux_full >= lo) & (flux_full < hi),
    "Bright": flux_full >= hi,
}

fig2, axs2 = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)

for ax, (label, mask) in zip(axs2, tiers.items()):
    m = np.asarray(mask)
    if m.sum() == 0:
        ax.set_title(f"{label} (no sources)")
        continue
    # Map mask to current fitter source indices
    kept_idx = np.where(fitter.kept)[0]
    mask_cur = m[fitter.kept]  # boolean over current (kept) sources
    if mask_cur.sum() == 0:
        ax.set_title(f"{label} (no sources)")
        continue
    wr_sub = wr_all[:, mask_cur]

    wr_med = np.nanmedian(wr_sub, axis=1)
    wr_p16 = np.nanpercentile(wr_sub, 16, axis=1)
    wr_p84 = np.nanpercentile(wr_sub, 84, axis=1)

    ax.axhline(0, color="gray", ls="--", alpha=0.5, zorder=0)
    ax.errorbar(
        radii,
        wr_med,
        yerr=[wr_med - wr_p16, wr_p84 - wr_med],
        fmt="o",
        color="C0",
        capsize=2,
        markersize=4,
        label=f"n={mask_cur.sum()}",
    )
    ax.set_xlabel("Aperture radius [pix]")
    ax.set_title(f"{label} (n={mask_cur.sum()})")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)

axs2[0].set_ylabel("Weighted residual\n(data − model) / σ")
fig2.suptitle("Weighted residuals split by source brightness", fontsize=12)
fig2.tight_layout()
fig2.savefig("psf_residual_by_flux.png", dpi=150)
print("Saved psf_residual_by_flux.png")

# --- Print summary statistics -------------------------------------------
print("\n--- Weighted residual statistics by radius ---")
for i, r in enumerate(radii):
    wr_r = wr_all[i, :]
    valid = wr_r[~np.isnan(wr_r)]
    if len(valid) > 0:
        print(
            f"  r={r:5.1f} pix: median={np.nanmedian(valid):+7.3f}  "
            f"mad={np.nanmedian(np.abs(valid)):.3f}  "
            f"n={len(valid):5d}"
        )

print("\nDone. Inspect the saved PNG files and the open figure windows.")
