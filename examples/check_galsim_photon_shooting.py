"""Verify GalSim photon-shooting across multiple realizations.

Simulates many independent photon-shooting realizations of a single
well-centered source (zero background, zero read noise).  The scatter
across realizations demonstrates that photon noise is inherent in
photon shooting, while the mean converges to the analytic expectation.
"""

import numpy as np
from astropy.table import Table

import matplotlib.pyplot as plt

import gcphotom as gcp
import galsim as gs

# --- Parameters ------------------------------------------------------------

shape = (129, 129)
ny, nx = shape
gamma_true = 3.0
alpha_true = 3.0
flux_true = 50000.0
x_src, y_src = 64.0, 64.0
n_realizations = 100

radii = np.logspace(np.log10(3), np.log10(30), num=12)
n_radii = len(radii)

cat = Table({"x": [x_src], "y": [y_src], "flux": [flux_true]})

# Analytic expectation
cumul_true = (
    np.asarray(gcp.gcmodel.moffat_flux(radii, gamma_true, alpha_true)) * flux_true
)

# Build the shifted Moffat once
base_moffat = gs.Moffat(beta=alpha_true, scale_radius=gamma_true, flux=flux_true)
dx = x_src - nx / 2.0 + 0.5
dy = y_src - ny / 2.0 + 0.5
moffat = base_moffat.shift(dx=dx, dy=dy)

# FFT reference (noiseless — drawn once)
img_fft = np.asarray(
    moffat.drawImage(method="auto", nx=nx, ny=ny, scale=1.0, dtype=np.float32).array,
    dtype=float,
)
cog_fft = gcp.extract_growth_curves(
    img_fft,
    cat,
    radii=radii,
    method="sinc",
    background_variance=np.zeros_like(img_fft),
    show_progress=False,
)
cumul_fft = cog_fft["flux_clean"][0]
annular_fft = gcp.gcmodel.annular_fluxes(cumul_fft)

# Multiple photon-shooting realizations
all_cumul = np.empty((n_realizations, n_radii))
all_centroids = np.empty((n_realizations, 2))

for i in range(n_realizations):
    rng = gs.BaseDeviate(42 + i)
    img = np.asarray(
        moffat.drawImage(
            method="phot", nx=nx, ny=ny, scale=1.0, dtype=np.float32, rng=rng
        ).array,
        dtype=float,
    )
    # Centroid
    total = img.sum()
    xc = np.sum(np.arange(nx) * img.sum(axis=0)) / total
    yc = np.sum(np.arange(ny) * img.sum(axis=1)) / total
    all_centroids[i] = [xc, yc]
    # Growth curve
    cog = gcp.extract_growth_curves(
        img,
        cat,
        radii=radii,
        method="sinc",
        background_variance=np.zeros_like(img),
        show_progress=False,
    )
    all_cumul[i] = cog["flux_clean"][0]

# Statistics across realizations
cumul_mean = np.mean(all_cumul, axis=0)
cumul_std = np.std(all_cumul, axis=0, ddof=1)
cumul_sem = cumul_std / np.sqrt(n_realizations)

annular_all = np.asarray([gcp.gcmodel.annular_fluxes(c) for c in all_cumul])
annular_mean = np.mean(annular_all, axis=0)
annular_std = np.std(annular_all, axis=0, ddof=1)
annular_sem = annular_std / np.sqrt(n_realizations)
annular_true = gcp.gcmodel.annular_fluxes(cumul_true)

# Centroid statistics
centroid_mean = np.mean(all_centroids, axis=0)
centroid_std = np.std(all_centroids, axis=0, ddof=1)


# --- Print summary ---------------------------------------------------------

print("=" * 60)
print("  Centroid scatter across realizations")
print("=" * 60)
print(f"  Input position:             ({x_src:.2f}, {y_src:.2f})")
print(f"  Mean centroid:              ({centroid_mean[0]:.4f}, {centroid_mean[1]:.4f})")
print(f"  Std of centroid:            ({centroid_std[0]:.4f}, {centroid_std[1]:.4f})")
print(
    f"  Expected σ_centroid ≈ FWHM/(2.355√N) = {1.3 / (2.355 * np.sqrt(flux_true)):.4f}"
)

print()
print("=" * 60)
print("  Flux recovery (cumulative at largest aperture)")
print("=" * 60)
print(f"  Injected flux:         {flux_true:.0f} ADU")
print(
    f"  Mean phot (sinc):      {cumul_mean[-1]:.0f} ± {cumul_std[-1]:.0f} ADU  "
    f"(pull: {(cumul_mean[-1] - cumul_true[-1]) / cumul_std[-1]:.2f})"
)
print(
    f"  FFT (sinc):            {cumul_fft[-1]:.0f} ADU  "
    f"({(cumul_fft[-1] / cumul_true[-1] - 1) * 100:+.3f}%)"
)
print(
    f"  Expected Poisson σ:    {np.sqrt(flux_true):.0f} ADU  "
    f"({np.sqrt(flux_true) / flux_true * 100:.2f}%)"
)

print()
print("=" * 60)
print("  Annular profiles — mean ± std across realizations")
print("=" * 60)
print(
    f"  {'radius':>8s}  {'ann_mean':>10s}  {'ann_true':>10s}  {'ann_std':>10s}  "
    f"{'Poisson σ':>10s}  {'pull':>8s}"
)
print("  " + "  ".join(["-" * 8, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 8]))
for i, r in enumerate(radii):
    poisson_sigma = np.sqrt(annular_true[i]) if annular_true[i] > 0 else np.nan
    pull = (
        (annular_mean[i] - annular_true[i]) / annular_std[i]
        if annular_std[i] > 0
        else np.nan
    )
    print(
        f"  {r:>8.2f}  {annular_mean[i]:>10.1f}  {annular_true[i]:>10.1f}  "
        f"{annular_std[i]:>10.1f}  {poisson_sigma:>10.1f}  {pull:>+7.2f}"
    )


# --- Figure ----------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes

# Row 1: Cumulative curves
for i in range(n_realizations):
    ax1.plot(radii, all_cumul[i], color="C0", alpha=0.15, lw=0.8)
ax1.plot(
    radii, cumul_mean, "o-", color="C0", label=f"Phot mean (n={n_realizations})", ms=4
)
ax1.plot(radii, cumul_fft, "s-", color="C1", label="FFT (noiseless)", ms=4)
ax1.plot(radii, cumul_true, "k-", label="Analytic Moffat", lw=1.5)
ax1.set_ylabel("Cumulative flux [ADU]")
ax1.set_title("Cumulative growth curves")
ax1.legend(frameon=False, fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.axhline(0, color="gray", ls="--", alpha=0.5)
ax2.errorbar(
    radii,
    cumul_mean - cumul_true,
    yerr=cumul_sem,
    fmt="o",
    color="C0",
    capsize=2,
    ms=4,
    label="Phot mean ± SEM",
)
ax2.plot(radii, cumul_fft - cumul_true, "s-", color="C1", label="FFT residual", ms=4)
ax2.set_ylabel("Extracted − Analytic [ADU]")
ax2.set_title("Cumulative residuals")
ax2.legend(frameon=False, fontsize=8)
ax2.grid(True, alpha=0.3)

ax3.axhline(0, color="gray", ls="--", alpha=0.5)
ax3.errorbar(
    radii,
    cumul_mean - cumul_fft,
    yerr=cumul_sem,
    fmt="o",
    color="C0",
    capsize=2,
    ms=4,
    label="Phot - FFT",
)
ax3.set_ylabel("Phot − FFT [ADU]")
ax3.set_title("Photon shooting vs FFT")
ax3.legend(frameon=False, fontsize=8)
ax3.grid(True, alpha=0.3)

# Row 2: Annular profiles and noise
for i in range(n_realizations):
    ann = gcp.gcmodel.annular_fluxes(all_cumul[i])
    ax4.plot(radii, ann, color="C0", alpha=0.15, lw=0.8)
ax4.plot(
    radii, annular_mean, "o-", color="C0", label=f"Phot mean (n={n_realizations})", ms=4
)
ax4.plot(radii, annular_fft, "s-", color="C1", label="FFT (noiseless)", ms=4)
ax4.plot(radii, annular_true, "k-", label="Analytic", lw=1.5)
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_ylabel("Annular flux [ADU]")
ax4.set_title("Annular profiles")
ax4.legend(frameon=False, fontsize=8)
ax4.grid(True, alpha=0.3)

ax5.axhline(0, color="gray", ls="--", alpha=0.5)
ax5.errorbar(
    radii,
    annular_mean - annular_true,
    yerr=annular_sem,
    fmt="o",
    color="C0",
    capsize=2,
    ms=4,
    label="Phot mean ± SEM",
)
ax5.plot(
    radii, annular_fft - annular_true, "s-", color="C1", label="FFT residual", ms=4
)
ax5.set_ylabel("Extracted − Analytic [ADU]")
ax5.set_title("Annular residuals")
ax5.legend(frameon=False, fontsize=8)
ax5.grid(True, alpha=0.3)

# RMS scatter vs Poisson expectation
annular_rms = np.sqrt(np.mean((annular_all - annular_true[None, :]) ** 2, axis=0))
poisson_sigma = np.where(annular_true > 0, np.sqrt(annular_true), np.nan)
ax6.plot(radii, annular_rms, "o-", color="C0", label="Measured RMS", ms=4)
ax6.plot(radii, poisson_sigma, "s--", color="C3", label="√(annular_true)", ms=4)
ax6.set_ylabel("Scatter [ADU]")
ax6.set_title("Noise: measured vs Poisson expectation")
ax6.legend(frameon=False, fontsize=8)
ax6.grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel("Aperture radius [pix]")

fig.suptitle(
    f"GalSim photon shooting — {n_realizations} realizations\n"
    f"Moffat γ={gamma_true}, α={alpha_true}, flux={flux_true} ADU, "
    f"source at ({x_src}, {y_src})",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("check_galsim_photon_shooting.png", dpi=150)
print(f"\nSaved check_galsim_photon_shooting.png")

plt.show()
