"""Compare extracted curve of growth to the analytic Moffat profile.

A single noiseless source is placed at the centre of a small image and its
curve of growth is extracted via :func:`gcphotom.extract_growth_curves`
using both the photutils ``"exact"`` method and the Bickerton & Lupton
sinc-interpolated method.  The result is compared to the analytic Moffat
profile, revealing pixelisation biases in standard aperture photometry
and their removal with the sinc method.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import gcphotom as gcp


def analytic_profile(radii, gamma, alpha):
    """Cumulative and annular Moffat flux (normalised to 1 at infinity)."""
    cumul = gcp.gcmodel.moffat_flux(radii, gamma, alpha)
    annular = gcp.gcmodel.annular_fluxes(cumul)
    return np.asarray(cumul), np.asarray(annular)


# --- Single noiseless source at the image centre -------------------------
shape = (129, 129)  # odd so that pixel (64, 64) is the exact centre
gamma_true = 3.0
alpha_true = 3.0

cat = Table({"x": [64.0], "y": [64.0], "flux": [50000.0]})
img, _ = gcp.simulate_image(
    shape,
    cat,
    gamma=gamma_true,
    alpha=alpha_true,
    background=0.0,
    read_noise=0.0,
    seed=42,
)
bkg_var = np.zeros_like(img, dtype=float)

# --- Analytic expectation -------------------------------------------------
radii = np.logspace(np.log10(3), np.log10(30), num=10)
cumul_true, annular_true = analytic_profile(radii, gamma_true, alpha_true)
flux_true = float(cat["flux"][0])
cumul_true_scaled = cumul_true * flux_true
annular_true_scaled = annular_true * flux_true

# --- Extract with both methods at centred and offset positions ------------
methods = {"photutils": {}, "sinc": {}}
for m in methods:
    cog = gcp.extract_growth_curves(
        img, cat, background_variance=bkg_var, show_progress=False, method=m
    )
    methods[m]["centred"] = cog["flux_clean"][0]

    cat_off = Table({"x": [64.3], "y": [64.2], "flux": [50000.0]})
    cog_off = gcp.extract_growth_curves(
        img, cat_off, background_variance=bkg_var, show_progress=False, method=m
    )
    methods[m]["offset"] = cog_off["flux_clean"][0]

# --- Plot -----------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

colors = {"photutils": "C0", "sinc": "C3"}
markers = {"centred": "o", "offset": "s"}

# -- Cumulative --
for m in methods:
    for off in ("centred", "offset"):
        cumul = methods[m][off]
        label = f"{m} ({off})" if off == "offset" else f"{m} ({off})"
        ax1.plot(
            radii, cumul, markers[off] + "-", color=colors[m], label=f"{m} {off}", ms=4
        )
ax1.plot(radii, cumul_true_scaled, "k-", label="Analytic Moffat", lw=1.5)
ax1.set_xlabel("Aperture radius [pix]")
ax1.set_ylabel("Cumulative flux [ADU]")
ax1.set_title("Curve of growth")
ax1.legend(frameon=False, fontsize=8)
ax1.grid(True, alpha=0.3)

# -- Cumulative residual --
ax2.axhline(0, color="gray", ls="--", alpha=0.5)
for m in methods:
    for off in ("centred", "offset"):
        cres = methods[m][off] - cumul_true_scaled
        ax2.plot(
            radii, cres, markers[off] + "-", color=colors[m], label=f"{m} {off}", ms=4
        )
ax2.set_xlabel("Aperture radius [pix]")
ax2.set_ylabel("Extracted − Analytic [ADU]")
ax2.set_title("Cumulative residual")
ax2.legend(frameon=False, fontsize=8)
ax2.grid(True, alpha=0.3)

# -- Annular --
for m in methods:
    for off in ("centred", "offset"):
        ann = gcp.gcmodel.annular_fluxes(methods[m][off])
        ax3.plot(
            radii, ann, markers[off] + "-", color=colors[m], label=f"{m} {off}", ms=4
        )
ax3.plot(radii, annular_true_scaled, "k-", label="Analytic Moffat", lw=1.5)
ax3.set_xlabel("Aperture radius [pix]")
ax3.set_ylabel("Annular flux [ADU]")
ax3.set_title("Annular profile")
ax3.legend(frameon=False, fontsize=8)
ax3.grid(True, alpha=0.3)

# -- Annular residual (fractional) --
ax4.axhline(0, color="gray", ls="--", alpha=0.5)
for m in methods:
    for off in ("centred", "offset"):
        ann = gcp.gcmodel.annular_fluxes(methods[m][off])
        frac = np.where(
            annular_true_scaled > 0,
            (ann - annular_true_scaled) / annular_true_scaled * 100,
            np.nan,
        )
        ax4.plot(
            radii, frac, markers[off] + "-", color=colors[m], label=f"{m} {off}", ms=4
        )
ax4.set_xlabel("Aperture radius [pix]")
ax4.set_ylabel("Fractional residual [%]")
ax4.set_title("Annular fractional residual")
ax4.legend(frameon=False, fontsize=8)
ax4.grid(True, alpha=0.3)

fig.suptitle(
    f"Curve-of-growth extraction vs analytic Moffat (γ={gamma_true}, α={alpha_true})",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("cog_vs_model.png", dpi=150)
print("Saved cog_vs_model.png\n")

# --- Print comparison at inner aperture -----------------------------------
print("Comparison at r = 3.0 pix (centred):")
for m in methods:
    ann = gcp.gcmodel.annular_fluxes(methods[m]["centred"])
    frac = (ann[0] - annular_true_scaled[0]) / annular_true_scaled[0] * 100
    print(f"  {m:12s}: annular residual = {frac:+.4f}%")
