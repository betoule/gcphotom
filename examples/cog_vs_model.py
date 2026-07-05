"""Compare extracted curve of growth to the analytic Moffat profile.

A single noiseless source is placed at the centre of a small image and its
curve of growth is extracted via :func:`gcphotom.extract_growth_curves`.
The result is compared to the analytic Moffat cumulative and annular flux,
revealing pixelisation biases in the aperture-photometry extraction.
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

# --- Extract at the exact centre -----------------------------------------
# Noiseless image → pass zero background variance to skip estimation.
bkg_var = np.zeros_like(img, dtype=float)
cog = gcp.extract_growth_curves(
    img,
    cat,
    background_variance=bkg_var,
    show_progress=False,
)
radii = cog["radius"]
cumul_data = cog["flux_clean"][0]  # (n_radii,)
annular_data = gcp.gcmodel.annular_fluxes(cumul_data)

# --- Analytic expectation -------------------------------------------------
cumul_true, annular_true = analytic_profile(radii, gamma_true, alpha_true)

# Scale by the input flux for absolute comparison
flux_true = float(cat["flux"][0])
cumul_true_scaled = cumul_true * flux_true
annular_true_scaled = annular_true * flux_true

# --- Also extract at a sub-pixel offset to check centering sensitivity ---
cat_off = Table({"x": [64.3], "y": [64.2], "flux": [50000.0]})
cog_off = gcp.extract_growth_curves(
    img, cat_off, background_variance=bkg_var, show_progress=False
)
cumul_off = cog_off["flux_clean"][0]
annular_off = gcp.gcmodel.annular_fluxes(cumul_off)

# --- Plot -----------------------------------------------------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

# -- Cumulative --
ax1.plot(radii, cumul_data, "o-", label="Extracted (centred)", ms=4)
ax1.plot(radii, cumul_off, "s-", label="Extracted (offset 0.3 px)", ms=4)
ax1.plot(radii, cumul_true_scaled, "k-", label="Analytic Moffat", lw=1.5)
ax1.set_xlabel("Aperture radius [pix]")
ax1.set_ylabel("Cumulative flux [ADU]")
ax1.set_title("Curve of growth")
ax1.legend(frameon=False)
ax1.grid(True, alpha=0.3)

# -- Cumulative residual --
resid_cumul = cumul_data - cumul_true_scaled
resid_cumul_off = cumul_off - cumul_true_scaled
ax2.axhline(0, color="gray", ls="--", alpha=0.5)
ax2.plot(radii, resid_cumul, "o-", label="Centred", ms=4)
ax2.plot(radii, resid_cumul_off, "s-", label="Offset 0.3 px", ms=4)
ax2.set_xlabel("Aperture radius [pix]")
ax2.set_ylabel("Extracted − Analytic [ADU]")
ax2.set_title("Cumulative residual")
ax2.legend(frameon=False)
ax2.grid(True, alpha=0.3)

# -- Annular --
ax3.plot(radii, annular_data, "o-", label="Extracted (centred)", ms=4)
ax3.plot(radii, annular_off, "s-", label="Extracted (offset 0.3 px)", ms=4)
ax3.plot(radii, annular_true_scaled, "k-", label="Analytic Moffat", lw=1.5)
ax3.set_xlabel("Aperture radius [pix]")
ax3.set_ylabel("Annular flux [ADU]")
ax3.set_title("Annular profile")
ax3.legend(frameon=False)
ax3.grid(True, alpha=0.3)

# -- Annular residual (fractional) --
frac_resid = np.where(
    annular_true_scaled > 0,
    (annular_data - annular_true_scaled) / annular_true_scaled * 100,
    np.nan,
)
frac_resid_off = np.where(
    annular_true_scaled > 0,
    (annular_off - annular_true_scaled) / annular_true_scaled * 100,
    np.nan,
)
ax4.axhline(0, color="gray", ls="--", alpha=0.5)
ax4.plot(radii, frac_resid, "o-", label="Centred", ms=4)
ax4.plot(radii, frac_resid_off, "s-", label="Offset 0.3 px", ms=4)
ax4.set_xlabel("Aperture radius [pix]")
ax4.set_ylabel("Fractional residual [%]")
ax4.set_title("Annular fractional residual")
ax4.legend(frameon=False)
ax4.grid(True, alpha=0.3)

fig.suptitle(
    f"Curve-of-growth extraction vs analytic Moffat (γ={gamma_true}, α={alpha_true})",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("cog_vs_model.png", dpi=150)
print("Saved cog_vs_model.png\n")

# --- Print residuals ------------------------------------------------------
print("  r[pix]  cumul_resid  annular_resid(%)  annular_resid_off(%)")
for i, r in enumerate(radii):
    print(
        f"  {r:5.1f}  {resid_cumul[i]:+10.2f}  "
        f"{frac_resid[i]:+12.4f}  {frac_resid_off[i]:+18.4f}"
    )
