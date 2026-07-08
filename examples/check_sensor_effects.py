"""Show how the growth curve of a single bright star changes with sensor
effects and star colour, using chromatic photon shooting.

Uses ``galsim-phot`` with the LSST ITL sensor model and top-hat r-band.
"""

import numpy as np
from astropy.table import Table

import matplotlib.pyplot as plt

import gcphotom as gcp

# --- Simulation parameters --------------------------------------------------

shape = (129, 129)
flux_true = 50000.0
x_src, y_src = 64.0, 64.0

# Aperture radii for growth-curve extraction (pixels)
radii = np.logspace(np.log10(2), np.log10(30), num=15)

# Star colours to compare (BP-RP)
colours = [-0.3, 0.0, 1.0, 2.0]  # blue → red

# --- Render and extract -----------------------------------------------------

# Results:  dict[colour] -> { "no_sensor": cumul, "with_sensor": cumul }
results = {}

for bp_rp in colours:
    cat = Table({"x": [x_src], "y": [y_src], "flux": [flux_true], "bp_rp": [bp_rp]})

    # Without sensor
    img, _ = gcp.simulate_image_galsim(
        shape,
        cat,
        background=0,
        read_noise=0,
        method="phot",
        max_phot_sources=1,
        chromatic=True,
        bandpass="r",
        sensor=False,
        seed=42,
    )
    cog = gcp.extract_growth_curves(
        img,
        cat,
        radii=radii,
        method="sinc",
        background_variance=np.zeros_like(img),
        show_progress=False,
    )
    cumul_no = cog["flux_clean"][0]

    # With sensor (brighter-fatter + diffusion at LSST nominal)
    img_s, _ = gcp.simulate_image_galsim(
        shape,
        cat,
        background=0,
        read_noise=0,
        method="phot",
        max_phot_sources=1,
        chromatic=True,
        bandpass="r",
        sensor=True,
        bf_strength=1.0,
        diffusion_factor=1.0,
        seed=42,
    )
    cog_s = gcp.extract_growth_curves(
        img_s,
        cat,
        radii=radii,
        method="sinc",
        background_variance=np.zeros_like(img_s),
        show_progress=False,
    )
    cumul_s = cog_s["flux_clean"][0]

    results[bp_rp] = {"no_sensor": cumul_no, "with_sensor": cumul_s}

# --- Plot -------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
(ax_cumul_no, ax_cumul_yes), (ax_diff, ax_detail) = axes

colors_plot = ["C0", "C1", "C2", "C3"]

# Top left: cumulative curves, no sensor
for bp_rp, c in zip(colours, colors_plot):
    ax_cumul_no.plot(
        radii,
        results[bp_rp]["no_sensor"],
        marker="o",
        ms=4,
        color=c,
        label=f"bp-rp = {bp_rp:+.1f}",
    )
ax_cumul_no.set_title("No sensor")
ax_cumul_no.set_ylabel("Cumulative flux [ADU]")
ax_cumul_no.legend(frameon=False, fontsize=8)
ax_cumul_no.grid(True, alpha=0.3)

# Top right: cumulative curves, with sensor
for bp_rp, c in zip(colours, colors_plot):
    ax_cumul_yes.plot(
        radii,
        results[bp_rp]["with_sensor"],
        marker="s",
        ms=4,
        color=c,
        label=f"bp-rp = {bp_rp:+.1f}",
    )
ax_cumul_yes.set_title("With sensor (BF + diffusion)")
ax_cumul_yes.legend(frameon=False, fontsize=8)
ax_cumul_yes.grid(True, alpha=0.3)

# Bottom left: fractional difference (sensor / no_sensor - 1)
ax_diff.axhline(0, color="gray", ls="--", alpha=0.4)
for bp_rp, c in zip(colours, colors_plot):
    no = results[bp_rp]["no_sensor"]
    yes = results[bp_rp]["with_sensor"]
    frac = np.where(no > 0, yes / no - 1.0, np.nan)
    ax_diff.plot(
        radii, frac * 100, marker="^", ms=4, color=c, label=f"bp-rp = {bp_rp:+.1f}"
    )
ax_diff.set_ylabel("Fractional difference [%]")
ax_diff.set_title("Sensor / No sensor − 1")
ax_diff.legend(frameon=False, fontsize=8)
ax_diff.grid(True, alpha=0.3)

# Bottom right: zoom on the inner region for one colour
bp_rp_mid = 0.0
no = results[bp_rp_mid]["no_sensor"]
yes = results[bp_rp_mid]["with_sensor"]
inner = radii < 15
ax_detail.plot(
    radii[inner],
    no[inner],
    "o-",
    color="C0",
    ms=4,
    label=f"No sensor, bp-rp={bp_rp_mid:+.1f}",
)
ax_detail.plot(
    radii[inner],
    yes[inner],
    "s-",
    color="C3",
    ms=4,
    label=f"Sensor, bp-rp={bp_rp_mid:+.1f}",
)
ax_detail.set_title(f"Zoom (bp-rp = {bp_rp_mid:+.1f})")
ax_detail.legend(frameon=False, fontsize=8)
ax_detail.grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel("Aperture radius [pix]")

fig.suptitle(
    f"Chromatic PSF — single star, flux={flux_true} ADU, r-band, LSST ITL sensor",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("check_sensor_effects.png", dpi=150)
print("Saved check_sensor_effects.png")

plt.show()
