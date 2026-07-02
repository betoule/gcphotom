import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

import gcphotom as gcp
from gcphotom.plots import binplot

# ------------------------------------------------------------------
# 1. Load real background image
# ------------------------------------------------------------------


def snls_detrend(fname, ext=1):
    """Load and detrend an SNLS FITS file."""
    data, header = pyfits.getdata(fname, ext, header=True)
    data = data.astype(float)

    overscanA = slice(0, 32), slice(0, 4644)
    overscanB = slice(2080, 2112), slice(0, 4644)
    illuregionT = slice(2, 4610), slice(32, 2080)

    ovA = np.median(data.T[overscanA], 0)
    ovB = np.median(data.T[overscanB], 0)

    data.T[:1056, :] -= ovA
    data.T[1056:, :] -= ovB
    data[(data <= 3000) | (data > 60000)] = np.nan

    return data[illuregionT]


real_image = snls_detrend("1013453o.fz", 13)

# Mask NaN pixels (replaced by median of valid pixels)
nan_mask = np.isnan(real_image)
if np.any(nan_mask):
    median_val = np.nanmedian(real_image)
    real_image[nan_mask] = median_val


# ------------------------------------------------------------------
# 2. Generate and inject source catalog
# ------------------------------------------------------------------

n_sources = 1000
gamma = 3
alpha = 3
seed = 42

sim_cat = gcp.make_realistic_source_catalog(
    n_sources=n_sources, shape=real_image.shape, seed=seed
)

# Simulate sources only — no background, no read noise.
# Poisson noise is applied to (source_model + 0), i.e. source-only.
source_image, sim_cat = gcp.simulate_image(
    shape=real_image.shape,
    catalog=sim_cat,
    gamma=gamma,
    alpha=alpha,
    background=0,
    read_noise=0,
    seed=seed,
)

# Composite: real background + injected sources
image = real_image + source_image


# ------------------------------------------------------------------
# 3. Detect sources
# ------------------------------------------------------------------

seg, det_cat = gcp.detect_and_segment(image, n_pixels=5)
bads = (det_cat.ellipticity * det_cat.area).value > 6


# ------------------------------------------------------------------
# 4. Growth curve extraction (on full detection catalog)
# ------------------------------------------------------------------
# Must use the full det_cat so that segmentation labels align with
# source indices. Filtering to injected sources happens afterwards.

cog = gcp.extract_growth_curves(image, det_cat, segmentation_image=seg)


# ------------------------------------------------------------------
# 5. Match detection catalog to injected catalog, keep intersection
# ------------------------------------------------------------------

# cross_match aligns sim_cat columns to det_cat ordering.
# Rows in matched that have NaN values = detected sources with no
# injected counterpart (pre-existing sources in the real image).
matched = gcp.cross_match(det_cat, sim_cat)

injected_mask = ~np.isnan(matched["flux"])
input_cat = matched[injected_mask]

print(f"Detected sources  : {len(det_cat)}")
print(f"Injected sources  : {len(sim_cat)}")
print(f"Matched (injected): {injected_mask.sum()}")

# Filter growth curves and bads to injected sources only
# radius is 1D (shared), all other arrays are 2D (n_sources, ...)
cog = {k: v[injected_mask] if v.ndim == 2 else v for k, v in cog.items()}
bads_injected = bads[injected_mask]

fitter = gcp.Fitter(cog, bads=bads_injected)
best_fit, extra = fitter.fit(learning_rate=1e-2, niter=2000)
fitter.detect_contamination(best_fit)
best_fit, extra = fitter.fit(learning_rate=1e-2, niter=2000)
best_fit_no_back, extra_no_back = fitter.fit(
    learning_rate=1e-2,
    niter=2000,
    fix={"back": np.full(len(best_fit["back"]), np.mean(best_fit["back"]))},
)
fitted = fitter.results(best_fit)
fitted_no_back = fitter.results(best_fit_no_back)


# ------------------------------------------------------------------
# 7. PSF photometry baseline
# ------------------------------------------------------------------

psf_results, epsf_res = gcp.psf_photometry(
    image - np.median(real_image), det_cat[injected_mask], nstars=30, fit_shape=11
)
print(f"ePSF: {epsf_res.iterations} iterations, converged={epsf_res.converged}")
psf_cat = gcp.cross_match(psf_results, sim_cat)

# ------------------------------------------------------------------
# 8. Aperture photometry with constant correction
# ------------------------------------------------------------------

fwhm = gcp.gcmodel.gamma2fwhm(fitted["gamma"], fitted["alpha"])
r_core_idx = np.argmin(np.abs(cog["radius"] - fwhm))
r_corr_idx = np.argmin(np.abs(cog["radius"] - 3 * fwhm))
r_core = cog["radius"][r_core_idx]
r_corr = cog["radius"][r_corr_idx]
ac = gcp.gcmodel.moffat_flux(
    r_corr, fitted["gamma"], fitted["alpha"]
) / gcp.gcmodel.moffat_flux(r_core, fitted["gamma"], fitted["alpha"])
bkg = fitted["back"][fitter.kept]
flux_ap = (cog["flux_clean"][:, r_core_idx][fitter.kept] - bkg * np.pi * r_core**2) * ac
kept_cat = input_cat[fitter.kept]
print(
    f"Aperture: r_core={r_core:.1f}px, r_corr={r_corr:.1f}px, "
    f"FWHM={fwhm:.2f}px, AC={ac:.4f}"
)


# ------------------------------------------------------------------
# 9. Diagnostics
# ------------------------------------------------------------------

print(f"PSF: gamma={fitted['gamma']:.2f}, alpha={fitted['alpha']:.2f}")

poorly = (
    np.abs(fitted["flux"] / input_cat["flux"] - 1)
    > 4 * fitted["std_errors"]["flux"] / input_cat["flux"]
)
bad = (fitted["ngoods"] < 5) & (fitted["chi2"] > 10 * fitted["ngoods"])
unrecognized = poorly & ~bad


# Flux reconstruction plot
plt.figure("Flux reconstruction")
for index in ~poorly, poorly:
    plt.errorbar(
        input_cat["flux"][index],
        ((fitted["flux"] / input_cat["flux"])[index] - 1) * 100,
        (fitted["std_errors"]["flux"] / input_cat["flux"])[index] * 100,
        marker=".",
        alpha=0.2,
        ls="None",
    )
binplot(
    input_cat["flux"],
    ((fitted["flux"] / input_cat["flux"]) - 1) * 100,
    data=False,
    method="median",
    color="k",
    zorder=10,
    logbins=True,
    scale_err=True,
    label="estimated background",
)
binplot(
    input_cat["flux"],
    ((fitted_no_back["flux"] / input_cat["flux"]) - 1) * 100,
    data=False,
    method="median",
    color="r",
    zorder=10,
    logbins=True,
    scale_err=True,
    label="no back",
)
binplot(
    psf_cat["flux"],
    ((psf_results["flux_fit"] / psf_cat["flux"]) - 1) * 100,
    data=False,
    method="median",
    color="b",
    zorder=10,
    logbins=True,
    scale_err=True,
    label="PSF photometry",
)
binplot(
    kept_cat["flux"],
    ((flux_ap / kept_cat["flux"]) - 1) * 100,
    data=False,
    method="median",
    color="m",
    zorder=10,
    logbins=True,
    scale_err=True,
    label="aperture + AC",
)
plt.xlabel("Simulated flux [ADU]")
plt.ylabel("Reconstruction error [%]")
plt.xscale("log")
plt.ylim(-2, 2)
plt.axhline(0, color="k")
plt.legend(loc="best", frameon=False)
plt.savefig("semi_realistic_reconstruction.png")
print("Saved semi_realistic_reconstruction.png")


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------


def crop(image, center=None, width=(100, 100)):
    """Crop a region around a source."""
    if center is None:
        center = image.shape[0] // 2, image.shape[1] // 2
    else:
        center = (int(center[0]), int(center[1]))
    a, b = max(0, center[1] - width[0]), min(image.shape[0], center[1] + width[0])
    c, d = max(0, center[0] - width[1]), min(image.shape[1], center[0] + width[1])
    return image[a:b, c:d], a, b, c, d


def show(s):
    """Show a crop of the image and segmentation around source s."""
    fig = plt.figure()
    ax1, ax2 = fig.subplots(1, 2, sharex=True, sharey=True)
    cr, a, b, c, d = crop(image, center=(s["x"], s["y"]))
    ax1.imshow(cr, norm="symlog")
    ax2.imshow(crop(seg.data, center=(s["x"], s["y"]))[0])
    ax1.plot(input_cat["x"] - c, input_cat["y"] - a, "w+")
    ax1.plot(s["x"] - c, s["y"] - a, "rx")
    ax1.set_ylim(0, b - a)
    ax2.set_xlim(0, d - c)
    ax1.set_title(f'{s["x"]:.1f}, {s["y"]:.1f}: {s["flux"]:.2f}')
