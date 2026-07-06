import gcphotom as gcp, numpy as np
from gcphotom.plots import binplot
import matplotlib
import importlib.util as _util

if _util.find_spec("tkinter") is not None:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder

# 1. Simulate a realistic astronomical image
image, sim_cat = gcp.simulate_image(n_sources=1000, background=100, read_noise=5)

# 2. Detect sources and build segmentation image (now takes care of background estimation)
seg, det_cat, bkg_map, bkg_var_map = gcp.detect_and_segment(image, n_pixels=5)
# let's kill unrecognized blends
bads = (det_cat.ellipticity * det_cat.area).value > 6

# 3. Extract growth curves with contamination estimation
cog = gcp.extract_growth_curves(
    image, det_cat, segmentation_image=seg, background_variance=bkg_var_map
)

# 4. Fit all growth curves
fitter = gcp.Fitter(cog, bads=bads)
best_fit, extra = fitter.fit(learning_rate=1e-2, niter=2000)
fitter.detect_contamination(best_fit)
best_fit, extra = fitter.fit(learning_rate=1e-2, niter=2000, compute_uncertainty=True)
best_fit_no_back, extra_no_back = fitter.fit(
    learning_rate=1e-2,
    niter=2000,
    fix={"back": np.full(len(best_fit["back"]), np.mean(best_fit["back"]))},
)
# best_fit2, extra2 = fitter.fit()
fitted = fitter.results(best_fit)
fitted_no_back = fitter.results(best_fit_no_back)

# 5. Match (return the matched reordered version of the sim_cat)
input_cat = gcp.cross_match(det_cat, sim_cat)

# 6. Inspect results
print(f"PSF: gamma={fitted['gamma']:.2f}, alpha={fitted['alpha']:.2f}")

# 7. PSF photometry baseline (empirical PSF)
psf_results, epsf_res = gcp.psf_photometry(
    image - 100, det_cat, nstars=30, fit_shape=25
)
print(f"ePSF: {epsf_res.iterations} iterations, converged={epsf_res.converged}")
psf_cat = gcp.cross_match(psf_results, sim_cat)

# 8. Aperture photometry with constant correction
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
flux_ap_full = np.full(len(det_cat), np.nan)
flux_ap_full[fitter.kept] = flux_ap
kept_cat = input_cat[fitter.kept]
print(
    f"Aperture: r_core={r_core:.1f}px, r_corr={r_corr:.1f}px, "
    f"FWHM={fwhm:.2f}px, AC={ac:.4f}"
)


poorly = (
    np.abs(fitted["flux"] / input_cat["flux"] - 1)
    > 4 * fitted["std_errors"]["flux"] / input_cat["flux"]
)
bad = (fitted["ngoods"] < 5) & (fitted["chi2"] > 10 * fitted["ngoods"])
unrecognized = poorly & ~bad

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
plt.savefig("reconstruction_quality.png")
plt.show()

# plt.figure('Simulated image')
# plt.imshow(image, norm='symlog')
# # Let us plot stars with good fit yet poorly reconstructed flux
# plt.plot(sim_cat['x'], sim_cat['y'], 'w+')
# plt.plot(input_cat['x'][unrecognized], input_cat['y'][unrecognized], 'r+')
# plt.colorbar()


def crop(image, center=None, width=(100, 100)):
    """Quick function to crop the surrounding of a source in an image

    Parameters:
    -----------
    image: jnp.ndarray
           The original image
    center: tuple of int
            The center of the crop region
    width: tuple of int
           half the width of the crop region
    Returns:
    --------
    jnp.ndarray
           Cropped image
    """
    if center is None:
        center = image.shape[0] // 2, image.shape[1] // 2
    else:
        center = (int(center[0]), int(center[1]))
    a, b = max(0, center[1] - width[0]), min(image.shape[0], center[1] + width[0])
    c, d = max(0, center[0] - width[1]), min(image.shape[1], center[0] + width[1])
    return image[a:b, c:d], a, b, c, d


def show(s):
    fig = plt.figure()
    ax1, ax2 = fig.subplots(1, 2, sharex=True, sharey=True)
    cr, a, b, c, d = crop(image, center=(s["x"], s["y"]))
    ax1.imshow(cr, norm="symlog")
    ax2.imshow(crop(seg.data, center=(s["x"], s["y"]))[0])
    ax1.plot(sim_cat["x"] - c, sim_cat["y"] - a, "w+")
    ax1.plot(s["x"] - c, s["y"] - a, "rx")
    ax1.set_ylim(0, b - a)
    ax2.set_xlim(0, d - c)
    ax1.set_title(f'{s["x"]:.1f}, {s["y"]:.1f}: {s["flux"]:.2f}')

    # plt.plot(input_cat['x'][unrecognized], input_cat['y'][unrecognized], 'r+')
