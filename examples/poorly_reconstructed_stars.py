import gcphotom as gcp, numpy as np
import matplotlib.pyplot as plt

# 1. Simulate a realistic astronomical image
image, sim_cat = gcp.simulate_image(n_sources=1000, background=100, read_noise=5)

# 2. Detect sources and build segmentation image (now takes care of background estimation)
seg, det_cat = gcp.detect_and_segment(image)

# 3. Extract growth curves with contamination estimation (takes care of error estimation)
cog = gcp.extract_growth_curves(image, det_cat, segmentation_image=seg)

# 4. Fit all growth curves
fitter = gcp.Fitter(cog)
best_fit, extra = fitter.fit()
fitter.detect_contamination(best_fit)
best_fit, extra = fitter.fit()
fitted = fitter.results(best_fit)

# 5. Match (return the matched reordered version of the sim_cat)
input_cat = gcp.cross_match(det_cat, sim_cat)

# 6. Inspect results
print(f"PSF: gamma={fitted['gamma']:.2f}, alpha={fitted['alpha']:.2f}")

poorly = (
    np.abs(fitted["flux"] / input_cat["flux"] - 1)
    > 3 * fitted["std_errors"]["flux"] / input_cat["flux"]
)
bad = (fitted["ngoods"] < 5) & (fitted["chi2"] > 10 * fitted["ngoods"])
unrecognized = poorly & ~bad

plt.figure("Flux reconstruction")
for index in ~poorly, poorly:
    plt.errorbar(
        input_cat["flux"][index],
        ((fitted["flux"] / input_cat["flux"])[index] - 1) * 100,
        (fitted["std_errors"]["flux"] / input_cat["flux"])[index] * 100,
        marker="o",
        ls="None",
    )
plt.xlabel("Simulated flux [ADU]")
plt.ylabel("Reconstruction error [%]")
plt.xscale("log")
plt.axhline(0, color="k")

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
    a, b = max(0, center[0] - width[0]), min(image.shape[0], center[0] + width[0])
    c, d = max(0, center[1] - width[1]), min(image.shape[1], center[1] + width[1])
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

    # plt.plot(input_cat['x'][unrecognized], input_cat['y'][unrecognized], 'r+')
