"""Compare robust loss functions for growth curve fitting.

Demonstrates how different M-estimator loss functions affect flux
reconstruction in a realistic crowded-field scenario with natural
aperture contamination from undetected sources and PSF tails.
"""

import time

import matplotlib
import importlib.util as _util

if _util.find_spec("tkinter") is not None:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gcphotom as gcp

SEED = 42
NITER = 1000

print("--- Robust loss comparison ---")
print(f"Seed: {SEED}, niter: {NITER}")

# 1. Simulate
print("\n1. Simulating image (1000 sources)...", end=" ")
t0 = time.time()
image, sim_cat = gcp.simulate_image(
    n_sources=1000, background=100, read_noise=5, seed=SEED
)
print(f"done in {time.time() - t0:.1f}s")

# 2. Detect
print("2. Detecting sources...", end=" ")
t0 = time.time()
seg, det_cat, _, bkg_var_map = gcp.detect_and_segment(image)
n_det = len(det_cat)
print(f"found {n_det} sources in {time.time() - t0:.1f}s")

# 3. Extract growth curves
print("3. Extracting growth curves...", end=" ")
t0 = time.time()
cog = gcp.extract_growth_curves(
    image, det_cat, segmentation_image=seg, background_variance=bkg_var_map
)
print(f"done in {time.time() - t0:.1f}s")

# 4. Fit with different loss functions
fitter = gcp.Fitter(cog)
initial_guess = fitter.initial_guess()

# Using OrderedDict to preserve iteration order across Python versions
losses = {
    "chi2": lambda x: x**2,
    "Tukey c=4.685": gcp.tukey(),
    "Pseudo-Huber c=2": gcp.pseudo_huber(c=2.0),
    "Cauchy c=2": gcp.cauchy(c=2.0),
}

results = {}
for i, (name, loss_fn) in enumerate(losses.items()):
    print(f"4.{i+1}. Fitting with {name}...", end=" ", flush=True)
    t0 = time.time()
    bf, extra = fitter.fit(initial_guess=initial_guess, loss=loss_fn, niter=NITER)
    elapsed = time.time() - t0
    res = fitter.results(bf)
    results[name] = {"bf": bf, "res": res, "extra": extra}

    n_good = int(np.isfinite(res["flux"]).sum())
    gamma = float(bf["gamma"])
    loss_end = float(extra["loss"][-1])
    n_iter = len(extra["loss"])
    print(
        f"{elapsed:.0f}s, {n_iter} iters, gamma={gamma:.3f}, "
        f"final loss={loss_end:.4f}, {n_good} sources"
    )

# 5. Cross-match detected to simulated catalog
print("5. Cross-matching...", end=" ", flush=True)
t0 = time.time()
input_cat = gcp.cross_match(det_cat, sim_cat)
print(f"done in {time.time() - t0:.1f}s")

matched = np.isfinite(input_cat["flux"])
true_flux = np.asarray(input_cat["flux"][matched], dtype=float)
n_matched = matched.sum()
print(f"   {n_matched} / {n_det} sources matched to truth")

# 6. Plot
print("6. Plotting...", end=" ", flush=True)

colors = {
    "chi2": "#555555",
    "Tukey c=4.685": "#2166ac",
    "Pseudo-Huber c=2": "#d6604d",
    "Cauchy c=2": "#b2182b",
}

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1.2]}
)

# -- Panel 1: Binned flux ratio vs true flux --
bins = np.logspace(
    np.log10(np.nanmin(true_flux) * 0.9),
    np.log10(np.nanmax(true_flux) * 1.1),
    15,
)

for name in losses:
    ratio = np.asarray(results[name]["res"]["flux"][matched], dtype=float) / true_flux

    bin_centers = []
    bin_medians = []
    bin_low = []
    bin_high = []
    for i in range(len(bins) - 1):
        mask = (true_flux >= bins[i]) & (true_flux < bins[i + 1])
        n_bin = mask.sum()
        if n_bin >= 10:
            bc = np.sqrt(bins[i] * bins[i + 1])
            bin_centers.append(bc)
            med = np.nanmedian(ratio[mask])
            bin_medians.append(med)
            scatter = 1.4826 * np.nanmedian(np.abs(ratio[mask] - med))
            bin_low.append(med - scatter)
            bin_high.append(med + scatter)

    ax1.plot(
        bin_centers,
        bin_medians,
        "-",
        color=colors[name],
        label=name,
        linewidth=2,
    )
    ax1.fill_between(bin_centers, bin_low, bin_high, color=colors[name], alpha=0.12)

ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
ax1.set_xscale("log")
ax1.set_xlabel("True flux")
ax1.set_ylabel("Fitted / True")
ax1.set_title(
    f"Flux reconstruction — {n_matched} matched sources, "
    f"$\\gamma_{{\\rm true}}$=3, $\\alpha_{{\\rm true}}$=3"
)
ax1.legend(fontsize=9, loc="upper left", ncol=2)

# -- Panel 2: PSF parameter biases (gamma and alpha) --
true_val = {"gamma": 3.0, "alpha": 3.0}
x_pos = np.arange(len(losses))
width = 0.30

for j, par in enumerate(["gamma", "alpha"]):
    offset = (j - 0.5) * width
    vals = [results[n]["bf"][par] - true_val[par] for n in losses]
    bars = ax2.bar(
        x_pos + offset,
        vals,
        width,
        color=[colors[n] for n in losses],
        edgecolor="white",
        alpha=0.6 + 0.4 * (1 - j),
        label=f"${par}$" if j == 0 else None,
    )
    for i, (name, v) in enumerate(zip(losses, vals)):
        y_pos = v + 0.005 * (1 if v >= 0 else -1)
        ax2.text(
            i + offset,
            y_pos,
            f"{v:+.3f}",
            fontsize=6,
            ha="center",
            va="bottom" if v >= 0 else "top",
        )

ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(list(losses.keys()), fontsize=8, rotation=20, ha="right")
ax2.set_ylabel("Estimated $-$ True")
ax2.set_title("PSF parameter bias")

plt.tight_layout()
plt.savefig("robust_loss_comparison.png", dpi=150)
print("saved robust_loss_comparison.png")
plt.show()

# 7. Print summary table
print("\n--- Summary ---")
print(
    f"{'Loss':<20} {'Gamma':>7} {'Alpha':>7} {'Median bias':>12} "
    f"{'MAD scatter':>12}"
)
print("-" * 60)
for name in losses:
    ratio = np.asarray(results[name]["res"]["flux"][matched], dtype=float) / true_flux
    med_bias = np.nanmedian(ratio - 1.0)
    mad_scatter = 1.4826 * np.nanmedian(np.abs(ratio - 1.0 - med_bias))
    g = results[name]["bf"]["gamma"]
    a = results[name]["bf"]["alpha"]
    print(f"{name:<20} {g:7.3f} {a:7.3f} {med_bias:+12.4f} {mad_scatter:12.4f}")

print("\nDone.")
