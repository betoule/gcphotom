import re
from pathlib import Path

import numpy as np

from gcphotom.gcmodel import Fitter


def _add_fields(arr, **fields):
    dt = arr.dtype
    new_dt = np.dtype(dt.descr + [(name, float) for name in fields])
    new_arr = np.empty(arr.shape, dtype=new_dt)
    for name in dt.names:
        new_arr[name] = arr[name]
    for name, values in fields.items():
        new_arr[name] = np.asarray(values, dtype=float)
    return new_arr


def _parse_radius_columns(cat):
    radii = set()
    for name in cat.dtype.names:
        m = re.match(r"^apfl_(\d+\.\d+)$", name)
        if m:
            radii.add(float(m.group(1)))
    return sorted(radii)


def _column_name(prefix, radius):
    return f"{prefix}_{radius:.2f}"


def _read_columns(cat, prefix, radii):
    """Read columns with a given prefix (e.g. 'apfl', 'apother') into a 2D array."""
    n_src = len(cat)
    n_rad = len(radii)
    arr = np.zeros((n_src, n_rad))
    for i, r in enumerate(radii):
        arr[:, i] = cat[_column_name(prefix, r)]
    return arr


def parse_snls_catalog(path):
    """Load an SNLS forced photometry catalog.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npy`` forced photometry catalog.

    Returns
    -------
    gc_result : dict
        Dictionary with keys ``radius``, ``flux``, ``flux_clean``,
        ``background_var``, ``contamination``, compatible with
        :class:`gcphotom.gcmodel.Fitter`.
        ``flux`` and ``flux_clean`` both contain the raw ``apfl`` values
        (contamination is not subtracted — it is provided separately for
        masking via :func:`build_contamination_mask`).
    meta : dict
        Additional data needed for processing: the original catalog array
        (``'cat'``), the cumulative contamination (``'contamination'``),
        and the cumulative bad-pixel flags (``'bad'``).
    """
    cat = np.load(path)
    radii = _parse_radius_columns(cat)

    flux = _read_columns(cat, "apfl", radii)
    bkg_var = _read_columns(cat, "apvar", radii)
    contamination = _read_columns(cat, "apother", radii)

    gc_result = {
        "radius": np.array(radii),
        "flux": flux.copy(),
        "flux_clean": flux.copy(),
        "background_var": bkg_var,
        "contamination": np.zeros_like(contamination),
    }
    # apbad might not exist in all catalogs
    apbad_prefix = "apbad"
    if any(name.startswith(apbad_prefix) for name in cat.dtype.names):
        bad = _read_columns(cat, apbad_prefix, radii)
    else:
        bad = np.zeros_like(contamination)

    meta = {
        "cat": cat,
        "contamination": contamination,
        "bad": bad,
    }
    return gc_result, meta


def build_contamination_mask(meta, cum_contamination=None, cum_bad=None):
    """Build a boolean mask for annular bins that are free of contamination and bad pixels.

    Parameters
    ----------
    meta : dict
        The second output of :func:`parse_snls_catalog`, containing
        the cumulative ``contamination`` and ``bad`` arrays.
    cum_contamination : ndarray or None
        Override for the cumulative contamination array. If None, use
        ``meta['contamination']``.
    cum_bad : ndarray or None
        Override for the cumulative bad-pixel array. If None, use
        ``meta['bad']``.

    Returns
    -------
    mask : ndarray of bool
        Boolean array of shape ``(n_radii, n_sources)`` where ``True``
        indicates a clean annular bin.
    """
    if cum_contamination is None:
        cum_contamination = meta["contamination"]
    if cum_bad is None:
        cum_bad = meta["bad"]

    # Convert cumulative contamination/bad to annular via diff with prepend
    ann_cont = np.diff(cum_contamination, prepend=0, axis=1)
    ann_bad = np.diff(cum_bad, prepend=0, axis=1)

    mask = (ann_cont == 0) & (ann_bad == 0)
    return mask.T  # transpose to (n_radii, n_sources) to match Fitter convention


def filter_to_reference(
    cat,
    ref_cat,
    band="g",
    min_flux=10000.0,
    exclude_windowed=True,
    exclude_saturated=True,
):
    """Build a boolean mask selecting forced-catalog entries that match reference stars.

    Parameters
    ----------
    cat : ndarray
        Forced photometry catalog with ``bindex``, ``windowed``, ``saturated`` fields.
    ref_cat : ndarray
        Reference catalog with ``index``, ``star``, ``star_<band>``,
        ``flux_<band>`` fields.
    band : str
        Band to select from the reference catalog (e.g. ``'g'``, ``'r'``, ``'i'``).
    min_flux : float
        Minimum flux in the reference band to select a source.
    exclude_windowed : bool
        If True, exclude windowed objects.
    exclude_saturated : bool
        If True, exclude saturated objects.

    Returns
    -------
    mask : ndarray of bool
        Boolean array of length ``len(cat)``, True for entries passing all filters.
    """
    star_lookup = np.zeros(ref_cat["index"].max() + 1, dtype=bool)

    star_band_col = f"star_{band}"
    if star_band_col in ref_cat.dtype.names:
        star_flag = ref_cat[star_band_col]
    else:
        star_flag = ref_cat["star"]

    flux_band_col = f"flux_{band}"
    star_lookup[ref_cat["index"]] = star_flag & (ref_cat[flux_band_col] > min_flux)

    mask = star_lookup[cat["bindex"]]

    if exclude_windowed:
        mask &= ~cat["windowed"]
    if exclude_saturated:
        mask &= ~cat["saturated"]

    return mask


def process_single(
    cat_path,
    ref_cat,
    band="g",
    min_flux=10000.0,
    exclude_windowed=True,
    exclude_saturated=True,
    mask_contamination=True,
    learning_rate=5e-3,
    niter=10000,
):
    """Load, filter and fit a single forced photometry catalog.

    Parameters
    ----------
    cat_path : str or Path
        Path to the forced photometry catalog.
    ref_cat : ndarray
        Reference catalog.
    band : str
        Band selection.
    min_flux : float
        Minimum reference flux.
    exclude_windowed : bool
        Exclude windowed sources.
    exclude_saturated : bool
        Exclude saturated sources.
    mask_contamination : bool
        If True, flag contaminated and bad-pixel annular bins as not-good
        in the fitter (matching the moffatphot.py approach).
    learning_rate : float
        Adam learning rate for the fitter.
    niter : int
        Number of optimizer iterations.

    Returns
    -------
    result : dict or None
        Dictionary with keys ``path``, ``cat``, ``bf``, ``extra``, ``fitter``,
        ``n_initial``, ``n_selected``. Returns ``None`` if no sources remain
        after filtering.
    """
    gc_result, meta = parse_snls_catalog(cat_path)
    cat = meta["cat"]

    mask = filter_to_reference(
        cat,
        ref_cat,
        band=band,
        min_flux=min_flux,
        exclude_windowed=exclude_windowed,
        exclude_saturated=exclude_saturated,
    )

    n_initial = len(cat)
    n_selected = int(mask.sum())

    if n_selected == 0:
        return None

    cat_sel = cat[mask]
    sliced = {
        "radius": gc_result["radius"],
        "flux": gc_result["flux"][mask],
        "flux_clean": gc_result["flux_clean"][mask],
        "background_var": gc_result["background_var"][mask],
        "contamination": gc_result["contamination"][mask],
    }

    fitter = Fitter(sliced)

    if mask_contamination:
        meta_sel = {
            "contamination": meta["contamination"][mask],
            "bad": meta["bad"][mask],
        }
        goods_mask = build_contamination_mask(meta_sel)
        fitter.goods = fitter.goods & goods_mask

    bf, extra = fitter.fit(learning_rate=learning_rate, niter=niter)
    fitter.detect_contamination(bf)
    if fitter.fluxes.shape[1] > 0:
        bf, extra = fitter.fit(learning_rate=learning_rate, niter=niter)

    return {
        "path": str(cat_path),
        "cat": cat_sel,
        "bf": bf,
        "extra": extra,
        "fitter": fitter,
        "n_initial": n_initial,
        "n_selected": n_selected,
    }


def write_catalog(result, output_path):
    """Append fitted columns to the filtered catalog and save as ``.npy``.

    Parameters
    ----------
    result : dict
        Output from :func:`process_single`.
    output_path : str or Path
        Path to save the output ``.npy`` file.
    """
    cat = result["cat"]
    fitter = result["fitter"]
    bf = result["bf"]

    res = fitter.results(bf)

    out = _add_fields(
        cat,
        mflux=res["flux"],
        mback=res["back"],
        mgoods=res["ngoods"],
        mchi2=res["chi2"],
    )
    np.save(str(output_path), out)


def default_output_path(cat_path):
    """Derive output path by replacing ``'forced'`` with ``'mophot'`` in the filename."""
    p = Path(cat_path)
    return p.parent / p.name.replace("forced", "mophot")
