# Aperture Contamination Estimation — Detailed Plan

## Goal

Estimate the contaminating flux in each aperture from neighboring sources, using a segmentation mask derived from source detection.

## Pipeline Overview

```
image (with background)
  │
  ▼
[1] detect_and_segment(image, background)
  │  → segmentation_image (SegmentationImage)
  │  → positions (detected centroids)
  │  → labels (segment label per source)
  │
  ▼
[2] extract_growth_curves(image, positions, segmentation_image=..., labels=...)
  │  For each source i:
  │    a) _extract_single_growth_curve(image, pos_i, radii)         → flux_total[i, :]
  │    b) _extract_single_growth_curve(image, pos_i, radii, mask)   → flux_clean[i, :]
  │       where mask = (seg.data != label_i)
  │    c) contamination[i, :] = flux_total - flux_clean
  │
  ▼
  result dict: {radius, flux, flux_err, flux_clean, contamination}
```

## Step 1: Detection & Segmentation

### New function: `detect_and_segment(image, background, n_sigma=3.0, n_pixels=10, deblend=True)`

**Purpose:** Produce a segmentation mask and a detection catalog from a raw image.

**Parameters:**
- `image`: 2D image data with background included.
- `background`: float — the background level to subtract. This is provided by the caller (e.g., estimated via sigma-clipping or from the fitter). The function does not estimate it internally.
- `n_sigma`: detection significance threshold (passed to `detect_threshold`).
- `n_pixels`: minimum connected pixels for a valid source.
- `deblend`: if True, run `deblend_sources` to separate overlapping sources.

**Algorithm:**
1. Subtract background: `subtracted = image - background`
2. Estimate detection threshold:
   `threshold = detect_threshold(image, n_sigma=n_sigma, background=background)`
3. Detect sources:
   `seg = detect_sources(subtracted, threshold, n_pixels=n_pixels)`
4. Deblend overlapping sources (if `deblend=True`):
   `seg = deblend_sources(subtracted, seg, n_levels=32, contrast=0.001, progress_bar=False)`
5. Build detection catalog from `SourceCatalog(subtracted, seg)`:
   - Extract centroids (`x_centroid`, `y_centroid`)
   - Store segment label for each source

**Returns:**
```python
{
    "segmentation_image": SegmentationImage,  # photutils SegmentationImage
    "positions": (n_sources, 2) ndarray,      # detected centroids (x, y)
    "labels": 1D ndarray,                     # segment label for each source
}
```

**Notes:**
- Deblending is essential: without it, two close stars would be a single segment and we couldn't separate their fluxes.
- The detection catalog positions (centroids) are what we use for photometry. This ensures each aperture is centered on a detected source.
- Returns positions as numpy arrays to match the existing `extract_growth_curves` API.
- Deblending uses fixed defaults (`n_levels=32`, `contrast=0.001`) — sufficient for test simulations.

## Step 2: Contamination via Masked Growth Curves

### Modified function: `_extract_single_growth_curve(image, position, radii, error=None, mask=None)`

Add a `mask` parameter that is passed directly to `CurveOfGrowth`. The existing behavior is unchanged when `mask=None`.

**How CurveOfGrowth handles mask:**
- `mask` is a boolean 2D array of the same shape as `image`.
- Pixels where `mask` is `True` are **excluded** from the aperture sum.
- We use `mask = (segmentation_image.data != label_i)` which excludes all pixels not belonging to source `i`.

### Modified function: `extract_growth_curves(image, positions, radii=None, error=None, segmentation_image=None, labels=None)`

**New parameters:**
- `segmentation_image`: SegmentationImage or None. When provided, contamination is computed.
- `labels`: 1D array of segment labels (required if `segmentation_image` is provided).

**Algorithm:**

```python
def extract_growth_curves(image, positions, radii=None, error=None,
                          segmentation_image=None, labels=None):
    if radii is None:
        radii = np.logspace(np.log10(3), np.log10(30), num=10)

    n_sources = len(positions)
    n_radii = len(radii)
    flux = np.zeros((n_sources, n_radii))
    flux_err = np.zeros((n_sources, n_radii))

    if segmentation_image is not None:
        flux_clean = np.zeros((n_sources, n_radii))
        seg_data = segmentation_image.data

    for i, pos in enumerate(positions):
        # a) Total growth curve (existing behavior)
        _, profile, profile_err = _extract_single_growth_curve(
            image, pos, radii, error=error
        )
        flux[i] = profile
        flux_err[i] = profile_err

        # b) Clean growth curve (others masked)
        if segmentation_image is not None:
            mask = seg_data != labels[i]
            _, clean_profile, _ = _extract_single_growth_curve(
                image, pos, radii, mask=mask
            )
            flux_clean[i] = clean_profile

    result = {
        "radius": radii,
        "flux": flux,
        "flux_err": flux_err,
    }
    if segmentation_image is not None:
        result["flux_clean"] = flux_clean
        result["contamination"] = flux - flux_clean

    return result
```

**Returns (with segmentation):**
```python
{
    "radius": radii,                    # 1D array
    "flux": flux_total,                 # (n_sources, n_radii)
    "flux_err": flux_err_total,         # (n_sources, n_radii)
    "flux_clean": flux_clean,           # (n_sources, n_radii)
    "contamination": flux - flux_clean, # (n_sources, n_radii)
}
```

**Returns (without segmentation):**
Same as current behavior — only `radius`, `flux`, `flux_err`.

**Key detail:** The mask `seg_data != labels[i]` excludes all pixels whose segment label is not `labels[i]`. This includes background pixels (label 0), which is correct because the image is background-subtracted — those pixels are near zero and excluding them has no effect.

## Step 3: Cross-Match (Validation Helper)

### New function: `cross_match(input_positions, detected_positions, tolerance=5.0)`

**Purpose:** Match input/simulated positions to detected positions for validation.

**Returns:**
```python
{
    "match_indices": ndarray,    # for each input position, index in detected (or -1)
    "match_distances": ndarray,  # distance in pixels (or inf if unmatched)
}
```

**Algorithm:** Nearest-neighbor matching. For each input position, find the closest detected position. If distance > tolerance, mark as unmatched (-1).

## File Changes

### `src/gcphotom/aperture.py`

**New imports:**
```python
from photutils.segmentation import (
    SegmentationImage,
    detect_sources,
    detect_threshold,
    deblend_sources,
    SourceCatalog,
)
```

**New public functions:**
1. `detect_and_segment(image, background, n_sigma=3.0, n_pixels=10, deblend=True)` — detection pipeline
2. `cross_match(input_positions, detected_positions, tolerance=5.0)` — validation helper

**Modified functions:**
1. `_extract_single_growth_curve` — add `mask=None` parameter, pass to `CurveOfGrowth`
2. `extract_growth_curves` — add `segmentation_image` and `labels` parameters; when provided, call `_extract_single_growth_curve` twice per source (with/without mask) and compute contamination

**No new private functions needed** — the mask is created inline as `seg_data != labels[i]`.

### `src/gcphotom/__init__.py`

Add `detect_and_segment` and `cross_match` to imports and `__all__`.

### `tests/test_aperture.py`

All tests use **controlled catalogs** with known bright, well-separated positions (not random fluxes from `make_source_catalog`).

A helper fixture creates a catalog with specified positions and high flux:

```python
@pytest.fixture
def controlled_catalog():
    """Create a catalog with known bright, well-separated sources."""
    def _make(positions, flux=1e5, shape=(256, 256), background=100, seed=42):
        cat = Table()
        cat["x"] = np.array([p[0] for p in positions])
        cat["y"] = np.array([p[1] for p in positions])
        cat["flux"] = np.full(len(positions), flux)
        return cat, simulate_image(shape, cat, gamma=2.5, alpha=3.0,
                                    background=background, seed=seed)
    return _make
```

**New test class: `TestDetectAndSegment`**

| Test | Description |
|------|-------------|
| `test_returns_expected_keys` | Result dict has `segmentation_image`, `positions`, `labels` |
| `test_detects_all_well_separated` | 5 bright separated stars → all 5 detected |
| `test_positions_close_to_truth` | Detected centroids within 1 pixel of injected positions |
| `test_labels_are_unique` | Each detected source has a distinct segment label |
| `test_deblends_close_pair` | 2 bright stars at 5 px separation → 2 separate labels |

**New test class: `TestExtractGrowthCurvesWithSegmentation`**

Tests the contamination computation via `extract_growth_curves` with segmentation.

| Test | Description |
|------|-------------|
| `test_returns_contamination_keys` | With segmentation, result has `contamination` and `flux_clean` |
| `test_without_segmentation_no_contamination` | Without segmentation, no contamination keys |
| `test_isolated_source_near_zero_contamination` | Single isolated star → contamination ≈ 0 at all radii |
| `test_overlapping_pair_has_contamination` | Two close stars → contamination > 0 at large radii |
| `test_contamination_grows_with_radius` | Contamination increases with radius for contaminated source |
| `test_contamination_non_negative` | contamination ≥ 0 everywhere |
| `test_contamination_leq_total_flux` | contamination ≤ flux everywhere |
| `test_flux_clean_leq_flux_total` | flux_clean ≤ flux everywhere |
| `test_output_shapes` | All arrays have correct shapes |
| `test_end_to_end_simulated` | Full pipeline: simulate → detect → extract → contamination sensible |

**New test class: `TestCrossMatch`**

| Test | Description |
|------|-------------|
| `test_all_matched_for_well_separated` | All simulated positions match within tolerance |
| `test_unmatched_beyond_tolerance` | Position beyond tolerance → -1 |
| `test_close_pair_both_matched` | Close sources both matched to distinct detected sources |

## Design Decisions & Rationale

### 1. Contamination as absolute flux difference

`contamination = flux_total - flux_clean` gives absolute contaminating flux. Users can compute the fraction as `contamination / flux_total` if needed. This avoids division-by-zero for faint sources and keeps the units consistent.

### 2. Masking via CurveOfGrowth mask parameter

`CurveOfGrowth` accepts a boolean `mask` where `True` = excluded from the aperture sum. We pass `mask = (seg.data != label_i)` which excludes all pixels not belonging to source `i`. This is cleaner than manually zeroing pixels in the image — the masking is handled by photutils' exact overlap computation.

### 3. No standalone `estimate_contamination` function

Contamination estimation is integrated into `extract_growth_curves` by calling `_extract_single_growth_curve` twice per source (with and without mask). This avoids code duplication and keeps the contamination computation tied to the growth curve extraction.

### 4. Deblending by default with fixed parameters

Without deblending, blended sources share a single segment label. Deblending parameters use fixed defaults (`n_levels=32`, `contrast=0.001`) which work on the test simulations.

### 5. SegmentationImage object passed as-is

We pass the `SegmentationImage` object (not just the data array) to `extract_growth_curves`. The `.data` attribute provides the integer label array for boolean indexing. The object also carries `.labels` which lists all valid labels — useful for validation.

### 6. Detection catalog and segmentation are consistent

The `detect_and_segment` function produces both the positions and the segmentation image. They are guaranteed to be consistent: each position corresponds to a valid segment label. No cross-match is needed within the contamination estimation pipeline.

### 7. Error on contamination not reported

The errors on `flux_total` and `flux_clean` are correlated because they share pixels. We report only the point estimate for `contamination` and defer proper error propagation.
