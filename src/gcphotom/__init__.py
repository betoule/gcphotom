"""GCPhotom - Differentiable Growth Curve Photometry with Profile Fitting."""

__version__ = "0.1.0"

from gcphotom.aperture import (
    cross_match,
    detect_and_segment,
    estimate_error,
    extract_growth_curves,
)
from gcphotom.gcmodel import Fitter
from gcphotom.simulator import make_realistic_source_catalog, simulate_image, make_test_source_catalog

__all__ = [
    "cross_match",
    "detect_and_segment",
    "estimate_error",
    "extract_growth_curves",
    "Fitter",
    "make_realistic_source_catalog",
    "make_test_source_catalog",
    "simulate_image",
]
