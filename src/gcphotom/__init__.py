"""GCPhotom - Differentiable Growth Curve Photometry with Profile Fitting."""

__version__ = "0.1.0"

from gcphotom.aperture import estimate_error, extract_growth_curves
from gcphotom.gcmodel import Fitter
from gcphotom.simulator import make_source_catalog, simulate_image

__all__ = [
    "estimate_error",
    "extract_growth_curves",
    "Fitter",
    "make_source_catalog",
    "simulate_image",
]
