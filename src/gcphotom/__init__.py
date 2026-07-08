"""GCPhotom - Differentiable Growth Curve Photometry with Profile Fitting."""

__version__ = "0.1.0"

from gcphotom.background import estimate_background
from gcphotom.aperture import (
    cross_match,
    detect_and_segment,
    extract_growth_curves,
)
from gcphotom.gcmodel import Fitter
from gcphotom.jaxfitter import tukey, pseudo_huber, cauchy, parameter_uncertainty
from gcphotom.psf_photometry import psf_photometry
from gcphotom.simulator import (
    make_realistic_source_catalog,
    simulate_image,
    make_test_source_catalog,
)
from gcphotom.galsim_simulator import simulate_image_galsim
from gcphotom.gaia_catalog import make_gaia_source_catalog

from . import match, montecarlo  # public submodules

__all__ = [
    "cross_match",
    "estimate_background",
    "detect_and_segment",
    "extract_growth_curves",
    "Fitter",
    "tukey",
    "pseudo_huber",
    "cauchy",
    "parameter_uncertainty",
    "make_gaia_source_catalog",
    "make_realistic_source_catalog",
    "make_test_source_catalog",
    "match",
    "simulate_image",
    "simulate_image_galsim",
]
