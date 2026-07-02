"""2D background estimation using photutils.Background2D."""

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground


def estimate_background(image, box_size=(50, 50), filter_size=(3, 3), mask=None):
    """Estimate 2D background level and variance maps.

    Uses a mesh-based approach: divides the image into boxes, computes
    sigma-clipped median in each box, median-filters the mesh, then
    interpolates to full resolution.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Image data.
    box_size : tuple of int, optional
        Box size for mesh-based background estimation. Should be larger
        than typical source sizes but small enough to capture background
        variation. Default ``(50, 50)``.
    filter_size : tuple of int, optional
        Median filter window size applied to the mesh to suppress local
        under/over-estimation. Default ``(3, 3)``.
    mask : 2D `~numpy.ndarray` of bool or None
        Boolean mask where ``True`` indicates pixels to exclude (e.g.
        source mask from segmentation).

    Returns
    -------
    background : 2D `~numpy.ndarray`
        Full-resolution background level map.
    background_variance : 2D `~numpy.ndarray`
        Full-resolution background variance map (RMS squared).
    """
    bkg = Background2D(
        image,
        box_size=box_size,
        filter_size=filter_size,
        sigma_clip=SigmaClip(sigma=3.0, maxiters=10),
        bkg_estimator=MedianBackground(),
        mask=mask,
    )
    return bkg.background, bkg.background_rms**2
