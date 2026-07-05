import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

from gcphotom.plots import binplot


class TestBinplot:
    @pytest.mark.parametrize(
        "noplot,xerr,logbins",
        [
            (True, True, False),
            (False, False, False),
            (False, True, True),
        ],
    )
    def test_returns_finite_binned_values(self, noplot, xerr, logbins):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.randn(50)

        xbinned, yplot, yerr = binplot(
            x, y, nbins=5, noplot=noplot, xerr=xerr, logbins=logbins
        )

        assert len(xbinned) == 5
        assert np.all(np.isfinite(yplot))
