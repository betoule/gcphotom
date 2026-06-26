import numpy as np

from astropy.table import Table


def _euclidean(x1, y1, x2, y2):
    """Euclidean distance between points or arrays of points."""
    return np.sqrt(
        (np.asarray(x1) - np.asarray(x2)) ** 2 + (np.asarray(y1) - np.asarray(y2)) ** 2
    )


def _haversine(ra1, dec1, ra2, dec2):
    """Haversine angular distance (input in radians)."""
    dra = np.asarray(ra1) - np.asarray(ra2)
    return np.arccos(
        np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    )


def _gnomonic_projection(ra, dec, center=None):
    """Project ra/dec (degrees) to tangent-plane x/y using gnomonic projection.

    If center is None, use mean of the inputs.
    """
    ra = np.radians(ra)
    dec = np.radians(dec)
    if center is None:
        center = ra.mean(), dec.mean()
    else:
        center = np.radians(center)
    s, c = np.sin(dec), np.cos(dec)
    c2 = np.cos(center[0] - ra)
    d = np.sin(center[1]) * s + np.cos(center[1]) * c * c2
    x = (np.cos(center[1]) * np.sin(center[0] - ra)) / d
    y = (np.sin(center[1]) * c - np.cos(center[1]) * s * c2) / d
    return x, y


class NearestNeighAssoc:
    """Fixed-radius nearest-neighbor using a uniform grid for candidate pruning.

    Internal implementation detail. Not part of the public API contract.
    """

    def __init__(self, first=None, radius=1.0):
        self.radius = float(radius)
        self.belongs = {}
        self.clusters = []  # list of [x, y, count]
        self.x_bins = None
        self.y_bins = None
        if self.radius <= 0:
            self.radius = 0.0
            if first is not None:
                fx = np.asarray(first[0], dtype=float)
                fy = np.asarray(first[1], dtype=float)
                if len(fx) > 0:
                    self.clusters = [[fx[k], fy[k], 0] for k in range(len(fx))]
            return
        if first is not None:
            fx = np.asarray(first[0], dtype=float)
            fy = np.asarray(first[1], dtype=float)
            if len(fx) > 0:
                xmin, xmax = fx.min(), fx.max()
                ymin, ymax = fy.min(), fy.max()
                self.x_bins = np.arange(
                    xmin - 0.01 * self.radius, xmax + 0.01 * self.radius, self.radius
                )
                self.y_bins = np.arange(
                    ymin - 0.01 * self.radius, ymax + 0.01 * self.radius, self.radius
                )
                self.clusters = [[fx[k], fy[k], 0] for k in range(len(fx))]
                ii = np.digitize(fx, self.x_bins)
                jj = np.digitize(fy, self.y_bins)
                for k, (ik, jk) in enumerate(zip(ii, jj)):
                    self.belongs.setdefault((ik, jk), []).append(k)

    def _ensure_bins(self, x, y):
        if self.x_bins is None:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if len(x) == 0:
                # create a dummy bin range; match will return all -1 anyway
                self.x_bins = np.array([0.0])
                self.y_bins = np.array([0.0])
                return
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            self.x_bins = np.arange(
                xmin - 0.01 * self.radius, xmax + 0.01 * self.radius, self.radius
            )
            self.y_bins = np.arange(
                ymin - 0.01 * self.radius, ymax + 0.01 * self.radius, self.radius
            )

    def append(self, x, y, metric=_haversine):
        """Append new points and return their assigned cluster indices (for assoc use)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self._ensure_bins(x, y)
        ii = np.digitize(x, self.x_bins)
        jj = np.digitize(y, self.y_bins)
        index = np.zeros(len(x), dtype=int)
        for k, (ik, jk) in enumerate(zip(ii, jj)):

            cands = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    cands.extend(self.belongs.get((ik + di, jk + dj), []))
            if cands:
                xs = [self.clusters[l][0] for l in cands]
                ys = [self.clusters[l][1] for l in cands]
                dists = metric(x[k], y[k], xs, ys)
                l = np.argmin(dists)
                if dists[l] < self.radius:
                    m = cands[l]
                    index[k] = m
                    if len(self.clusters[m]) > 2:
                        self.clusters[m][2] += 1
                    continue
            # new cluster
            clu_i = len(self.clusters)
            index[k] = clu_i
            self.clusters.append([x[k], y[k], 1])
            self.belongs.setdefault((ik, jk), []).append(clu_i)
        return index

    def match(self, x, y, metric=_haversine):
        """Return for each (x,y) the index in clusters or -1 if outside radius."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.radius == 0:
            # exact match only
            index = np.full(len(x), -1, dtype=int)
            if not self.clusters:
                return index
            refx = np.array([c[0] for c in self.clusters])
            refy = np.array([c[1] for c in self.clusters])
            for k, (xk, yk) in enumerate(zip(x, y)):
                eq = (refx == xk) & (refy == yk)
                poss = np.where(eq)[0]
                if len(poss):
                    index[k] = poss[0]
            return index
        if self.x_bins is None or len(self.clusters) == 0:
            return np.full(len(x), -1, dtype=int)
        ii = np.digitize(x, self.x_bins)
        jj = np.digitize(y, self.y_bins)
        index = np.full(len(x), -1, dtype=int)
        for k, (ik, jk) in enumerate(zip(ii, jj)):
            cands = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    cands.extend(self.belongs.get((ik + di, jk + dj), []))
            if cands:
                xs = [self.clusters[l][0] for l in cands]
                ys = [self.clusters[l][1] for l in cands]
                dists = metric(x[k], y[k], xs, ys)
                l = np.argmin(dists)
                if dists[l] <= self.radius:
                    index[k] = cands[l]
        return index

    def get_cat(self):
        """Return clusters as a structured array (ra, dec, n) — for legacy assoc only."""
        if not self.clusters:
            return np.rec.fromarrays(
                [np.array([]), np.array([]), np.array([])], names="ra,dec,n"
            )
        arr = np.array(self.clusters, dtype=float)
        arr[:, 0] = np.degrees(arr[:, 0])
        arr[:, 1] = np.degrees(arr[:, 1])
        n = arr[:, 2] if arr.shape[1] > 2 else np.zeros(len(arr))
        return np.rec.fromarrays([arr[:, 0], arr[:, 1], n], names="ra,dec,n")


def match(refcat, cat, project=True, xy=False, radius=1.0, compute_distances=False):
    """Match catalog ``cat`` against reference ``refcat`` with fixed radius.

    Parameters
    ----------
    refcat : mapping
        Reference objects. Keys: 'x','y' (xy=True) or 'ra','dec' in degrees.
    cat : mapping
        Objects to match. Same keys as refcat.
    project : bool
        Project ra/dec via gnomonic around refcat center, then Euclidean.
    xy : bool
        Use raw x/y coordinates (no projection).
    radius : float
        Matching radius in the final coordinate units (projected units or radians).
    compute_distances : bool
        If True return a second array with exact distances (inf for unmatched).

    Returns
    -------
    indices : ndarray[int]
        For each row in cat, index into refcat or -1.
    distances : ndarray[float], optional
        Only when compute_distances=True.

    Notes
    -----
    Greedy nearest neighbor: multiple cat entries may match the same ref entry.
    """

    def _get(d, k):
        return np.asarray(d[k])

    if xy:
        xref = _get(refcat, "x").astype(float)
        yref = _get(refcat, "y").astype(float)
        xq = _get(cat, "x").astype(float)
        yq = _get(cat, "y").astype(float)
        metric = _euclidean
        xref_p, yref_p = xref, yref
        x_p, y_p = xq, yq
    elif project:
        ra_ref = _get(refcat, "ra").astype(float)
        dec_ref = _get(refcat, "dec").astype(float)
        ra_q = _get(cat, "ra").astype(float)
        dec_q = _get(cat, "dec").astype(float)
        if len(ra_ref) == 0 or len(ra_q) == 0 and len(ra_q) == 0:
            idx = np.full(len(ra_q), -1, dtype=int)
            if compute_distances:
                return idx, np.full(len(ra_q), np.inf)
            return idx
        center = [ra_ref.mean(), dec_ref.mean()] if len(ra_ref) > 0 else None
        xref_p, yref_p = _gnomonic_projection(ra_ref, dec_ref, center=center)
        x_p, y_p = _gnomonic_projection(ra_q, dec_q, center=center)
        metric = _euclidean
    else:
        # pure spherical, radius expected in radians
        xref_p = np.radians(_get(refcat, "ra").astype(float))
        yref_p = np.radians(_get(refcat, "dec").astype(float))
        x_p = np.radians(_get(cat, "ra").astype(float))
        y_p = np.radians(_get(cat, "dec").astype(float))
        metric = _haversine

    if len(xref_p) == 0:
        idx = np.full(len(x_p), -1, dtype=int)
        if compute_distances:
            return idx, np.full(len(x_p), np.inf)
        return idx

    assoc = NearestNeighAssoc(first=[xref_p, yref_p], radius=radius)
    idx = assoc.match(x_p, y_p, metric=metric)

    if not compute_distances:
        return idx

    dist = np.full(len(idx), np.inf)
    good = idx >= 0
    if np.any(good):
        if metric is _euclidean:
            dist[good] = _euclidean(
                x_p[good],
                y_p[good],
                [xref_p[i] for i in idx[good]],
                [yref_p[i] for i in idx[good]],
            )
        else:
            dist[good] = _haversine(
                x_p[good],
                y_p[good],
                [xref_p[i] for i in idx[good]],
                [yref_p[i] for i in idx[good]],
            )
    return idx, dist


def _get_xy(obj):
    """Extract (x, y) arrays from ndarray, Table, or SourceCatalog-like object."""
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 0], arr[:, 1]
    if hasattr(obj, "colnames"):
        try:
            x = np.asarray(obj["x"], dtype=float)
            y = np.asarray(obj["y"], dtype=float)
            return x, y
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    if hasattr(obj, "x_centroid") and hasattr(obj, "y_centroid"):
        x = np.asarray(obj.x_centroid, dtype=float)
        y = np.asarray(obj.y_centroid, dtype=float)
        return x, y
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0], arr[:, 1]
    raise TypeError("Cannot extract (x, y) positions from input")


def _empty_table_like(b):
    if hasattr(b, "colnames"):
        return Table({c: np.array([], dtype=float) for c in b.colnames})
    return Table()


def _no_match_table(b, n):
    if hasattr(b, "colnames"):
        cols = {}
        for c in b.colnames:
            arr = np.asarray(b[c])
            if np.issubdtype(arr.dtype, np.number):
                cols[c] = np.full(n, np.nan, dtype=float)
            else:
                cols[c] = np.array([None] * n, dtype=object)
        return Table(cols)
    return Table()


def _build_matched_table(b, idx, n):
    cols = {}
    for c in b.colnames:
        arr = np.asarray(b[c])
        if np.issubdtype(arr.dtype, np.number):
            out = np.full(n, np.nan, dtype=float)
            good = idx >= 0
            out[good] = arr[idx[good]]
            cols[c] = out
        else:
            out = np.empty(n, dtype=object)
            out[:] = None
            good = idx >= 0
            out[good] = arr[idx[good]]
            cols[c] = out
    return Table(cols)


def cross_match(a, b, tolerance=5.0):
    """Match positions or catalogs.

    Dispatch based on input types:

    - If both ``a`` and ``b`` are (N,2) ndarrays: legacy behavior.
      Returns dict with ``match_indices`` and ``match_distances``.
      Semantics: for each row in ``a``, index into ``b`` (or -1).

    - Otherwise: catalog-oriented matching.
      ``a`` provides query positions (e.g. detected catalog),
      ``b`` is the reference catalog to reorder (e.g. simulated truth).
      Returns an :class:`~astropy.table.Table` of length ``len(a)``,
      containing columns from ``b`` reordered so that row ``i`` in the
      result corresponds to the ``i``-th entry in ``a``. Unmatched entries
      are filled with ``NaN`` (numeric columns) or ``None`` (others).
    """

    # Legacy array path
    def _looks_like_pos(x):
        try:
            arr = np.asarray(x, dtype=float)
            return arr.ndim == 2 and arr.shape[1] == 2
        except (ValueError, TypeError):
            return False

    result = None
    if _looks_like_pos(a) and _looks_like_pos(b):
        ap = np.asarray(a, dtype=float)
        bp = np.asarray(b, dtype=float)
        n_in = len(ap)
        if n_in == 0:
            result = {
                "match_indices": np.array([], dtype=int),
                "match_distances": np.array([], dtype=float),
            }
        elif len(bp) == 0:
            result = {
                "match_indices": np.full(n_in, -1, dtype=int),
                "match_distances": np.full(n_in, np.inf),
            }
        else:
            ref = {"x": bp[:, 0], "y": bp[:, 1]}
            qry = {"x": ap[:, 0], "y": ap[:, 1]}
            idx, dist = match(
                ref,
                qry,
                project=False,
                xy=True,
                radius=float(tolerance),
                compute_distances=True,
            )
            result = {"match_indices": idx, "match_distances": dist}
    else:
        # Catalog-oriented path: a=det (query), b=sim (ref catalog)
        dx, dy = _get_xy(a)
        sx, sy = _get_xy(b)
        n = len(dx)
        if n == 0:
            result = _empty_table_like(b)
        elif len(sx) == 0:
            result = _no_match_table(b, n)
        else:
            ref = {"x": sx, "y": sy}
            qry = {"x": dx, "y": dy}
            idx, _ = match(
                ref,
                qry,
                project=False,
                xy=True,
                radius=float(tolerance),
                compute_distances=True,
            )
            if hasattr(b, "colnames"):
                result = _build_matched_table(b, idx, n)
            else:
                result = {"match_indices": idx, "match_distances": np.full(n, np.inf)}

    return result
