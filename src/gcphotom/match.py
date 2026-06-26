import numpy as np


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


def cross_match(input_positions, detected_positions, tolerance=5.0):
    """Match input positions to detected positions (efficient implementation).

    Parameters
    ----------
    input_positions : (N, 2) ndarray
        Query positions (e.g. true coordinates). Each row is (x, y) or (ra, dec) ?
        This wrapper assumes pixel-like cartesian coordinates (same as before).
    detected_positions : (M, 2) ndarray
        Reference positions to search in.
    tolerance : float
        Max distance for a valid match (same units as coordinates).

    Returns
    -------
    dict
        ``match_indices`` : (N,) int, index into detected or -1
        ``match_distances`` : (N,) float, distance or +inf
    """
    input_positions = np.asarray(input_positions, dtype=float)
    detected_positions = np.asarray(detected_positions, dtype=float)
    n_in = len(input_positions)
    if n_in == 0:
        return {
            "match_indices": np.array([], dtype=int),
            "match_distances": np.array([], dtype=float),
        }
    if len(detected_positions) == 0:
        return {
            "match_indices": np.full(n_in, -1, dtype=int),
            "match_distances": np.full(n_in, np.inf),
        }
    ref = {"x": detected_positions[:, 0], "y": detected_positions[:, 1]}
    qry = {"x": input_positions[:, 0], "y": input_positions[:, 1]}
    idx, dist = match(
        ref,
        qry,
        project=False,
        xy=True,
        radius=float(tolerance),
        compute_distances=True,
    )
    return {"match_indices": idx, "match_distances": dist}
