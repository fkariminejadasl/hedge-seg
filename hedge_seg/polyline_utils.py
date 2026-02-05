import math


# ----------------------------
# Geometry helpers
# ----------------------------
def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _cumulative_lengths(points):
    """Return cumulative arc-lengths along the polyline."""
    cum = [0.0]
    for i in range(1, len(points)):
        cum.append(cum[-1] + _dist(points[i - 1], points[i]))
    return cum


def resample_polyline_equidistant(points, n_points):
    """
    Resample a polyline into n_points approximately equidistant points along arc length.
    Keeps endpoints. Returns list of [x, y] floats.
    """
    if not points:
        return []
    if len(points) == 1:
        return [list(map(float, points[0])) for _ in range(n_points)]

    pts = [tuple(p) for p in points]
    cum = _cumulative_lengths(pts)
    total = cum[-1]

    if total == 0.0:
        # All points identical
        return [list(map(float, pts[0])) for _ in range(n_points)]

    # Target distances along the curve
    targets = [i * total / (n_points - 1) for i in range(n_points)]

    out = []
    seg_idx = 0
    for t in targets:
        while seg_idx < len(cum) - 2 and cum[seg_idx + 1] < t:
            seg_idx += 1

        d0 = cum[seg_idx]
        d1 = cum[seg_idx + 1]
        p0 = pts[seg_idx]
        p1 = pts[seg_idx + 1]

        if d1 == d0:
            out.append([float(p0[0]), float(p0[1])])
            continue

        alpha = (t - d0) / (d1 - d0)
        x = p0[0] + alpha * (p1[0] - p0[0])
        y = p0[1] + alpha * (p1[1] - p0[1])
        # out.append([int(round(x)), int(round(y))]) # int
        out.append([round(x, 1), round(y, 1)])

    return out


def bbox_from_polyline(line):
    if not line:
        return None
    xs = [p[0] for p in line]
    ys = [p[1] for p in line]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
