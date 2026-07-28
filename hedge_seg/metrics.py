"""
Detection metrics for polyline predictions.

Two metrics live here. Use `buffered_length_pr`. `matched_pr` is kept only so
the reason it was rejected stays reproducible.

buffered_length_pr (use this one)
    precision = share of predicted line length within r of any GT line
    recall    = share of GT line length within r of any predicted line
    Nothing is paired with anything. Partial coverage of a bent hedge counts as
    partly right, and a hedge that GT stored as several overlapping lines is
    not penalised.

matched_pr (do not use)
    Pairs one prediction with one GT line by Hungarian matching under a chamfer
    distance threshold, the way a box detector matches by IoU. Two ordinary
    cases break the pairing. A GT hedge that turns a corner is one L-shaped
    line, so a prediction covering one leg averages about 17 m of chamfer over
    the whole line and scores as a total miss. A hedge split into three GT
    lines can only be paired with one of them, so the other two count as
    misses. On the same predictions it reports F1 0.41 where buffered length
    reports 0.64, and the figures agree with 0.64.

Both are computed per image and then averaged over images. Never as a ratio of
two totals: errors of opposite sign cancel in a total, which is how the mean
line count made the epoch 150 checkpoint look better than it was.

Distances are in pixels. PDOK crops are 0.25 m per pixel, so 10 m is 40 px.
`meters_to_px` does the conversion.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

PDOK_PIXEL_SIZE_M = 0.25


def meters_to_px(meters, pixel_size_m=PDOK_PIXEL_SIZE_M):
    """Convert a distance in meters to pixels of a PDOK crop."""
    return float(meters) / float(pixel_size_m)


def polyline_length(polyline):
    """Arc length of one (K, 2) polyline, in pixels."""
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def straightness(polyline):
    """
    Chord length over arc length, in [0, 1]. 1 is a straight line, lower means
    more bend. Used to check whether predictions bend as often as the labels do.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    arc = polyline_length(pts)
    if arc < 1e-6:
        return 1.0
    return float(np.linalg.norm(pts[-1] - pts[0]) / arc)


def densify(polylines, step_px=4.0):
    """
    Resample every polyline to points at most step_px apart and stack them into
    one (N, 2) array.

    The buffered metric measures length, so the points have to be equally
    spaced. The stored polylines are not: they have 20 points each regardless
    of whether they span 20 m or 300 m, so counting stored points would weight
    a short hedge the same as a long one.
    """
    out = []
    for polyline in polylines:
        pts = np.asarray(polyline, dtype=np.float64)
        if pts.shape[0] < 2:
            out.append(pts.reshape(-1, 2))
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = seg.sum()
        if total < 1e-6:
            out.append(pts[:1])
            continue
        n = max(2, int(np.ceil(total / step_px)))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, total, n)
        out.append(np.stack([np.interp(t, cum, pts[:, k]) for k in (0, 1)], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(out, axis=0)


def _covered_fraction(points_a, points_b, radius_px):
    """Share of points_a lying within radius_px of any point of points_b."""
    if len(points_a) == 0:
        return None
    if len(points_b) == 0:
        return 0.0
    distances, _ = cKDTree(points_b).query(points_a)
    return float((distances <= radius_px).mean())


def buffered_length_pr(pred_polylines, gt_polylines, radius_px, step_px=4.0):
    """
    Buffered-length precision and recall for one image.

    Returns (precision, recall). An image with no GT and no predictions scores
    1.0 on both: nothing was there and nothing was drawn.
    """
    pred_points = densify(pred_polylines, step_px)
    gt_points = densify(gt_polylines, step_px)
    precision = _covered_fraction(pred_points, gt_points, radius_px)
    recall = _covered_fraction(gt_points, pred_points, radius_px)
    return (
        1.0 if precision is None else precision,
        1.0 if recall is None else recall,
    )


def chamfer_distance(polyline_a, polyline_b):
    """Symmetric mean chamfer distance between two point sets, in pixels."""
    a = np.asarray(polyline_a, dtype=np.float64)
    b = np.asarray(polyline_b, dtype=np.float64)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))


def chamfer_matrix(polylines_a, polylines_b):
    """Pairwise chamfer distances, shape (len(a), len(b))."""
    out = np.zeros((len(polylines_a), len(polylines_b)), dtype=np.float64)
    for i, a in enumerate(polylines_a):
        for j, b in enumerate(polylines_b):
            out[i, j] = chamfer_distance(a, b)
    return out


def matched_pr(pred_polylines, gt_polylines, thresh_px):
    """
    Chamfer + Hungarian precision and recall for one image. Rejected, see the
    module docstring. Returns (precision, recall, n_true_positive).
    """
    n_pred, n_gt = len(pred_polylines), len(gt_polylines)
    if n_pred == 0 or n_gt == 0:
        precision = 1.0 if n_pred == 0 and n_gt == 0 else 0.0
        recall = 1.0 if n_gt == 0 else 0.0
        return precision, recall, 0

    cost = chamfer_matrix(pred_polylines, gt_polylines)
    unmatchable = 1e6
    cost = np.where(cost <= thresh_px, cost, unmatchable)
    rows, cols = linear_sum_assignment(cost)
    n_tp = int((cost[rows, cols] < unmatchable).sum())
    return n_tp / n_pred, n_tp / n_gt, n_tp


def merge_close_polylines(polylines, thresh_px):
    """
    Join GT polylines that lie within thresh_px of each other, keeping the
    longest of each group. Tests the "one hedge stored as several overlapping
    lines" label artifact.

    Measured on the run 1 cluster val split it removes 0.8% of GT lines and
    moves F1 by 0.001, so the artifact is rare enough to ignore. Buffered
    length is immune to it anyway.
    """
    if len(polylines) < 2:
        return list(polylines)

    cost = chamfer_matrix(polylines, polylines)
    parent = list(range(len(polylines)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(polylines)):
        for j in range(i + 1, len(polylines)):
            if cost[i, j] <= thresh_px:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(len(polylines)):
        groups.setdefault(find(i), []).append(i)
    return [
        polylines[max(members, key=lambda k: polyline_length(polylines[k]))]
        for members in groups.values()
    ]


def macro_average(rows, key):
    """Mean of rows[i][key] over images. The only correct way to aggregate."""
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def f1(precision, recall):
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
