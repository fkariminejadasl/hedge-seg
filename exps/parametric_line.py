from math import comb

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Bezier (Bernstein basis)
# ----------------------------
def bezier_curve_bernstein(control_points, n=200):
    """
    Evaluate a Bezier curve using the Bernstein basis.

    control_points: (m, dim) array-like, m = degree+1
    n: number of samples along t in [0, 1]
    returns: (n, dim) curve points
    """
    P = np.asarray(control_points, dtype=float)
    m, dim = P.shape
    degree = m - 1

    ts = np.linspace(0.0, 1.0, n)

    # Build Bernstein basis matrix B of shape (n, m):
    # B[i, k] = C(degree, k) * t^k * (1-t)^(degree-k)
    B = np.empty((n, m), dtype=float)
    for k in range(m):
        B[:, k] = comb(degree, k) * (ts**k) * ((1.0 - ts) ** (degree - k))

    # Curve points are linear combination of control points
    # (n, m) @ (m, dim) -> (n, dim)
    curve = B @ P
    return curve


# ----------------------------
# Bezier (De Casteljau)
# ----------------------------
def bezier_curve(control_points, n=200):
    """
    control_points: (m, dim) array-like
    returns: (n, dim) curve points
    """
    P = np.asarray(control_points, dtype=float)
    ts = np.linspace(0.0, 1.0, n)
    curve = np.empty((n, P.shape[1]))

    for i, t in enumerate(ts):
        Q = P.copy()
        # De Casteljau
        for r in range(1, len(P)):
            Q[:-r] = (1 - t) * Q[:-r] + t * Q[1 : len(P) - r + 1]
        curve[i] = Q[0]
    return curve


# ----------------------------
# B-spline (Cox-de Boor)
# ----------------------------
def make_open_uniform_knot_vector(num_ctrl_pts, degree):
    """
    Open-uniform knot vector on [0,1].
    length = num_ctrl_pts + degree + 1
    """
    n = num_ctrl_pts
    p = degree
    # p+1 zeros, p+1 ones, uniform middle
    if n <= p:
        raise ValueError("num_ctrl_pts must be > degree")
    n_spans = n - p  # number of knot spans
    interior = np.linspace(0.0, 1.0, n_spans + 1)[1:-1]  # exclude endpoints
    kv = np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])
    return kv


def bspline_basis(i, p, t, kv):
    """Cox-de Boor recursion for basis N_{i,p}(t)."""
    if p == 0:
        if (kv[i] <= t < kv[i + 1]) or (
            t == kv[-1] and kv[i] < kv[i + 1] and kv[i + 1] == kv[-1]
        ):
            return 1.0
        return 0.0

    denom1 = kv[i + p] - kv[i]
    denom2 = kv[i + p + 1] - kv[i + 1]

    term1 = 0.0
    term2 = 0.0

    if denom1 != 0:
        term1 = (t - kv[i]) / denom1 * bspline_basis(i, p - 1, t, kv)
    if denom2 != 0:
        term2 = (kv[i + p + 1] - t) / denom2 * bspline_basis(i + 1, p - 1, t, kv)

    return term1 + term2


def bspline_curve(control_points, degree=3, n=300, knot_vector=None):
    """
    control_points: (m, dim) array-like
    degree: spline degree p
    knot_vector: optional custom knot vector
    returns: (n, dim) curve points
    """
    P = np.asarray(control_points, dtype=float)
    m, dim = P.shape
    p = degree

    kv = knot_vector
    if kv is None:
        kv = make_open_uniform_knot_vector(m, p)
    kv = np.asarray(kv, dtype=float)

    ts = np.linspace(kv[p], kv[-p - 1], n)  # valid parameter range for open knot vector
    curve = np.zeros((n, dim))

    for k, t in enumerate(ts):
        point = np.zeros(dim)
        for i in range(m):
            b = bspline_basis(i, p, t, kv)
            point += b * P[i]
        curve[k] = point

    return curve, kv


# ----------------------------
# Demo
# ----------------------------
if __name__ == "__main__":
    ctrl = np.array(
        [
            # [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 3.0],
            [4.0, 0.0],
            [5.0, 2.0],
        ]
    )

    bez = bezier_curve(ctrl, n=200)
    bbez = bezier_curve_bernstein(ctrl, n=200)
    bsp, kv = bspline_curve(ctrl, degree=3, n=400)

    plt.figure()
    plt.plot(ctrl[:, 0], ctrl[:, 1], "o-", label="control polygon")
    plt.plot(bez[:, 0], bez[:, 1], label="Bezier")
    plt.plot(bbez[:, 0], bbez[:, 1], label="Bezier (Bernstein)")
    plt.plot(bsp[:, 0], bsp[:, 1], label="B-spline (deg=3)")
    plt.axis("equal")
    plt.legend()
    plt.title("Bezier vs B-spline")
    plt.show(block=False)
    print("Done")
