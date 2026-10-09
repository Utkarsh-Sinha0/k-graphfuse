"""Certified analytic-tail planning for symmetric PageRank resolvents.

Uses the established Chebyshev accelerator rather than a monomial expansion.
The certificate excludes floating coefficient and execution roundoff.
"""
from fractions import Fraction as F
import math
from types import MappingProxyType

import numpy as np

from c2fuse import sqrt_bounds
from sparse_c2fuse import _canonical_sparse


def compile_ppr(s, alpha, tolerance, *, weights=None):
    """Bind a symmetric operator; positive weights tighten its spectral enclosure."""
    s = _canonical_sparse(s)
    if (s-s.T).nnz:
        raise ValueError("Chebyshev backend requires an exactly symmetric stored operator")
    try:
        alpha, tolerance = F(alpha), F(tolerance)
        alpha = F(int(alpha.numerator), int(alpha.denominator))
        tolerance = F(int(tolerance.numerator), int(tolerance.denominator))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Alpha and tolerance must be finite real rational values") from exc
    if not 0 < alpha <= 1 or tolerance <= 0:
        raise ValueError("Require 0 < alpha <= 1 and tolerance > 0")
    if weights is None:
        weights = np.ones(s.shape[0])
    if np.iscomplexobj(weights):
        raise ValueError("Weights must be real")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (s.shape[0],) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Weights must be finite positive values, one per vertex")
    weights = [F(float(v)) for v in weights]
    radius = F(0)
    for row in range(s.shape[0]):
        total = sum(abs(F(float(s.data[e]))) * weights[int(s.indices[e])]/weights[row]
                    for e in range(int(s.indptr[row]), int(s.indptr[row+1])))
        radius = max(radius, total)
    b = (1-alpha)*radius
    if b >= 1:
        raise ValueError("Spectral enclosure reaches the resolvent pole; supply tighter weights")
    if not b:
        degree, tail, ratio, scale = 0, F(0), 0., float(alpha)
    else:
        root_lower = sqrt_bounds(1-b*b)[0]
        if not root_lower:
            raise ValueError("Pole too close for this small planner's root enclosure")
        ratio_upper, scale_upper = b/(1+root_lower), alpha/root_lower
        degree = 0
        tail = 2*scale_upper*ratio_upper/(1-ratio_upper)
        while tail > tolerance:
            degree += 1
            tail *= ratio_upper
        root = math.sqrt(float(1-b*b))
        ratio, scale = float(b)/(1+root), float(alpha)/root
        if not 0 <= ratio < 1 or not math.isfinite(scale):
            raise ValueError("Floating coefficients cannot represent this resolvent")
    for array in (s.data, s.indices, s.indptr):
        array.setflags(write=False)
    return MappingProxyType({"s": s, "radius": radius, "alpha": alpha, "degree": degree,
                             "ratio": ratio, "scale": scale, "error_bound": tail,
                             "tolerance": tolerance, "backend": "symmetric Chebyshev resolvent"})


def evaluate_ppr(plan, x):
    """Apply the prepared resolvent approximation to one or many feature columns."""
    if np.iscomplexobj(x):
        raise ValueError("State must be real")
    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2) or x.shape[0] != plan["s"].shape[0] or not np.all(np.isfinite(x)):
        raise ValueError("State must be finite, with one row per vertex")
    y = plan["scale"]*x
    if plan["degree"]:
        previous, current = x, (plan["s"] @ x)/float(plan["radius"])
        y = y + 2*plan["scale"]*plan["ratio"]*current
        for k in range(2, plan["degree"]+1):
            previous, current = current, 2*(plan["s"] @ current)/float(plan["radius"])-previous
            y = y + 2*plan["scale"]*plan["ratio"]**k*current
    if not np.all(np.isfinite(y)):
        raise ArithmeticError("Floating Chebyshev execution overflowed")
    return y
