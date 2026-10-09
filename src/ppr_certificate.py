"""Exact residual audit of an approximate personalized PageRank vector.

This is a correctness guard, not a PageRank solver. Rational work is charged.
"""
from fractions import Fraction as F
import numpy as np

from sparse_c2fuse import _canonical_sparse


def certify_ppr(p, source, estimate, alpha):
    """Return a rigorous L1 error bound for alpha*(I-(1-alpha)*P)^-1*source."""
    p = _canonical_sparse(p)
    source, estimate = np.asarray(source), np.asarray(estimate)
    if any(v.shape != (p.shape[0],) for v in (source, estimate)):
        raise ValueError("Source and estimate must be finite real vectors of length n")
    if np.any(p.data < 0):
        raise ValueError("PageRank operator must be nonnegative")
    try:
        alpha = F(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Alpha must be finite and rational") from exc
    alpha = F(int(alpha.numerator), int(alpha.denominator))
    if not 0 < alpha <= 1:
        raise ValueError("Require 0 < alpha <= 1")
    try:
        source, estimate = ([F(v.item() if isinstance(v, np.generic) else v) for v in vector]
                            for vector in (source, estimate))
        source, estimate = ([F(int(v.numerator), int(v.denominator)) for v in vector]
                            for vector in (source, estimate))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Source and estimate must contain finite real rational values") from exc
    beta = 1-alpha
    column_sums = [F(0)] * p.shape[0]
    residual = [alpha*x-y for x, y in zip(source, estimate)]
    for row in range(p.shape[0]):
        for edge in range(int(p.indptr[row]), int(p.indptr[row+1])):
            column, weight = int(p.indices[edge]), F(float(p.data[edge]))
            column_sums[column] += weight
            residual[row] += beta * weight * estimate[column]
    contraction = beta * max(column_sums)
    if contraction >= 1:
        raise ValueError("Cannot certify with this L1 contraction bound")
    residual_norm = sum(abs(v) for v in residual)
    return {"l1_error_bound": residual_norm/(1-contraction),
            "exact_residual_l1": residual_norm, "contraction_bound": contraction,
            "meaning": "Exact stored operator and provided rational parameter/state values"}
