"""Outward sparse norm bounds using binary64 reductions, then exact rationals.

Assumes finite IEEE binary64 additions with standard relative-error bounds.
The additive allowance also covers flushing tiny values in these reductions.
Overflow falls back to the exact planner; it is never accepted as a certificate.
"""
from fractions import Fraction as F
import sys

import numpy as np

from c2fuse import sqrt_upper

EPS = F(1, 2**52)
TINY = F(4, 2**1022)


def _row_sum_upper(a):
    if not a.nnz:
        return F(0)
    count = 2 * (int(a.nnz) + a.shape[0] + 1)
    product = count * EPS
    gamma = product / (1-product) if product < 1 else F(1)
    if gamma >= 1:
        raise OverflowError("Reduction too long for the floating error model")
    absolute = a.copy()
    absolute.data = np.abs(absolute.data)
    with np.errstate(over="ignore", invalid="ignore"):
        rows = np.asarray(absolute.sum(axis=1, dtype=np.float64)).ravel()
    if not np.all(np.isfinite(rows)):
        raise OverflowError("Sparse reduction overflow")
    observed = F(float(np.max(rows)))
    return (observed + count*TINY/(1-product)) / (1-gamma)


def fast_sparse_bounds(a):
    """Upper enclosures for the exact rational interpretation of stored CSR A."""
    if sys.float_info.radix != 2 or sys.float_info.mant_dig != 53:
        raise OverflowError("Use the exact planner on non-binary64 systems")
    if not a.nnz:
        return F(0), F(0)
    transpose = a.T.tocsr()
    row, col = _row_sum_upper(a), _row_sum_upper(transpose)
    with np.errstate(over="ignore", invalid="ignore"):
        symmetric, skew = a + transpose, a - transpose
    # Omitted rounded zeros are covered by the union of A and A.T's entries.
    pair_allowance = 2 * int(a.nnz) * TINY
    h = (_row_sum_upper(symmetric) + pair_allowance) / (2*(1-EPS))
    s = (_row_sum_upper(skew) + pair_allowance) / (2*(1-EPS))
    rho = sqrt_upper(row*col)
    return rho, min(rho, sqrt_upper(h*h+s*s))
