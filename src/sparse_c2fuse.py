"""C2Fuse-Sparse: exact tail planning, sparse floating-point execution.

The certificate covers truncation on the canonical stored float64 operator,
interpreted exactly. It excludes floating coefficient conversion and roundoff.
"""
from fractions import Fraction
from types import MappingProxyType

import numpy as np
from scipy.sparse import csr_array, issparse

from c2fuse import _choose_degree, fuse, normalize, sqrt_upper


def sparse_bounds(a):
    """Exact row/column and Hermitian/skew bounds without an n-by-n array."""
    transpose = a.T.tocsr()
    transpose.sort_indices()
    row_max = col_max = h_max = s_max = Fraction(0)
    for i in range(a.shape[0]):
        p, p_end = int(a.indptr[i]), int(a.indptr[i + 1])
        t, t_end = int(transpose.indptr[i]), int(transpose.indptr[i + 1])
        row = col = h = s = Fraction(0)
        while p < p_end or t < t_end:
            j = int(a.indices[p]) if p < p_end else a.shape[0]
            k = int(transpose.indices[t]) if t < t_end else a.shape[0]
            left = right = Fraction(0)
            if j <= k:
                left = Fraction(float(a.data[p]))
                p += 1
            if k <= j:
                right = Fraction(float(transpose.data[t]))
                t += 1
            row += abs(left)
            col += abs(right)
            h += abs(left + right) / 2
            s += abs(left - right) / 2
        row_max, col_max = max(row_max, row), max(col_max, col)
        h_max, s_max = max(h_max, h), max(s_max, s)
    rho = sqrt_upper(row_max * col_max)
    radius = min(rho, sqrt_upper(h_max*h_max + s_max*s_max))
    return rho, radius


def _canonical_sparse(a):
    """Validate and copy the shared sparse-operator trust boundary."""
    if not issparse(a) or a.ndim != 2 or not a.shape[0] or a.shape[0] != a.shape[1]:
        raise ValueError("A must be a nonempty square SciPy sparse array or matrix")
    if np.issubdtype(a.dtype, np.complexfloating):
        raise ValueError("This implementation requires real entries")
    try:
        a = csr_array(a, dtype=np.float64, copy=True)
        a.check_format(full_check=True)
        a.sum_duplicates()
        a.eliminate_zeros()
        a.sort_indices()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid sparse operator") from exc
    if not np.all(np.isfinite(a.data)):
        raise ValueError("A must have finite float64 entries")
    return a


def compile_sparse(a, stages, tolerance=0, *, constant=Fraction(2), bounds="exact"):
    """Compile stages; bounds='fast' uses outward binary64 reduction bounds.

    The tolerance bounds exact output error divided by input Frobenius norm.
    Sparse inputs are copied, converted to float64, and canonicalized first.
    """
    a = _canonical_sparse(a)
    if bounds not in ("exact", "fast"):
        raise ValueError("Bounds must be 'exact' or 'fast'")
    try:
        tolerance, constant = Fraction(tolerance), Fraction(constant)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Tolerance and constant must be finite rational numbers") from exc
    if tolerance < 0 or constant < 2:
        raise ValueError("Require tolerance >= 0 and spectral constant >= 2")
    stages = [normalize(stage) for stage in stages]
    original_degree = sum(len(stage) - 1 for stage in stages)
    coefficients = fuse(stages)
    method_bounds = "exact rational rows"
    if bounds == "fast":
        from fast_bounds import fast_sparse_bounds
        try:
            rho, radius = fast_sparse_bounds(a)
            method_bounds = "outward binary64 rows"
        except OverflowError:
            rho, radius = sparse_bounds(a)
    else:
        rho, radius = sparse_bounds(a)
    degree, bound, method = _choose_degree(
        coefficients, rho, radius, tolerance, constant)
    kept = tuple(coefficients[:degree + 1])
    try:
        floats = tuple(np.asarray(b, dtype=np.float64) for b in kept)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Coefficients must fit finite float64 execution") from exc
    if any(not np.all(np.isfinite(b)) for b in floats):
        raise ValueError("Coefficients must fit finite float64 execution")
    for array in (a.data, a.indices, a.indptr, *kept, *floats):
        array.setflags(write=False)
    return MappingProxyType({
        "a": a, "coefficients": kept, "float_coefficients": floats,
        "degree": degree, "fused_degree": len(coefficients) - 1,
        "original_degree": original_degree, "stage_count": len(stages),
        "norm_radius": rho, "radius": radius, "error_bound": bound,
        "tolerance": tolerance, "constant": constant, "bound_method": method,
        "geometry_method": method_bounds})


def evaluate_sparse(plan, x):
    """Sparse Horner; no matrix powers, eigendecomposition or dense base."""
    if np.iscomplexobj(x):
        raise ValueError("State must be real")
    x = np.asarray(x, dtype=np.float64)
    a, coefficients = plan["a"], plan["float_coefficients"]
    expected_shape = (a.shape[0], coefficients[0].shape[0])
    if x.shape != expected_shape or not np.all(np.isfinite(x)):
        raise ValueError(f"State must be finite with shape {expected_shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        y = x @ coefficients[-1].T
        for coefficient in reversed(coefficients[:-1]):
            y = a @ y + x @ coefficient.T
    if not np.all(np.isfinite(y)):
        raise ArithmeticError("Floating execution overflowed")
    return y
