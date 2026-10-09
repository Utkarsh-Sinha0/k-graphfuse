"""C2Sensitivity: certify output AND directional derivative truncation.

Derived from the assumed complete Crouzeix theorem via a scaled block lift.
The block matrix is used in the proof and never materialized in execution.
Floating execution and coefficient conversion errors are excluded.
"""
from fractions import Fraction as F
from types import MappingProxyType

import numpy as np

from c2fuse import _choose_degree, norm_upper
from fast_bounds import fast_sparse_bounds
from sparse_c2fuse import _canonical_sparse, compile_sparse, sparse_bounds


def _tail_and_derivative(norms, first, u):
    value, derivative = F(0), F(0)
    for k in range(len(norms)-1, -1, -1):
        derivative = derivative*u + value
        value = value*u + (norms[k] if k >= first else 0)
    return value, derivative


def derivative_tail_bound(norms, degree, rho, radius, eta, constant=F(2)):
    """Convex exact-rational scaling search, compared with the classical tail."""
    indices = [k for k in range(degree+1, len(norms)) if k and norms[k]]
    if not indices or not eta:
        return F(0), None, "zero derivative tail"
    classical = eta*sum(k*norms[k]*rho**(k-1) for k in indices)
    if not radius or radius >= rho or indices[-1] == 1:
        return classical, None, "classical derivative"
    first = degree+1
    lo = 2*radius / ((indices[-1]-1)*eta)
    hi = 2*radius / (max(1, indices[0]-1)*eta)
    def slope(t):
        value, derivative = _tail_and_derivative(norms, first, radius+t*eta/2)
        return t*eta*derivative/2-value
    while slope(hi) < 0:
        hi *= 2
    for _ in range(24):
        middle = (lo+hi)/2
        if slope(middle) < 0:
            lo = middle
        else:
            hi = middle
    scale = (lo+hi)/2
    value, _ = _tail_and_derivative(norms, first, radius+scale*eta/2)
    lifted = constant*value/scale
    if lifted < classical:
        return lifted, scale, "scaled complete Crouzeix derivative"
    return classical, None, "classical derivative"


def compile_sensitivity(a, direction, stages, value_tolerance, derivative_tolerance,
                        *, constant=F(2), bounds="fast", parameter_radius=0):
    """Uniform output/derivative certificate for |theta| <= parameter_radius."""
    full = compile_sparse(a, stages, 0, constant=constant, bounds=bounds)
    direction = _canonical_sparse(direction)
    if direction.shape != full["a"].shape:
        raise ValueError("Direction must match the graph operator shape")
    try:
        value_tolerance, derivative_tolerance = F(value_tolerance), F(derivative_tolerance)
        parameter_radius = F(parameter_radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Budgets must be finite real rational values") from exc
    if value_tolerance < 0 or derivative_tolerance < 0 or parameter_radius < 0:
        raise ValueError("Budgets must be nonnegative")
    if bounds == "fast":
        try:
            eta = fast_sparse_bounds(direction)[0]
        except OverflowError:
            eta = sparse_bounds(direction)[0]
    else:
        eta = sparse_bounds(direction)[0]
    coefficients = full["coefficients"]
    norms = [norm_upper(b) for b in coefficients]
    rho = full["norm_radius"]+parameter_radius*eta
    radius = full["radius"]+parameter_radius*eta
    degree, _, _ = _choose_degree(coefficients, rho, radius,
                                   value_tolerance, full["constant"])
    while True:
        gradient_bound, scale, method = derivative_tail_bound(
            norms, degree, rho, radius, eta, full["constant"])
        if gradient_bound <= derivative_tolerance:
            break
        degree = next(k for k in range(degree+1, len(norms)) if norms[k])
    crouzeix = full["constant"]*sum(norms[k]*radius**k for k in range(degree+1, len(norms)))
    classical = sum(norms[k]*rho**k for k in range(degree+1, len(norms)))
    for array in (direction.data, direction.indices, direction.indptr):
        array.setflags(write=False)
    return MappingProxyType({**full, "coefficients": coefficients[:degree+1],
        "float_coefficients": full["float_coefficients"][:degree+1], "degree": degree,
        "error_bound": min(crouzeix, classical), "tolerance": value_tolerance,
        "direction": direction, "direction_norm_bound": eta,
        "derivative_error_bound": gradient_bound, "derivative_tolerance": derivative_tolerance,
        "derivative_scale": scale, "derivative_bound_method": method,
        "parameter_radius": parameter_radius, "parameter_norm_radius": rho,
        "parameter_numerical_radius": radius,
        "bound_method": "complete Crouzeix" if crouzeix < classical else "classical norm"})


def evaluate_sensitivity(plan, x, *, theta=0):
    """Coupled Horner; derivative of the retained polynomial, without finite differences."""
    if np.iscomplexobj(x):
        raise ValueError("State must be real")
    x = np.asarray(x, dtype=float)
    try:
        theta = float(theta)
        valid_theta = abs(F(theta)) <= plan["parameter_radius"]
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError("Theta must be finite and within the certified interval") from exc
    if not valid_theta:
        raise ValueError("Theta is outside the certified interval; recompile")
    coefficients = plan["float_coefficients"]
    if x.shape != (plan["a"].shape[0], coefficients[0].shape[0]) or not np.all(np.isfinite(x)):
        raise ValueError("Invalid state shape or nonfinite values")
    y = x @ coefficients[-1].T
    derivative = np.zeros_like(y)
    operator = plan["a"] if not theta else plan["a"]+theta*plan["direction"]
    with np.errstate(over="ignore", invalid="ignore"):
        for coefficient in reversed(coefficients[:-1]):
            derivative = operator @ derivative + plan["direction"] @ y
            y = operator @ y + x @ coefficient.T
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(derivative)):
        raise ArithmeticError("Floating sensitivity execution overflowed")
    return y, derivative
