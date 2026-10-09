"""Fuse matrix-valued polynomial stages and bound discarded terms.

The constant-two bound uses OpenAI math family 325 as an assumed theorem.
Bounds cover exact-arithmetic truncation; floating-point execution adds rounding.
"""
from fractions import Fraction
from math import isqrt

import numpy as np


def rational_matrix(value):
    a = np.asarray(value, dtype=object)
    if a.ndim != 2 or not a.size:
        raise ValueError("Expected a nonempty real rational matrix")
    try:
        return np.array([[Fraction(x.item() if isinstance(x, np.generic) else x)
                          for x in row] for row in a], dtype=object)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Entries must be finite real rational numbers") from exc


def sqrt_bounds(value):
    """Relative outward root enclosure, calculated entirely with integers."""
    value = Fraction(value)
    if value < 0:
        raise ValueError("Square root requires a nonnegative number")
    if not value:
        return Fraction(0), Fraction(0)
    number, denominator = int(value.numerator), int(value.denominator)
    exponent = number.bit_length()-denominator.bit_length()
    if ((number < denominator*(1 << exponent)) if exponent >= 0
            else (number*(1 << -exponent) < denominator)):
        exponent -= 1
    scale = 1 << max(0, 42-exponent//2)
    numerator = int(value.numerator) * scale * scale
    root = isqrt(numerator // denominator)
    lower = Fraction(root, scale)
    if root * root * denominator < numerator:
        root += 1
    return lower, Fraction(root, scale)


def sqrt_upper(value):
    return sqrt_bounds(value)[1]


def norm_upper(a):
    one = max(sum(abs(x) for x in col) for col in a.T)
    infinity = max(sum(abs(x) for x in row) for row in a)
    return sqrt_upper(one * infinity)


def normalize(coefficients):
    values = [rational_matrix(b) for b in coefficients]
    if not values or any(b.shape != values[0].shape for b in values):
        raise ValueError("Coefficients need one common square shape")
    if values[0].shape[0] != values[0].shape[1]:
        raise ValueError("Coefficient matrices must be square")
    while len(values) > 1 and not np.any(values[-1]):
        values.pop()
    return values


def fuse(stages):
    """Stages are listed in execution order; coefficient order is preserved."""
    stages = [normalize(stage) for stage in stages]
    if not stages:
        raise ValueError("At least one stage is required")
    q = stages[0][0].shape[0]
    if any(stage[0].shape != (q, q) for stage in stages):
        raise ValueError("All stages must use the same channel count")
    if len(stages) == 1:
        return stages[0]
    current = [rational_matrix(np.eye(q, dtype=int))]
    for stage in stages:
        result = [np.full((q, q), Fraction(0), dtype=object)
                  for _ in range(len(current) + len(stage) - 1)]
        for j, left in enumerate(stage):
            for k, right in enumerate(current):
                result[j + k] += left @ right
        current = normalize(result)
    return current


def _choose_degree(coefficients, rho, radius, tolerance, constant):
    """Shared exact tail calculation for dense and sparse operators."""
    norms = [norm_upper(b) for b in coefficients]
    tails = [Fraction(0)] * (len(coefficients) + 1)
    ordinary = tails.copy()
    powers_r, powers_rho = [Fraction(1)], [Fraction(1)]
    for _ in range(1, len(coefficients)):
        powers_r.append(powers_r[-1]*radius)
        powers_rho.append(powers_rho[-1]*rho)
    for k in range(len(coefficients) - 1, -1, -1):
        tails[k] = tails[k + 1] + norms[k] * powers_r[k]
        ordinary[k] = ordinary[k + 1] + norms[k] * powers_rho[k]
    for degree in range(len(coefficients)):
        crouzeix = constant * tails[degree + 1]
        classical = ordinary[degree + 1]
        error_bound = min(crouzeix, classical)
        if error_bound <= tolerance:
            return degree, error_bound, (
                "complete Crouzeix" if crouzeix < classical else "classical norm")
    raise AssertionError("The full polynomial has a zero truncation tail")


def compile_pipeline(a, stages, tolerance=0, *, constant=Fraction(2)):
    """Bind a plan to A and an absolute operator-error budget.

    constant >= 2 permits a conservative pre-2026 reference comparison.
    The returned certificate is exact for the rational interpretation of inputs.
    """
    a = rational_matrix(a)
    if a.shape[0] != a.shape[1]:
        raise ValueError("A must be square")
    tolerance, constant = Fraction(tolerance), Fraction(constant)
    if tolerance < 0 or constant < 2:
        raise ValueError("Require tolerance >= 0 and spectral constant >= 2")
    stages = [normalize(stage) for stage in stages]
    original_degree = sum(len(stage) - 1 for stage in stages)
    coefficients = fuse(stages)
    rho = norm_upper(a)
    h = norm_upper((a + a.T) / 2)
    s = norm_upper((a - a.T) / 2)
    radius = min(rho, sqrt_upper(h*h + s*s))
    degree, error_bound, method = _choose_degree(
        coefficients, rho, radius, tolerance, constant)
    kept = tuple(coefficients[:degree + 1])
    for matrix in (a, *kept):
        matrix.setflags(write=False)
    return {"a": a, "coefficients": kept, "degree": degree,
            "fused_degree": len(coefficients) - 1,
            "original_degree": original_degree, "stage_count": len(stages),
            "radius": radius, "norm_radius": rho, "error_bound": error_bound,
            "bound_method": method,
            "constant": constant, "tolerance": tolerance}


def evaluate(plan, x, *, exact=False):
    """Apply the plan to an n-by-q state array without a Kronecker matrix."""
    a, coefficients = plan["a"], plan["coefficients"]
    q = coefficients[0].shape[0]
    if exact:
        x = rational_matrix(x)
    else:
        if "float_a" not in plan:
            float_a = np.asarray(a, dtype=float)
            float_coefficients = tuple(np.asarray(b, dtype=float) for b in coefficients)
            if not np.all(np.isfinite(float_a)) or any(
                    not np.all(np.isfinite(b)) for b in float_coefficients):
                raise ValueError("Floating execution requires finite representable inputs")
            for matrix in (float_a, *float_coefficients):
                matrix.setflags(write=False)
            plan["float_a"], plan["float_coefficients"] = float_a, float_coefficients
        a, coefficients = plan["float_a"], plan["float_coefficients"]
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("Floating execution requires finite representable inputs")
    if x.shape != (a.shape[0], q):
        raise ValueError(f"State must have shape {(a.shape[0], q)}")
    y = x @ coefficients[-1].T
    for b in reversed(coefficients[:-1]):
        y = a @ y + x @ b.T
    if not exact and not np.all(np.isfinite(y)):
        raise ArithmeticError("Floating execution overflowed")
    return y
