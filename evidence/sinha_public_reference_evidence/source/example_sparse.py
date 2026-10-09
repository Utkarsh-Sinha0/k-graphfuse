"""Runnable small directed graph filter; prints the bound and actual reference error."""
from fractions import Fraction as F
import numpy as np
from scipy.sparse import csr_array

from sparse_c2fuse import compile_sparse, evaluate_sparse


def main():
    a = csr_array([[0.125, 0.25, 0.0],
                   [0.0, 0.125, 0.25],
                   [0.25, 0.0, 0.125]])
    mix = np.array([[1, 1], [-1, 1]])
    coefficients = [np.eye(2, dtype=int)] + [
        [[F(int(v), 4**k) for v in row] for row in mix]
        for k in range(1, 33)]
    plan = compile_sparse(a, [coefficients], F(1, 10**8))
    full = compile_sparse(a, [coefficients], 0)
    x = np.array([[1., 0.], [0., 1.], [1., 1.]])
    y = evaluate_sparse(plan, x)
    error = np.linalg.norm(y-evaluate_sparse(full, x)) / np.linalg.norm(x)
    assert error <= float(plan["error_bound"]) + 1e-14
    print(f"Graph passes: {full['degree']} -> {plan['degree']}")
    print(f"Exact truncation bound per input norm: {float(plan['error_bound']):.12g}")
    print(f"Observed floating error per input norm: {error:.12g}")
    print(y)


if __name__ == "__main__":
    main()
