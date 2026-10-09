"""Seeded synthetic sparse CPU cases, including classical and older-bound controls."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from fractions import Fraction as F
from pathlib import Path
from statistics import median
from time import perf_counter
import json
import math
import platform

import numpy as np
import scipy
from scipy.sparse import csr_array

from c2fuse import _choose_degree, norm_upper, normalize, sqrt_upper
from sparse_c2fuse import compile_sparse, evaluate_sparse


def timed(functions, rng):
    for fn in functions:
        fn()
    samples = [[] for _ in functions]
    for _ in range(7):
        for index in rng.permutation(len(functions)):
            start = perf_counter(); functions[index]()
            samples[index].append(perf_counter() - start)
    return [{"median": median(values), "min": min(values), "max": max(values)}
            for values in samples]


def direct(a, coefficients, x):
    if not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite input")
    y = x @ coefficients[-1].T
    for coefficient in reversed(coefficients[:-1]):
        y = a @ y + x @ coefficient.T
    if not np.all(np.isfinite(y)):
        raise ArithmeticError("Nonfinite output")
    return y


def degree_controls(coefficients, plan, budget):
    coefficients = normalize(coefficients)
    new = _choose_degree(coefficients, plan["norm_radius"], plan["radius"], budget, F(2))[0]
    old = _choose_degree(coefficients, plan["norm_radius"], plan["radius"],
                         budget, 1 + sqrt_upper(2))[0]
    # Using rho as both radii makes the classical norm tail the selected bound.
    classical = _choose_degree(coefficients, plan["norm_radius"], plan["norm_radius"],
                               budget, F(2))[0]
    return new, old, classical


def run():
    rng = np.random.default_rng(20261009)
    q, degree = 4, 48
    coefficients = [np.eye(q, dtype=int)] + [
        np.array([[F(int(v), 10 * 3**k) for v in row] for row in
                  rng.integers(-2, 3, size=(q, q))], dtype=object)
        for k in range(1, degree + 1)]
    cases = []
    for n in (10_000, 100_000):
        # Every row has a diagonal and six seeded directed edges. Repeats are summed.
        rows = np.repeat(np.arange(n), 7)
        columns = np.column_stack((np.arange(n), rng.integers(0, n, (n, 6)))).ravel()
        weights = np.column_stack((np.full(n, 0.125),
                                   rng.choice([-1., 1.], size=(n, 6))/64)).ravel()
        cases.append(("directed sparse graph", csr_array((weights, (rows, columns)),
                                                         shape=(n, n)), coefficients, F(1, 10**8)))
    n = 100_000
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    rows = np.concatenate((np.arange(n), even, odd))
    columns = np.concatenate((np.arange(n), odd, (odd+1) % n))
    weights = np.concatenate((np.tile([-0.2, -0.1], n//2),
                              np.ones(n//2), np.full(n//2, 1/64)))
    coupled = csr_array((weights, (rows, columns)), shape=(n, n))
    cases.append(("coupled nonnormal directed graph", coupled, coefficients, F(1, 10**8)))
    diagonal = np.arange(10_000)
    cases.append(("no removable tail", csr_array((np.ones(10_000), (diagonal, diagonal)),
                                                shape=(10_000, 10_000)),
                  [np.eye(q, dtype=int)] * 17, F(0)))
    results = []
    for label, a, coeffs, budget in cases:
        n = a.shape[0]
        x = rng.normal(size=(n, q))
        start = perf_counter(); plan = compile_sparse(a, [coeffs], budget)
        setup_seconds = perf_counter() - start
        full_coeffs = tuple(np.asarray(b, dtype=float) for b in coeffs)
        new_degree, old_degree, classical_degree = degree_controls(coeffs, plan, budget)
        assert new_degree == plan["degree"]
        full = lambda: direct(plan["a"], full_coeffs, x)
        optimized = lambda: evaluate_sparse(plan, x)
        classic = lambda: direct(plan["a"], full_coeffs[:classical_degree + 1], x)
        full_time, optimized_time, classic_time = timed((full, optimized, classic), rng)
        error = np.linalg.norm(full() - optimized()) / np.linalg.norm(x)
        bound = float(plan["error_bound"])
        assert error <= bound + 1e-12, (label, error, bound)
        saving = full_time["median"] - optimized_time["median"]
        a_bytes = sum(array.nbytes for array in (plan["a"].data,
                                                plan["a"].indices, plan["a"].indptr))
        results.append({
            "case": label, "vertices": n, "stored_nonzeros": int(plan["a"].nnz),
            "channels": q, "full_degree": len(coeffs)-1, "selected_degree": new_degree,
            "older_complete_bound_degree": old_degree, "classical_norm_degree": classical_degree,
            "bound_method": plan["bound_method"], "budget": float(budget),
            "exact_truncation_bound": bound, "observed_error_per_input_norm": float(error),
            "compile_seconds": setup_seconds,
            "full_sparse_horner_seconds": full_time,
            "optimized_sparse_horner_seconds": optimized_time,
            "classical_cut_sparse_horner_seconds": classic_time,
            "speedup_vs_full_sparse_horner": full_time["median"] / optimized_time["median"],
            "speedup_vs_classical_cut": classic_time["median"] / optimized_time["median"],
            "break_even_repeated_calls": math.ceil(setup_seconds/saving) if saving > 0 else None,
            "stored_csr_operator_bytes": a_bytes,
            "dense_float64_operator_bytes_calculated_not_allocated": 8*n*n,
            "input_state_bytes": x.nbytes})
        print(f"{label}: n={n}, degree={new_degree}, speedup={results[-1]['speedup_vs_full_sparse_horner']:.3f}", flush=True)
    # Budget intentionally crosses one degree boundary; it is not representative traffic.
    probe = compile_sparse(coupled[:32, :32], [coefficients], 0)
    budget = F(11, 5) * sum(norm_upper(b) * probe["radius"]**k
                           for k, b in enumerate(coefficients) if k > 8)
    new, old, classical = degree_controls(coefficients, probe, budget)
    assert new < old
    result = {
        "date": "2026-10-09", "platform": platform.platform(),
        "processor": platform.processor(), "numpy": np.__version__, "scipy": scipy.__version__,
        "seed": 20261009, "timing": "Seven warmed CPU runs per method, interleaved in seeded random order; median/min/max; one BLAS thread",
        "scope": "Synthetic kernels; full and classical-cut sparse Horner controls; no industrial/SOTA claim",
        "certificate": "Exact-arithmetic truncation; floating conversion and rounding excluded",
        "memory": "CSR buffers measured; hypothetical dense bytes calculated; no peak-RSS measurement",
        "constant_only_boundary": {"new_degree": new, "older_complete_bound_degree": old,
                                   "classical_norm_degree": classical,
                                   "deliberately_selected_budget": float(budget)},
        "results": results}
    Path(__file__).with_name("benchmark_sparse_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
