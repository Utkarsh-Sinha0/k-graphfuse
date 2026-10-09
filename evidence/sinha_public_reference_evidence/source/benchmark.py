"""Reproducible CPU cases; reference kernels have identical data/layout."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from fractions import Fraction as F
from pathlib import Path
from statistics import median
from time import perf_counter
import json
import platform
import numpy as np
from c2fuse import compile_pipeline, evaluate, norm_upper, sqrt_upper


def elapsed(fn):
    fn(); samples = []
    for _ in range(7):
        start = perf_counter(); fn(); samples.append(perf_counter() - start)
    return median(samples)


def direct(a, coefficients, x):
    y = x @ coefficients[-1].T
    for b in reversed(coefficients[:-1]):
        y = a @ y + x @ b.T
    return y


def run():
    rng = np.random.default_rng(20261009)
    rows = []
    q = 8
    coeffs = [np.eye(q, dtype=int)] + [
        np.array([[F(int(v), 10 * 3**k) for v in row] for row in
                  rng.integers(-2, 3, size=(q, q))], dtype=object)
        for k in range(1, 49)]
    float_coeffs = [np.asarray(b, dtype=float) for b in coeffs]
    for n in (64, 256, 512):
        a = np.zeros((n, n));
        for k in range(0, n, 2):
            a[k:k+2, k:k+2] = [[-0.2, 1.0], [0.0, -0.1]]
        x = rng.normal(size=(n, q))
        start = perf_counter(); plan = compile_pipeline(a, [coeffs], F(1, 10**8))
        evaluate(plan, x)
        compile_seconds = perf_counter() - start
        old = compile_pipeline(a, [coeffs], F(1, 10**8), constant=1+sqrt_upper(2))
        baseline = lambda: direct(a, float_coeffs, x)
        improved = lambda: evaluate(plan, x)
        reference_seconds = elapsed(baseline)
        execution_seconds = elapsed(improved)
        relative_error = np.linalg.norm(baseline()-improved()) / np.linalg.norm(x)
        bound = float(plan['error_bound'])
        assert relative_error <= bound + 1e-12
        rows.append({"case": "48-degree block polynomial", "n": n, "channels": q,
                     "original_degree": 48, "selected_degree": plan['degree'],
                     "pre_2026_constant_degree": old['degree'],
                     "bound_method": plan['bound_method'], "error_per_input_norm": relative_error,
                     "exact_arithmetic_error_bound": bound,
                     "compile_seconds": compile_seconds, "reference_seconds": reference_seconds,
                     "execution_seconds": execution_seconds,
                     "execution_speedup_vs_full_degree": reference_seconds / execution_seconds,
                     "amortization_calls": (int(np.ceil(compile_seconds /
                          (reference_seconds-execution_seconds)))
                          if reference_seconds > execution_seconds else None)})
    n = 256
    a = rng.normal(size=(n, n)) / n
    shear = np.zeros((q, q), dtype=int); shear[0, 7] = 1
    identity = np.eye(q, dtype=int)
    stages = [[identity, shear], [identity, -shear]] * 16
    float_stages = [[np.asarray(b, dtype=float) for b in stage] for stage in stages]
    x = rng.normal(size=(n, q))
    start = perf_counter(); plan = compile_pipeline(a, stages)
    evaluate(plan, x)
    compile_seconds = perf_counter()-start
    def chain():
        y = x.copy()
        for stage in float_stages: y = direct(a, stage, y)
        return y
    chain_seconds = elapsed(chain); fused_seconds = elapsed(lambda: evaluate(plan, x))
    assert np.linalg.norm(chain()-evaluate(plan,x)) / np.linalg.norm(x) < 1e-12
    rows.append({"case":"32 reversible shear stages", "n":n,"channels":q,
                 "original_degree":32,"selected_degree":plan['degree'],
                 "exact_arithmetic_error_bound":0,"compile_seconds":compile_seconds,
                 "reference_seconds":chain_seconds,"execution_seconds":fused_seconds,
                 "execution_speedup_vs_unfused":chain_seconds/fused_seconds,
                 "amortization_calls":(int(np.ceil(compile_seconds/(chain_seconds-fused_seconds)))
                                       if chain_seconds > fused_seconds else None),
                 "gain_attribution":"Exact algebraic fusion; does not require a new theorem"})
    # A deliberately chosen boundary demonstrates the effect of the new constant alone.
    small = [[F(-1,5),F(1)],[F(0),F(-1,10)]]
    probe = compile_pipeline(small,[coeffs])
    budget = F(11,5)*sum(norm_upper(b)*probe['radius']**k
                        for k,b in enumerate(coeffs) if k>8)
    new = compile_pipeline(small,[coeffs],budget)
    old = compile_pipeline(small,[coeffs],budget,constant=1+sqrt_upper(2))
    assert new['degree'] < old['degree']
    result = {"hardware":platform.platform(),"processor":platform.processor(),
              "numpy":np.__version__,"timing":"Median of 7 warmed CPU runs; BLAS thread environment set to 1",
              "rounding":"Empirical error check only; certificate excludes floating roundoff",
              "baseline_scope":"Full-degree Horner and unfused stage execution; not industry SOTA",
              "constant_only_boundary_case":{"new_degree":new['degree'],"old_degree":old['degree'],
                   "chosen_budget":float(budget),"construction":"Budget selected at a degree boundary; not representative"},
              "results":rows}
    Path('benchmark_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': run()
