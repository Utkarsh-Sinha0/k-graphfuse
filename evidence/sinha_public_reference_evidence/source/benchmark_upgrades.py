"""Public-graph and adversarial controls; all wins retain their comparator labels."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from fractions import Fraction as F
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import scipy
from scipy.sparse import coo_array, csr_array, diags, eye_array
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from benchmark_sparse import direct, timed
from chebyshev_ppr import compile_ppr, evaluate_ppr
from local_c2fuse import prepare_local, evaluate_local
from ppr_certificate import certify_ppr
from sparse_c2fuse import compile_sparse, evaluate_sparse


def snap(data_dir, name, undirected):
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / (name + ".txt.gz")
    subdirectory = "bigdata/communities/" if name in ("com-dblp.ungraph", "com-youtube.ungraph") else ""
    url = f"https://snap.stanford.edu/data/{subdirectory}{name}.txt.gz"
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    edges = np.loadtxt(io.BytesIO(gzip.decompress(raw)), dtype=np.int64)
    labels = np.unique(edges)
    mapped = np.searchsorted(labels, edges)
    adjacency = csr_array((np.ones(len(edges)), (mapped[:, 0], mapped[:, 1])),
                          shape=(len(labels), len(labels)))
    adjacency.sum_duplicates()
    if undirected:
        adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1
    return adjacency, {"name": name, "url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                       "vertices": len(labels), "stored_edges_after_processing": int(adjacency.nnz),
                       "processing": "Remap IDs, remove loops, binarize; symmetrize if undirected"}


def cheby_ppr(s, x, alpha, tolerance):
    """ChebyPower-style recurrence from 2412.10789, not its optimized C++ code."""
    beta = 1-alpha
    root = math.sqrt(1-beta*beta)
    ratio = beta/(1+root)
    scale = alpha/root
    degree = 0
    while 2*scale*ratio**(degree+1)/(1-ratio) > tolerance:
        degree += 1
    previous, current = x, s @ x
    y = scale*x
    if degree:
        y = y + 2*scale*ratio*current
    for k in range(2, degree+1):
        previous, current = current, 2*(s @ current)-previous
        y = y + 2*scale*ratio**k*current
    return y, degree


def normalized_symmetric(adjacency):
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    np.divide(1, np.sqrt(degree), out=inv, where=degree > 0)
    s = (diags(inv) @ adjacency @ diags(inv)).tocsr()
    # Isolated vertices keep their probability rather than silently losing it.
    s = (s+s.T)*0.5 + diags((degree == 0).astype(float))
    return s, degree


def run(data_dir):
    isolated, _ = normalized_symmetric(csr_array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]))
    np.testing.assert_array_equal(isolated.toarray(), [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    rng = np.random.default_rng(20261009)
    coefficients = [np.eye(4, dtype=int)] + [
        np.array([[F(int(v), 10*3**k) for v in row] for row in rng.integers(-2, 3, (4, 4))], dtype=object)
        for k in range(1, 49)]
    report = {"date": "2026-10-09", "platform": platform.platform(),
              "numpy": np.__version__, "scipy": scipy.__version__, "seed": 20261009,
              "setup_timing": "One complete compilation per mode; observed setup, not a median",
              "query_timing": "Seven warmed interleaved CPU runs per method; one BLAS thread",
              "comparators": "Existing C2Fuse, full sparse Horner, local full polynomial and a ChebyPower-style Python port",
              "sota_status": "No optimized recent author implementation executed; no SOTA win established",
              "public_graphs": [], "filter_queries": [], "local_queries": [], "ppr_control": []}
    n = 100_000
    rows = np.repeat(np.arange(n), 7)
    columns = np.column_stack((np.arange(n), rng.integers(0, n, (n, 6)))).ravel()
    values = np.column_stack((np.full(n, 0.125), rng.choice([-1., 1.], (n, 6))/64)).ravel()
    a = csr_array((values, (rows, columns)), shape=(n, n))
    setup = {}
    plans = {}
    for mode in ("exact", "fast"):
        start = perf_counter(); plans[mode] = compile_sparse(a, [coefficients], F(1, 10**8), bounds=mode)
        setup[mode+"_seconds"] = perf_counter()-start
    assert plans["fast"]["norm_radius"] >= plans["exact"]["norm_radius"]
    assert plans["fast"]["radius"] >= plans["exact"]["radius"]
    assert plans["fast"]["degree"] == plans["exact"]["degree"]
    setup.update({"vertices": n, "nonzeros": int(a.nnz), "degree": plans["fast"]["degree"],
                  "speedup": setup["exact_seconds"]/setup["fast_seconds"],
                  "fast_geometry": plans["fast"]["geometry_method"]})
    report["planner_setup"] = setup
    print("Planner setup measured", setup, flush=True)

    for name, undirected in (("ca-GrQc", True), ("email-Eu-core", False)):
        adjacency, provenance = snap(data_dir, name, undirected)
        report["public_graphs"].append(provenance)
        maximum = float(np.max(np.asarray(adjacency.sum(axis=1)).ravel()))
        operator = adjacency/(8*maximum) + 0.125*eye_array(adjacency.shape[0], format="csr")
        start = perf_counter(); plan = compile_sparse(operator, [coefficients], F(1, 10**8), bounds="fast")
        compile_seconds = perf_counter()-start
        x = rng.normal(size=(operator.shape[0], 4))
        full_coeffs = tuple(np.asarray(b, dtype=float) for b in coefficients)
        functions = (lambda: direct(plan["a"], full_coeffs, x), lambda: evaluate_sparse(plan, x))
        baseline, optimized = timed(functions, rng)
        error = np.linalg.norm(functions[0]()-functions[1]())/np.linalg.norm(x)
        assert error <= float(plan["error_bound"])+1e-12
        report["filter_queries"].append({"graph": name, "vertices": operator.shape[0],
            "full_degree": 48, "selected_degree": plan["degree"], "compile_seconds": compile_seconds,
            "full_sparse_horner_seconds": baseline, "optimized_seconds": optimized,
            "speedup_vs_full_sparse_horner": baseline["median"]/optimized["median"],
            "exact_truncation_bound": float(plan["error_bound"]), "observed_error_per_input_norm": float(error),
            "scope": "Artificial decaying linear filter applied to real topology; not a recommendation task"})
        count, components = connected_components(adjacency, directed=False)
        sizes = np.bincount(components)
        candidates = np.flatnonzero((sizes >= 2) & (sizes <= 20))
        if len(candidates):
            component = int(candidates[np.argmax(sizes[candidates])])
            source = int(np.flatnonzero(components == component)[0])
            local_benchmark(report, name+" selected small component", plan, coefficients, source, rng,
                            {"selected_component_size": int(sizes[component]),
                             "selection": "Deliberately selected a small nontrivial weak component"})
        if undirected:
            s, degree = normalized_symmetric(adjacency)
            alpha, budget = 0.05, 1e-8
            power_degree = math.ceil(math.log(budget)/math.log(1-alpha))-1
            polynomial = [[[alpha*(1-alpha)**k]] for k in range(power_degree+1)]
            start = perf_counter(); power = compile_sparse(s, [polynomial], F(budget), bounds="fast")
            power_setup = perf_counter()-start
            source = int(np.argmax(degree))
            x = np.zeros((s.shape[0], 1)); x[source] = 1
            reference = spsolve(eye_array(s.shape[0], format="csr")-(1-alpha)*s, alpha*x.ravel())
            start = perf_counter()
            adapted = compile_ppr(s, alpha, F(budget), weights=np.sqrt(np.maximum(degree, 1)))
            adapted_setup = perf_counter()-start
            methods = (lambda: evaluate_sparse(power, x), lambda: cheby_ppr(s, x, alpha, budget)[0],
                       lambda: evaluate_ppr(adapted, x))
            power_time, cheby_time, adapted_time = timed(methods, rng)
            errors = [float(np.linalg.norm(fn().ravel()-reference)) for fn in methods]
            assert max(errors) < 2*budget
            report["ppr_control"].append({"graph": name, "source": source, "source_degree": float(degree[source]),
                "alpha": alpha, "target": "Common symmetric-normalized resolvent; empirical L2 budget 1e-8",
                "power_degree": power["degree"], "cheby_degree": cheby_ppr(s, x, alpha, budget)[1],
                "c2_compile_seconds": power_setup, "c2_query_seconds": power_time,
                "chebypower_style_query_seconds": cheby_time, "empirical_l2_errors": errors,
                "c2_speedup_vs_chebypower_style": cheby_time["median"]/power_time["median"],
                "adapted_compile_seconds": adapted_setup, "adapted_query_seconds": adapted_time,
                "adapted_degree": adapted["degree"], "adapted_analytic_tail_bound": float(adapted["error_bound"]),
                "adapted_speedup_vs_original_c2": power_time["median"]/adapted_time["median"],
                "adapted_speedup_vs_chebypower_style": cheby_time["median"]/adapted_time["median"],
                "conclusion": "Adapted backend uses the established accelerator; no novel SOTA claim"})
        print("Public graph", name, "finished", flush=True)
    # Deliberately local synthetic workload, plus a local full-degree control.
    columns = np.arange(n)
    operator = csr_array((np.full(n, 0.25), ((columns//20)*20+(columns+1)%20, columns)), shape=(n, n))
    plan = compile_sparse(operator, [coefficients], F(1, 10**8), bounds="fast")
    local_benchmark(report, "synthetic 100k vertices in 20-node cycles", plan, coefficients, 0, rng,
                    {"selection": "Synthetic disconnected graph; favorable locality intentionally constructed"})
    # Exact general norm-transfer counterexample, without asserting the full algorithm is wrong.
    alpha, epsilon = F(1, 5), F(1, 10_000)
    p = csr_array([[0., 1.], [1., 0.]])
    estimate = np.array([float(F(5, 9)-epsilon), float(F(4, 9)+epsilon)])
    guard = certify_ppr(p, [1, 0], estimate, alpha)
    report["accuracy_guard"] = {"degree_normalized_error": float(epsilon),
        "normalized_residual_from_exact_counterexample": float((2-alpha)*epsilon/alpha),
        "residual_amplification": float((2-alpha)/alpha),
        "exact_a_posteriori_l1_bound_for_float_estimate": float(guard["l1_error_bound"]),
        "scope": "Refutes the general error-to-residual inference alone; not a refutation of ChebyPush"}
    Path(__file__).with_name("benchmark_upgrades_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def local_benchmark(report, label, plan, coefficients, source, rng, extra):
    start = perf_counter(); local = prepare_local(plan)
    index_seconds = perf_counter()-start
    shape = (plan["a"].shape[0], 4)
    x = coo_array(([1., 2.], ([source, source], [0, 3])), shape=shape)
    dense = x.toarray()
    full = prepare_local(compile_sparse(plan["a"], [coefficients], 0, bounds="fast"))
    functions = (lambda: evaluate_sparse(plan, dense), lambda: evaluate_local(local, x),
                 lambda: evaluate_local(full, x))
    global_time, local_time, full_local_time = timed(functions, rng)
    answer = functions[1]().toarray()
    np.testing.assert_allclose(answer, functions[0](), atol=1e-14, rtol=1e-14)
    full_error = np.linalg.norm(answer-functions[2]().toarray())/np.linalg.norm(dense)
    assert full_error <= float(plan["error_bound"])+1e-12
    report["local_queries"].append({"case": label, **extra, "vertices": shape[0], "source": source,
        "retained_degree": plan["degree"], "output_vertices": len(np.unique(functions[1]().row)),
        "csc_index_seconds": index_seconds, "global_retained_seconds": global_time,
        "local_retained_seconds": local_time, "local_full_polynomial_seconds": full_local_time,
        "speedup_vs_global_retained": global_time["median"]/local_time["median"],
        "speedup_vs_local_full_polynomial": full_local_time["median"]/local_time["median"],
        "scope": "Includes local discovery and sparse output; excludes one-time CSC index; not ChebyPush"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../../work/graph_stress_data"))
    run(parser.parse_args().data_dir)
