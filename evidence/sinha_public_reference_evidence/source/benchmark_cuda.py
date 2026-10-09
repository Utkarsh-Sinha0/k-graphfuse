"""Precision-constrained CPU/CUDA tests; transfer, capture and memory costs explicit."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from fractions import Fraction as F
import gc
import json
import math
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import csr_array, diags
import torch
from threadpoolctl import threadpool_limits

from c2fuse import norm_upper, sqrt_upper
from cuda_c2fuse import prepare_cuda, CudaExecutor
from sensitivity_c2fuse import compile_sensitivity, evaluate_sensitivity
from sparse_c2fuse import compile_sparse, evaluate_sparse
from chebyshev_ppr import compile_ppr, evaluate_ppr
from scipy.sparse import eye_array
from scipy.sparse.linalg import spsolve
from benchmark_upgrades import snap, normalized_symmetric, cheby_ppr


def cpu_time(fn):
    fn(); values = []
    for _ in range(5):
        start = perf_counter(); fn(); values.append(perf_counter()-start)
    return {"median_ms": 1000*median(values), "min_ms": 1000*min(values), "max_ms": 1000*max(values)}


def cuda_time(fn, events=True):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(7):
        if events:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); end.synchronize()
            values.append(start.elapsed_time(end))
        else:
            torch.cuda.synchronize(); start = perf_counter(); fn(); torch.cuda.synchronize()
            values.append(1000*(perf_counter()-start))
    return {"median_ms": median(values), "min_ms": min(values), "max_ms": max(values)}


def arrays(value):
    return value if isinstance(value, tuple) else (value,)


def nonnormal(n):
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    return csr_array((np.concatenate((np.tile([-.2, -.1], n//2), np.ones(n//2), np.full(n//2, 1/64))),
                      (np.concatenate((np.arange(n), even, odd)),
                       np.concatenate((np.arange(n), odd, (odd+1) % n)))), shape=(n, n))


def benchmark_case(label, a, coefficients, x, budget, *, direction=None, resolvent_plan=None):
    start = perf_counter()
    if resolvent_plan is not None:
        plan, full = resolvent_plan, None
        cpu = lambda: evaluate_ppr(plan, x)
        reference = spsolve(eye_array(a.shape[0], format="csr")-(1-float(plan["alpha"]))*a,
                             float(plan["alpha"])*x)
    elif direction is None:
        plan = compile_sparse(a, [coefficients], budget, bounds="fast")
        full = compile_sparse(a, [coefficients], 0, bounds="fast")
        cpu = lambda: evaluate_sparse(plan, x)
        reference = evaluate_sparse(full, x)
    else:
        plan = compile_sensitivity(a, direction, [coefficients], budget, budget,
                                   parameter_radius=F(1, 64), bounds="fast")
        full = compile_sensitivity(a, direction, [coefficients], 0, 0,
                                   parameter_radius=F(1, 64), bounds="fast")
        cpu = lambda: evaluate_sensitivity(plan, x)
        reference = evaluate_sensitivity(full, x)
    compile_seconds = perf_counter()-start
    cpu_candidates = []
    for threads in (1, 2, 4, 8):
        with threadpool_limits(limits=threads, user_api="blas"):
            cpu_candidates.append({"blas_threads": threads, "backend": "current C2Fuse CPU",
                                   "timing": cpu_time(cpu)})
            if resolvent_plan is not None:
                cpu_candidates.append({"blas_threads": threads, "backend": "ChebyPower-style CPU port",
                    "timing": cpu_time(lambda: cheby_ppr(a, x, float(plan["alpha"]), float(budget))[0])})
    best_cpu = min(cpu_candidates, key=lambda v: v["timing"]["median_ms"])
    baseline = best_cpu["timing"]
    input_norm = np.linalg.norm(x)
    reference_values = arrays(reference)
    row = {"case": label, "vertices": a.shape[0], "channels": x.shape[1], "nonzeros": int(a.nnz),
           "degree": plan["degree"], "full_degree": full["degree"] if full else None, "two_plan_compile_seconds": compile_seconds,
           "cpu_optimized": baseline, "exact_value_tail_bound": float(plan["error_bound"]),
           "cpu_candidates": cpu_candidates, "best_cpu": best_cpu,
           "candidates": [], "acceptance_budget_per_input_norm": float(budget)}
    if direction is not None:
        norms = [norm_upper(b) for b in full["coefficients"]]
        rho, eta = plan["parameter_norm_radius"], plan["direction_norm_bound"]
        classical_degree = next(m for m in range(len(norms)) if
            eta*sum(k*norms[k]*rho**(k-1) for k in range(m+1, len(norms))) <= budget)
        old = compile_sensitivity(a, direction, [coefficients], budget, budget,
                                 parameter_radius=F(1, 64), bounds="fast", constant=1+sqrt_upper(2))
        row.update({"derivative_bound": float(plan["derivative_error_bound"]),
                    "derivative_method": plan["derivative_bound_method"],
                    "optimized_block_scale": float(plan["derivative_scale"]) if plan["derivative_scale"] else None,
                    "classical_derivative_degree": classical_degree, "older_complete_constant_degree": old["degree"],
                    "parameter_interval": [-1/64, 1/64],
                    "all_norms": "Frobenius state norm; operator-norm graph direction bound"})
    for dtype in (torch.float64, torch.float32):
        modes = ((False, False), (False, True)) if resolvent_plan is not None else ((False, False), (False, True), (True, True))
        for packed, capture in modes:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            initial_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            candidate = {"dtype": str(dtype).split(".")[-1], "packed": packed, "capture": capture}
            prepared = executor = tensor = answer = None
            try:
                start = perf_counter()
                tensor = torch.tensor(x, dtype=dtype, device="cuda")
                prepared = prepare_cuda(plan, dtype=dtype, packed=packed)
                executor = CudaExecutor(prepared, tensor, capture=capture)
                torch.cuda.synchronize()
                candidate["upload_and_prepare_seconds"] = perf_counter()-start
                answer = executor(tensor)
                errors = [float(np.linalg.norm(actual.detach().cpu().numpy()-expected)/input_norm)
                          for actual, expected in zip(arrays(answer), reference_values)]
                candidate["errors_per_input_norm_against_cpu_full"] = errors.copy()
                if direction is not None:
                    for theta in (-1/64, 1/64):
                        expected = evaluate_sensitivity(full, x, theta=theta)
                        actual = executor(tensor, theta=theta)
                        errors.extend(float(np.linalg.norm(v.cpu().numpy()-w)/input_norm)
                                      for v, w in zip(actual, expected))
                    candidate["endpoint_errors_per_input_norm"] = errors[2:]
                    executor(tensor, theta=0)
                candidate["accepted_empirically"] = all(error <= float(budget) for error in errors)
                candidate["kernel_resident"] = cuda_time(executor.replay_prevalidated)
                candidate["checked_resident_api"] = cuda_time(lambda: executor(tensor), events=False)
                def round_trip():
                    incoming = torch.tensor(x, dtype=dtype, device="cuda")
                    result = executor(incoming)
                    return tuple(v.cpu().numpy() for v in arrays(result))
                candidate["host_round_trip"] = cuda_time(round_trip, events=False)
                candidate["extra_peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()-initial_bytes
                candidate["torch_reserved_bytes"] = torch.cuda.memory_reserved()
                candidate["within_2gib_torch_peak_cap"] = candidate["extra_peak_torch_allocated_bytes"] <= 2*1024**3
            except (RuntimeError, ValueError) as exc:
                candidate["error"] = str(exc)[:500]
                candidate["accepted_empirically"] = False
            finally:
                del answer, executor, prepared, tensor
                gc.collect(); torch.cuda.empty_cache()
            row["candidates"].append(candidate)
    valid = [candidate for candidate in row["candidates"] if candidate.get("accepted_empirically")
             and candidate.get("within_2gib_torch_peak_cap")]
    if valid:
        best = min(valid, key=lambda v: v["checked_resident_api"]["median_ms"])
        end_to_end = min(valid, key=lambda v: v["host_round_trip"]["median_ms"])
        row["best_valid_resident_gpu"] = {k: best[k] for k in ("dtype", "packed", "capture", "checked_resident_api", "kernel_resident", "extra_peak_torch_allocated_bytes")}
        row["best_valid_host_gpu"] = {k: end_to_end[k] for k in ("dtype", "packed", "capture", "host_round_trip")}
        row["resident_speedup_vs_cpu_optimized"] = baseline["median_ms"]/best["checked_resident_api"]["median_ms"]
        row["host_speedup_vs_cpu_optimized"] = baseline["median_ms"]/end_to_end["host_round_trip"]["median_ms"]
        row["best_observed_end_to_end_device"] = "GPU" if row["host_speedup_vs_cpu_optimized"] > 1 else "CPU"
        if direction is not None:
            prepared = prepare_cuda(full)
            tensor = torch.tensor(x, dtype=torch.float64, device="cuda")
            executor = CudaExecutor(prepared, tensor)
            row["full_degree_cuda_control"] = cuda_time(lambda: executor(tensor), events=False)
            row["cuda_truncation_speedup_same_device"] = row["full_degree_cuda_control"]["median_ms"]/best["checked_resident_api"]["median_ms"]
            del executor, prepared, tensor
            gc.collect(); torch.cuda.empty_cache()
    print(label, {k: row.get(k) for k in ("degree", "resident_speedup_vs_cpu_optimized", "host_speedup_vs_cpu_optimized", "classical_derivative_degree")}, flush=True)
    return row


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a real CUDA GPU")
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(0)
    report = {"date": "2026-10-09", "platform": platform.platform(), "numpy": np.__version__,
              "scipy": scipy.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": properties.name, "total_vram_bytes": properties.total_memory,
              "compute_capability": [properties.major, properties.minor], "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
              "timing": "CPU median of five runs, BLAS thread candidates 1/2/4/8; GPU median of seven runs; CUDA events for kernel, synchronized wall time for checked API and transfers",
              "quality": "Measured against full float64 CPU polynomial; empirical gate, not a complete rounding proof",
              "scope": "Synthetic kernel workloads and one public topology; not industrial SOTA",
              "results": []}
    rng = np.random.default_rng(20261010)
    for n, q in ((10_000, 4), (100_000, 4), (100_000, 32), (1_000_000, 8)):
        rows = np.repeat(np.arange(n), 7)
        columns = np.column_stack((np.arange(n), rng.integers(0, n, (n, 6)))).ravel()
        weights = np.column_stack((np.full(n, 0.125), rng.choice([-1., 1.], (n, 6))/64)).ravel()
        a = csr_array((weights, (rows, columns)), shape=(n, n))
        coefficients = [np.eye(q, dtype=int)] + [
            np.array([[F(int(v), 10*3**k) for v in row] for row in rng.integers(-2, 3, (q, q))], dtype=object)
            for k in range(1, 49)]
        x = rng.normal(size=(n, q))
        report["results"].append(benchmark_case(f"directed polynomial n={n} q={q}", a, coefficients, x, F(1, 10**8)))
        Path(__file__).with_name("benchmark_cuda_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    n, q = 100_000, 4
    coefficients = [np.eye(q, dtype=int)] + [
        np.array([[F(int(v), 10*3**k) for v in row] for row in rng.integers(-2, 3, (q, q))], dtype=object)
        for k in range(1, 49)]
    row = benchmark_case("robust value and derivative", nonnormal(n), coefficients,
                          rng.normal(size=(n, q)), F(1, 10**8),
                          direction=diags(np.tile([1., -1.], n//2), format="csr"))
    report["results"].append(row)
    adjacency, provenance = snap(Path("../../work/graph_stress_data"), "ca-GrQc", True)
    s, degree = normalized_symmetric(adjacency)
    start = perf_counter()
    ppr = compile_ppr(s, F(1, 20), F(1, 10**8), weights=np.sqrt(np.maximum(degree, 1)))
    ppr_compile = perf_counter()-start
    row = benchmark_case("public ca-GrQc symmetric resolvent q=16", s, None,
                          rng.normal(size=(s.shape[0], 16)), F(1, 10**8), resolvent_plan=ppr)
    row["public_data"] = provenance
    row["ppr_compile_seconds"] = ppr_compile
    report["results"].append(row)
    Path(__file__).with_name("benchmark_cuda_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__": run()
