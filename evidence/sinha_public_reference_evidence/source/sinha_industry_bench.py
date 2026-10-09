"""Sinha GraphFuse: reproducible large-graph, same-machine GPU comparisons."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
from fractions import Fraction as F
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import threading
from time import perf_counter, sleep
from datetime import datetime, timezone
import traceback
import unittest
import sys

import numpy as np
import scipy
from scipy.sparse import csr_array, diags, eye_array
from scipy.sparse.linalg import cg
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import psutil
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr
import pylibcugraph as plc

from sparse_c2fuse import compile_sparse, evaluate_sparse
from sensitivity_c2fuse import compile_sensitivity, evaluate_sensitivity
from cuda_c2fuse import prepare_cuda, CudaExecutor
from chebyshev_ppr import compile_ppr, evaluate_ppr
from benchmark_upgrades import snap, normalized_symmetric
from benchmark_cuda import nonnormal
from c2fuse import sqrt_upper

SEED = 20261009
SCALE_GPU = "--scale-gpu" in sys.argv
OUTPUT = Path(os.environ.get("SINHA_OUTPUT", "/kaggle/working/sinha_results"))
DATA = OUTPUT.parent / "sinha_data"
REPORT = {"project": "Sinha GraphFuse", "schema_version": 1, "seed": SEED,
          "started_utc": datetime.now(timezone.utc).isoformat(), "cases": [], "failures": [],
          "scope": "Finite polynomial feature action, affine-parameter sensitivity, and symmetric personalized PageRank",
          "precision_policy": "Only accuracy-accepted candidates enter speedup comparisons; all rejected candidates retained",
          "latency_policy": "Synchronized wall time; 31 GPU or 9 CPU samples; separate preparation, first call and host round trip",
          "limits": ["Single GPU; no service load, availability or distributed-system resilience test",
                     "Numeric references and residual audits are floating computations, not complete roundoff proofs",
                     "Public topologies with constructed filters/features; no downstream fraud or recommendation quality claim",
                     "Published external numbers use different hardware and contracts; not direct speedup denominators"]}

def save():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "sinha_results.json").write_text(json.dumps(REPORT, indent=2, allow_nan=False))

def digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def digest_sparse(a):
    h = hashlib.sha256(str(a.shape).encode())
    for array in (a.indptr, a.indices, a.data):
        h.update(np.ascontiguousarray(array).view(np.uint8))
    return h.hexdigest()

def cleanup():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()

class Monitor:
    """Sample total process RSS and device use; allocator peaks recorded separately."""
    def __enter__(self):
        self.samples, self.stop_event = [], threading.Event()
        self.nvml = self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml, self.handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass
        def sample():
            while not self.stop_event.is_set():
                row = {"t": perf_counter(), "rss": psutil.Process().memory_info().rss}
                if self.nvml is not None:
                    try:
                        row["device_used_bytes"] = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
                        row["board_power_w"] = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
                    except Exception:
                        pass
                self.samples.append(row)
                self.stop_event.wait(.02)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
    def result(self):
        powers = [s for s in self.samples if "board_power_w" in s]
        return {"rss_peak_sampled_bytes": max((s["rss"] for s in self.samples), default=0),
                "device_peak_sampled_bytes": max((s.get("device_used_bytes", 0) for s in self.samples), default=0),
                "power_sample_count": len(powers),
                "board_power_sample_mean_w": float(np.mean([s["board_power_w"] for s in powers])) if powers else None,
                "sampling_interval_seconds": .02,
                "scope": "Total process/device samples during preparation and calls; short peaks may be missed"}

def timing(fn, count, gpu):
    fn()
    samples = []
    for _ in range(count):
        if gpu: torch.cuda.synchronize()
        start = perf_counter()
        value = fn()
        if gpu: torch.cuda.synchronize()
        samples.append(1000 * (perf_counter()-start))
        del value
    return {"samples_ms": samples, "median_ms": float(np.median(samples)),
            "p95_ms": float(np.quantile(samples, .95)), "min_ms": min(samples),
            "max_ms": max(samples), "sample_count": count,
            "batch_queries_per_second": 1000 / float(np.median(samples))}

def host_arrays(value):
    items = value if isinstance(value, tuple) else (value,)
    return tuple(v.detach().cpu().numpy() if isinstance(v, torch.Tensor)
                 else cp.asnumpy(v) if isinstance(v, cp.ndarray) else np.asarray(v) for v in items)

def quality(values, reference, xnorm):
    errors = [float(np.linalg.norm(a-b)/xnorm) for a, b in zip(values, reference)]
    relative = [float(np.linalg.norm(a-b)/max(np.linalg.norm(b), np.finfo(float).tiny))
                for a, b in zip(values, reference)]
    return {"error_per_input_frobenius_norm": errors, "relative_output_errors": relative,
            "finite": all(np.isfinite(a).all() for a in values)}

def cpu_horner(a, coefficients, x):
    y = x @ coefficients[-1].T
    for b in reversed(coefficients[:-1]):
        y = a @ y + x @ b.T
    return y

def cupy_horner(a, coefficients, x):
    y = x @ coefficients[-1].T
    for b in reversed(coefficients[:-1]):
        y = a @ y + x @ b.T
    return y

def torch_factory(plan, x, dtype, packed, capture, theta=0):
    tensor = torch.tensor(x, device="cuda", dtype=dtype)
    prepared = prepare_cuda(plan, dtype=dtype, packed=packed)
    executor = CudaExecutor(prepared, tensor, capture=capture)
    def resident(): return executor(tensor, theta=theta)
    def round_trip():
        incoming = torch.tensor(x, device="cuda", dtype=dtype)
        return tuple(v.cpu().numpy() for v in (lambda v: v if isinstance(v, tuple) else (v,))(executor(incoming, theta=theta)))
    return resident, round_trip, {"executor": executor}

def cupy_factory(a, coefficients, x, dtype):
    matrix = cp_csr((cp.asarray(a.data, dtype=dtype), cp.asarray(a.indices, dtype=cp.int32),
                     cp.asarray(a.indptr, dtype=cp.int32)), shape=a.shape)
    tensor = cp.asarray(x, dtype=dtype)
    bs = [cp.asarray(b.T, dtype=dtype) for b in coefficients]
    # Store the original orientation for the ordinary full/truncated Horner control.
    bs = [b.T for b in bs]
    def resident(): return cupy_horner(matrix, bs, tensor)
    def round_trip(): return cp.asnumpy(cupy_horner(matrix, bs, cp.asarray(x, dtype=dtype)))
    return resident, round_trip, {}

def cupy_ppr_factory(plan, x, dtype, capture):
    a = plan["s"]
    matrix = cp_csr((cp.asarray(a.data, dtype=dtype), cp.asarray(a.indices, dtype=cp.int32),
                     cp.asarray(a.indptr, dtype=cp.int32)), shape=a.shape)
    tensor = cp.asarray(x, dtype=dtype)
    def kernel(incoming):
        y = plan["scale"]*incoming
        if plan["degree"]:
            previous, current = incoming, (matrix @ incoming)/float(plan["radius"])
            y = y+2*plan["scale"]*plan["ratio"]*current
            for k in range(2, plan["degree"]+1):
                previous, current = current, 2*(matrix @ current)/float(plan["radius"])-previous
                y = y+2*plan["scale"]*plan["ratio"]**k*current
        return y
    if capture:
        torch.cuda.synchronize()
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            for _ in range(3): kernel(tensor)
        stream.synchronize()
        with stream:
            stream.begin_capture()
            output = kernel(tensor)
            graph = stream.end_capture()
        def resident():
            graph.launch()
            return output.copy()
        def host():
            tensor.set(x)
            return cp.asnumpy(resident())
        return resident, host, {"graph": graph, "stream": stream}
    def resident(): return kernel(tensor)
    def host(): return cp.asnumpy(kernel(cp.asarray(x, dtype=dtype)))
    return resident, host, {}

def measure_candidate(label, factory, reference, xnorm, budget, gpu=True, threads=None, check_extra=None):
    cleanup()
    start_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = {"method": label, "gpu": gpu, "budget": budget, "cpu_blas_threads": threads}
    fn = host = state = value = None
    try:
        with Monitor() as monitor, threadpool_limits(limits=threads or 1, user_api="blas"):
            start = perf_counter()
            fn, host, state = factory()
            if gpu: torch.cuda.synchronize()
            result["prepare_seconds"] = perf_counter()-start
            start = perf_counter()
            value = fn()
            if gpu: torch.cuda.synchronize()
            result["first_call_ms"] = 1000*(perf_counter()-start)
            result["quality"] = quality(host_arrays(value), reference, xnorm)
            accepted = result["quality"]["finite"] and max(result["quality"]["error_per_input_frobenius_norm"]) <= budget
            del value
            value = None
            if check_extra is not None:
                result["additional_quality"] = check_extra(fn, state)
                accepted = accepted and result["additional_quality"]["accepted"]
            result["accepted"] = bool(accepted)
            result["resident"] = timing(fn, 31 if gpu else 9, gpu)
            if host is not None: result["host_round_trip"] = timing(host, 11, gpu)
            result["torch_extra_peak_allocated_bytes"] = max(0, torch.cuda.max_memory_allocated()-start_alloc)
        result["resources"] = monitor.result()
    except Exception as exc:
        result.update({"accepted": False, "error": type(exc).__name__ + ": " + str(exc),
                       "traceback": traceback.format_exc()})
    finally:
        del fn, host, state, value
        cleanup()
    print("CANDIDATE", label, result.get("accepted"), result.get("resident", {}).get("median_ms"), flush=True)
    return result

def polynomial_case(name, a, q, decay=3, sensitivity=False, error_budget=1e-8):
    rng = np.random.default_rng(SEED + a.shape[0] + q + int(sensitivity))
    x = rng.normal(size=(a.shape[0], q))
    coefficients = [np.eye(q, dtype=int)] + [np.array(
        [[F(int(v), 10*decay**k) if decay > 1 else F(int(v), 490)
          for v in row] for row in rng.integers(-2, 3, (q, q))], dtype=object) for k in range(1, 49)]
    budget = F(str(error_budget))
    direction = diags(np.where(np.arange(a.shape[0]) % 2, -1., 1.), format="csr") if sensitivity else None
    start = perf_counter()
    if sensitivity:
        plan = compile_sensitivity(a, direction, [coefficients], budget, budget, parameter_radius=F(1, 64))
    else:
        plan = compile_sparse(a, [coefficients], budget, bounds="fast")
    setup = perf_counter()-start
    prior_start = perf_counter()
    if sensitivity:
        prior = compile_sensitivity(a, direction, [coefficients], budget, budget,
                                    parameter_radius=F(1, 64), constant=1+sqrt_upper(2))
    else:
        prior = compile_sparse(a, [coefficients], budget, bounds="fast", constant=1+sqrt_upper(2))
    prior_compile_seconds = perf_counter()-prior_start
    full_bs = [np.asarray(b, dtype=float) for b in coefficients]
    start = perf_counter()
    if sensitivity:
        full = compile_sensitivity(a, direction, [coefficients], 0, 0, parameter_radius=F(1, 64))
        reference = evaluate_sensitivity(full, x)
    else:
        full = compile_sparse(a, [coefficients], 0, bounds="fast")
        reference = cpu_horner(a, full_bs, x)
    reference_time = perf_counter()-start
    references = reference if isinstance(reference, tuple) else (reference,)
    row = {"case": name, "kind": "sensitivity" if sensitivity else "polynomial",
           "vertices": a.shape[0], "nonzeros": int(a.nnz), "features": q, "degree": plan["degree"],
           "full_degree": 48, "coefficient_decay_base": decay,
           "operator_sha256": digest_sparse(a), "input_sha256": digest_array(x),
           "compile_seconds": setup, "reference_prepare_seconds": reference_time,
           "older_complete_constant_degree": prior["degree"],
           "older_complete_constant_compile_seconds": prior_compile_seconds,
           "value_truncation_bound": float(plan["error_bound"]), "candidates": [], "budget": float(budget),
           "reference": "Full-degree SciPy/NumPy float64 evaluation", "status": "running"}
    if sensitivity:
        row.update({"derivative_truncation_bound": float(plan["derivative_error_bound"]),
                    "parameter_interval": [-1/64, 1/64]})
    REPORT["cases"].append(row)
    save()
    for threads in (() if SCALE_GPU else (1, 2, 4)):
        for retained in (False, True):
            p = plan if retained else full
            function = evaluate_sensitivity if sensitivity else evaluate_sparse
            label = "SciPy " + ("retained" if retained else "full") + " FP64"
            if retained and plan["degree"] == full["degree"]:
                assert all(np.array_equal(b, c) for b, c in zip(plan["float_coefficients"], full["float_coefficients"]))
                assert digest_sparse(plan["a"]) == digest_sparse(full["a"])
                alias = copy.deepcopy(row["candidates"][-1])
                alias["method"] = label
                alias["shared_timing_from"] = "SciPy full FP64 at the same thread setting"
                alias["shared_timing_reason"] = "Same degree, copied operator bytes and coefficient bytes; evaluator ignores certificate metadata"
                row["candidates"].append(alias)
                save()
                continue
            row["candidates"].append(measure_candidate(label, lambda p=p: (lambda: function(p, x), None, {}),
                references, np.linalg.norm(x), float(budget), gpu=False, threads=threads))
            save()
    specifications = []
    vram = torch.cuda.get_device_properties(0).total_memory
    for mode in ("full", "sinha", "prior"):
        p = {"full": full, "sinha": plan, "prior": prior}[mode]
        for dtype in (torch.float64, torch.float32):
            for packed, capture in ((False, False), (True, False), (False, True), (True, True)) if SCALE_GPU else ((False, False), (False, True), (True, True)):
                size = x.size * (8 if dtype == torch.float64 else 4)
                estimated = size * (p["degree"]+1) * (3 if packed else 2) if capture else size*5
                prefix = {"sinha": "Sinha GraphFuse ", "full": "PyTorch full-degree control ",
                          "prior": "Optimized older complete-bound control "}[mode]
                label = prefix + f"{str(dtype).split('.')[-1]} packed={packed} capture={capture}"
                if estimated > .65*vram:
                    row["candidates"].append({"method": label, "accepted": False,
                        "not_run": "Static capture/workspace estimate exceeds 65% of device VRAM",
                        "estimated_workspace_bytes": estimated})
                    continue
                specifications.append((label, p, dtype, packed, capture))
    rng.shuffle(specifications)
    endpoint_references = {theta: evaluate_sensitivity(full, x, theta=theta) for theta in (-1/64, 1/64)} if sensitivity else {}
    for label, p, dtype, packed, capture in specifications:
        extra = None
        if sensitivity:
            def extra(fn, state):
                executor = state["executor"]
                tensor = torch.tensor(x, dtype=executor.prepared["dtype"], device="cuda")
                errors = []
                for theta in (-1/64, 1/64):
                    expected = endpoint_references[theta]
                    errors.extend(quality(host_arrays(executor(tensor, theta=theta)), expected, np.linalg.norm(x))["error_per_input_frobenius_norm"])
                executor(tensor, theta=0)
                return {"endpoint_errors": errors, "accepted": max(errors) <= float(budget)}
        row["candidates"].append(measure_candidate(label,
            lambda p=p, dtype=dtype, packed=packed, capture=capture: torch_factory(p, x, dtype, packed, capture),
            references, np.linalg.norm(x), float(budget), check_extra=extra))
        save()
    if not sensitivity:
        for retained in (False, True):
            bs = full_bs[:plan["degree"]+1] if retained else full_bs
            for dtype in (cp.float64, cp.float32):
                label = f"CuPy/cuSPARSE {'retained-degree' if retained else 'full-degree'} {np.dtype(dtype).name}"
                row["candidates"].append(measure_candidate(label, lambda bs=bs, dtype=dtype: cupy_factory(a, bs, x, dtype),
                    references, np.linalg.norm(x), float(budget)))
                save()
    row["status"] = "complete"
    save()
    print("CASE COMPLETE", name, flush=True)

def ppr_case(name, adjacency, s, degrees, q, alpha, goal):
    rng = np.random.default_rng(SEED + adjacency.shape[0] + q)
    n = s.shape[0]
    sources = [int(np.argmax(degrees))] + rng.choice(n, size=q-1, replace=False).tolist()
    isolated_count = int(np.count_nonzero(degrees == 0))
    if isolated_count:
        adjacency = (adjacency + diags((degrees == 0).astype(float))).tocsr()
        degrees = np.maximum(degrees, 1)
    root_degree = np.sqrt(degrees)
    assert np.all(root_degree > 0)
    p = np.zeros((n, q))
    p[sources, np.arange(q)] = 1
    x = p / root_degree[:, None]
    norm = np.linalg.norm(x)
    # Cauchy-Schwarz converts the uniform symmetric L2 tail to a per-query L1 target.
    operator_budget = goal / (4*np.linalg.norm(root_degree)*norm)
    start = perf_counter()
    plan = compile_ppr(s, F(str(alpha)), F(float(operator_budget)), weights=root_degree)
    setup = perf_counter()-start
    matrix = eye_array(n, format="csr")-(1-alpha)*s
    def cg_cpu():
        answers = []
        for column in range(q):
            value, info = cg(matrix, alpha*x[:, column], rtol=1e-13, atol=0, maxiter=5000)
            if info: raise RuntimeError(f"SciPy CG did not converge: {info}")
            answers.append(value)
        return np.column_stack(answers)
    start = perf_counter()
    reference = cg_cpu()
    reference_seconds = perf_counter()-start
    reference_pr = root_degree[:, None]*reference
    transition = adjacency @ diags(1/degrees)
    def ppr_quality(symmetric):
        pr = root_degree[:, None]*symmetric
        residual = alpha*p + (1-alpha)*(transition @ pr)-pr
        errors = np.sum(np.abs(pr-reference_pr), axis=0)
        residual_bounds = np.sum(np.abs(residual), axis=0)/alpha
        k = min(100, n)
        overlaps = [len(set(np.argpartition(pr[:, i], -k)[-k:]) &
                        set(np.argpartition(reference_pr[:, i], -k)[-k:]))/k for i in range(q)]
        return {"l1_errors_against_cg": errors.tolist(), "floating_l1_residual_over_alpha": residual_bounds.tolist(),
                "top100_overlap": overlaps, "mass_errors": np.abs(pr.sum(axis=0)-1).tolist(),
                "minimum_score": float(pr.min()),
                "accepted": bool(np.isfinite(pr).all() and max(errors) <= goal and max(residual_bounds) <= goal),
                "scope": "Floating achieved-error audit; top-k ties can differ"}
    row = {"case": name, "kind": "personalized_pagerank", "vertices": n, "nonzeros": int(s.nnz),
           "features": q, "sources": sources, "teleport_probability": alpha, "l1_goal": goal,
           "isolates_converted_to_self_loops": isolated_count,
           "degree": plan["degree"], "compile_seconds": setup, "reference_prepare_seconds": reference_seconds,
           "symmetric_operator_truncation_bound": float(plan["error_bound"]),
           "l1_truncation_bound_after_similarity": float(plan["error_bound"])*np.linalg.norm(root_degree)*norm,
           "operator_sha256": digest_sparse(s), "input_sha256": digest_array(x),
           "reference": "Independent SciPy conjugate-gradient solve, rtol=1e-13",
           "reference_quality": ppr_quality(reference), "candidates": [], "status": "running"}
    assert row["reference_quality"]["accepted"]
    REPORT["cases"].append(row)
    save()
    def audit(fn, state):
        return ppr_quality(host_arrays(fn())[0])
    for threads in (1, 2, 4):
        row["candidates"].append(measure_candidate("SciPy CG FP64", lambda: (cg_cpu, None, {}),
            (reference,), norm, 1., gpu=False, threads=threads, check_extra=audit))
        row["candidates"].append(measure_candidate("SciPy Chebyshev FP64", lambda: (lambda: evaluate_ppr(plan, x), None, {}),
            (reference,), norm, 1., gpu=False, threads=threads, check_extra=audit))
        save()
    for dtype in (torch.float64, torch.float32):
        for capture in (False, True):
            row["candidates"].append(measure_candidate(f"Sinha GraphFuse Chebyshev {str(dtype).split('.')[-1]} capture={capture}",
                lambda dtype=dtype, capture=capture: torch_factory(plan, x, dtype, False, capture),
                (reference,), norm, 1., check_extra=audit))
            save()
            row["candidates"].append(measure_candidate(f"Optimized established Chebyshev control {str(dtype).split('.')[-1]} capture={capture}",
                lambda dtype=dtype, capture=capture: torch_factory(plan, x, dtype, False, capture),
                (reference,), norm, 1., check_extra=audit))
            save()
    for dtype in (cp.float64, cp.float32):
        for capture in (False, True):
            row["candidates"].append(measure_candidate(
                f"CuPy/cuSPARSE Chebyshev {np.dtype(dtype).name} capture={capture}",
                lambda dtype=dtype, capture=capture: cupy_ppr_factory(plan, x, dtype, capture),
                (reference,), norm, 1., check_extra=audit))
            save()
        def vendor_factory(dtype=dtype):
            coo = adjacency.tocoo()
            src, dst = cp.asarray(coo.row, dtype=cp.int32), cp.asarray(coo.col, dtype=cp.int32)
            weights = cp.ones(adjacency.nnz, dtype=dtype)
            handle = plc.ResourceHandle()
            graph = plc.SGGraph(handle, plc.GraphProperties(is_symmetric=True, is_multigraph=False),
                                src, dst, weight_array=weights, store_transposed=True,
                                renumber=False, do_expensive_check=True)
            vertices = cp.arange(n, dtype=cp.int32)
            sums = cp.asarray(degrees, dtype=dtype)
            seeds = [cp.asarray([source], dtype=cp.int32) for source in sources]
            personalization = cp.ones(1, dtype=dtype)
            roots = cp.asarray(root_degree)
            def resident():
                answers = []
                for seed in seeds:
                    ids, values = plc.personalized_pagerank(
                        handle, graph, vertices, sums, None, None, seed, personalization,
                        alpha=1-alpha, epsilon=alpha*goal/10,
                        max_iterations=1500, do_expensive_check=False)
                    ordered = cp.empty(n, dtype=values.dtype)
                    ordered[ids] = values
                    answers.append(ordered/roots)
                return cp.column_stack(answers)
            def host(): return cp.asnumpy(resident())
            return resident, host, {"graph": graph, "handle": handle}
        row["candidates"].append(measure_candidate(f"NVIDIA pylibcugraph personalized PageRank {np.dtype(dtype).name}",
            vendor_factory, (reference,), norm, 1., check_extra=audit))
        save()
    row["status"] = "complete"
    save()
    print("CASE COMPLETE", name, flush=True)

def selfcheck():
    sample = np.array([1., 2., 3.])
    assert digest_array(sample) == digest_array(sample.copy())
    assert quality((sample,), (sample,), 1)["error_per_input_frobenius_norm"] == [0.]
    a = csr_array([[0., 1.], [1., 0.]])
    x = np.array([[1.], [0.]])
    bs = [np.eye(1), np.eye(1)]
    np.testing.assert_array_equal(cpu_horner(a, bs, x), [[1.], [1.]])
    np.testing.assert_allclose(cp.asnumpy(cupy_horner(cp_csr(a), [cp.asarray(b) for b in bs], cp.asarray(x))), [[1.], [1.]])
    graph = plc.SGGraph(plc.ResourceHandle(), plc.GraphProperties(is_symmetric=True, is_multigraph=False),
                        cp.asarray([0, 1], dtype=cp.int32), cp.asarray([1, 0], dtype=cp.int32),
                        weight_array=cp.ones(2, dtype=cp.float64), store_transposed=True,
                        renumber=False, do_expensive_check=True)
    ids, values = plc.personalized_pagerank(plc.ResourceHandle(), graph, None, None, None, None,
        cp.asarray([0], dtype=cp.int32), cp.ones(1, dtype=cp.float64), alpha=.8,
        epsilon=1e-12, max_iterations=1000, do_expensive_check=False)
    answer = np.empty(2)
    answer[cp.asnumpy(ids)] = cp.asnumpy(values)
    np.testing.assert_allclose(answer, [1/1.8, .8/1.8], atol=1e-11)
    print("HARNESS SELFCHECK PASSED", flush=True)

def main(smoke=False, community=False, scale_gpu=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    REPORT["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "cupy": cp.__version__, "pylibcugraph": importlib.metadata.version("pylibcugraph-cu12"),
        "cuda_build": torch.version.cuda, "gpus": [{"name": torch.cuda.get_device_name(i),
        "vram_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
        "active_device": 0, "cpu_count": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
        "blas": threadpool_info(), "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout}
    REPORT["source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")}
    save()
    selfcheck()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT["unit_tests"] = {"run": tests.testsRun, "failures": len(tests.failures),
                            "errors": len(tests.errors), "skipped": len(tests.skipped)}
    assert tests.wasSuccessful() and not tests.skipped
    save()
    REPORT["suite"] = "scale_gpu_recovery" if scale_gpu else "community_recovery" if community else "smoke" if smoke else "large"
    if scale_gpu:
        REPORT["latency_policy"] = "31 synchronized GPU samples; 11 transfer samples; CPU reference evaluated but no CPU timing suite rerun"
        REPORT["scope"] += "; scale recovery also checks packed execution without capture"
    datasets = [] if scale_gpu else [("com-dblp.ungraph", True), ("com-youtube.ungraph", True)] if community else [("ca-GrQc", True)] if smoke else [("ca-GrQc", True), ("com-dblp.ungraph", True),
               ("com-youtube.ungraph", True), ("roadNet-CA", True), ("web-Google", False)]
    REPORT["dataset_provenance"] = []
    for name, undirected in datasets:
        try:
            start = perf_counter()
            adjacency, provenance = snap(DATA, name, undirected)
            provenance["download_and_preprocess_seconds"] = perf_counter()-start
            REPORT["dataset_provenance"].append(provenance)
            save()
            if undirected:
                s, degrees = normalized_symmetric(adjacency)
                q = 4 if name == "ca-GrQc" else 8 if name in ("roadNet-CA", "com-youtube.ungraph") else 32
                polynomial_case(name+" normalized symmetric polynomial", s, q)
                if not smoke and name == "com-dblp.ungraph":
                    for error_budget in (1e-4, 1e-10):
                        polynomial_case(name+f" precision sweep budget={error_budget}", s, q,
                                        error_budget=error_budget)
                for alpha, goal in ((.2, 1e-5), (.05, 1e-8)):
                    ppr_case(name+f" PPR alpha={alpha} L1={goal}", adjacency, s, degrees,
                             1 if name in ("ca-GrQc", "roadNet-CA") else 4, alpha, goal)
            else:
                degree = np.asarray(adjacency.sum(axis=1)).ravel()
                operator = .9*(diags(1/np.maximum(degree, 1)) @ adjacency)
                polynomial_case(name+" directed row-normalized polynomial", operator.tocsr(), 16)
            del adjacency
            if undirected: del s, degrees
            cleanup()
        except Exception as exc:
            REPORT["failures"].append({"dataset": name, "error": str(exc), "traceback": traceback.format_exc()})
            save()
            print("DATASET FAILURE", name, str(exc), flush=True)
    if not smoke and not community:
        for n, q in (((5_000_000, 4),) if scale_gpu else ((1_000_000, 8), (5_000_000, 4))):
            try:
                rng = np.random.default_rng(SEED+n)
                columns = np.column_stack((np.arange(n), rng.integers(0, n, (n, 6)))).ravel()
                weights = np.column_stack((np.full(n, .125), rng.choice([-1., 1.], (n, 6))/64)).ravel()
                operator = csr_array((weights, columns, np.arange(0, 7*n+1, 7)), shape=(n, n))
                operator.sum_duplicates(); operator.sort_indices()
                del columns, weights
                polynomial_case(f"Synthetic directed {n} vertices", operator, q)
                del operator
                cleanup()
            except Exception as exc:
                REPORT["failures"].append({"synthetic_n": n, "error": str(exc), "traceback": traceback.format_exc()})
                save()
        polynomial_case("Weak-tail no-truncation control", .99*eye_array(100_000, format="csr"), 4, decay=1)
        polynomial_case("Affine sensitivity 1000000 vertices", nonnormal(1_000_000), 8, sensitivity=True)
    REPORT["finished_utc"] = datetime.now(timezone.utc).isoformat()
    REPORT["execution_status"] = "complete" if not REPORT["failures"] and all(c["status"] == "complete" for c in REPORT["cases"]) else "partial"
    save()
    print("SINHA BENCHMARK FINISHED", REPORT["execution_status"], len(REPORT["cases"]), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--community", action="store_true")
    parser.add_argument("--scale-gpu", action="store_true")
    args = parser.parse_args()
    main(args.smoke, args.community, args.scale_gpu)
