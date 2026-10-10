# K GraphFuse — precision GPU upgrade

> **Experimental source and evidence, 10 October 2026.** This is separate from the installed package API. Private notebook account metadata and installation/kernel logs are excluded. Read the [technical comparison contract](https://utkarsh-sinha0.github.io/k-graphfuse/benchmarks/precision-gpu/) before reusing a result.

Evaluate sparse matrix-valued polynomial graph filters and their affine-parameter sensitivities. This experimental source package combines exact ordered coefficient fusion, similarity-aware truncation budgets, fused CUDA execution and selective compensated accumulation.

Six constructed Tesla T4 workloads achieved **1.25–3.74x speedups** against independently tuned FP64 CuPy/PyTorch controls while passing the recorded speed and sampled-accuracy gates. Start with [results and comparison contract](PRODUCT_RESULTS.md), [mathematical derivation](DERIVATION.md), and [research sources](RESEARCH.md). The workload is resident polynomial calculation, not a native-cuGraph PageRank comparison.

## Requirements

CPU execution needs Python, NumPy and SciPy. Fused execution additionally needs a CUDA GPU and CuPy; the wide projected profile also needs PyTorch. Tests need Mpmath; benchmark harnesses also use threadpoolctl.

Recorded GPU versions are CuPy 14.2.0 and PyTorch 2.11.0+cu128. The T4 environment used NumPy 2.1.3, SciPy 1.16.3 and Mpmath 1.3.0; the RTX 3060 environment used NumPy 2.2.6 and SciPy 1.14.1. Other combinations have not been calibrated. Use a CUDA runtime compatible with the installed library builds and a GPU with FP64 support.

This is an unpackable source/evidence bundle. Add this folder and `source/core` to the Python import path. The recorded core is included so the bundle can be checked independently of a later public release.

## CPU quick start

Run from the extracted package directory:

```python
import sys
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "source/core")]

from fractions import Fraction
import numpy as np
from scipy.sparse import csr_array
from metric_prototype import compile_metric, evaluate_metric

A = csr_array([[0.0, 0.5], [0.25, 0.0]])
X = np.array([[1.0], [2.0]])
# One stage: I + (1/2) A, with one feature channel.
stages = [[[[1]], [[Fraction(1, 2)]]]]
plan = compile_metric(A, stages, Fraction(1, 10**16))
Y = evaluate_metric(plan, X)
np.testing.assert_allclose(Y, X + 0.5 * (A @ X))
print(Y, plan["original_error_bound"])
```

The budget bounds exact-arithmetic truncation error in output Frobenius norm per input Frobenius norm. Coefficient encoding and floating execution require separate accounting. This is a different contract from maximum relative PageRank error per vertex.

## Checked fused GPU execution

Continue the example:

```python
import cupy as cp
from fused_acc import prepare_acc, evaluate_acc

prepared = prepare_acc(plan, compensated=False)
X_gpu = cp.asarray(X, dtype=cp.float64)
Y_gpu = evaluate_acc(prepared, X_gpu)
np.testing.assert_allclose(Y_gpu.get(), Y)
```

`evaluate_acc` checks shape, FP64 dtype, device, contiguity, finite input, parameter interval, scaled-input overflow and finite output. It returns owned output arrays. One prepared workspace must execute queries serially; use separate workspaces for independent concurrent queries.

These are the profiles selected in the recorded T4 cases, **not an automatic policy for arbitrary graphs or devices**:

| Recorded workload | Preparation options | Truncation target |
|---|---|---|
| 4-channel values | `compensated=False` | `Fraction(1, 10**16)` |
| 100,000-vertex, 4-channel value + derivative | `compensated=False, derivative_compensated=True` | Both `Fraction(1, 10**16)` |
| 1,000,000-vertex, 4-channel value + derivative | `compensated=True` | Both `Fraction(1, 10**16)` |
| 100,000-vertex, 16-channel values | `compensated=False, projected=True, coefficient_correction=False, final_compensated=True` | `Fraction(1, 10**15)` |

The fused backend accepts 1–16 channels, int32-indexable state/workspace and a matching diagonal parameter direction. General directions can use the CPU or checked PyTorch executor. Double-double execution is available through `prepare_fused` / `evaluate_fused` in `fused_dd.py`; it is retained as an accuracy option, not the latency winner on every case.

## Joint value and derivative

```python
from scipy.sparse import diags

E = diags([1.0, -1.0], format="csr")
plan = compile_metric(
    A, stages, Fraction(1, 10**16),
    direction=E, derivative_tolerance=Fraction(1, 10**16),
    parameter_radius=Fraction(1, 64),
)
prepared = prepare_acc(plan, compensated=True)
Y_gpu, dY_gpu = evaluate_acc(prepared, X_gpu, theta=Fraction(1, 128))
```

This differentiates `A + theta E` with fixed coefficients and similarity scales. It is directional sensitivity. The exact-arithmetic certificate covers the specified parameter interval; recorded floating outputs were checked at the center and endpoints.

## Coefficient certificate and full evaluation

```python
from coefficient_certificate import encoding_certificate
certificate = encoding_certificate(plan, paired=True)
# Use paired=False for the high-only wide projection profile.
print(certificate["value_tail_and_encoding_bound"])
```

This adds an exact rational coefficient-encoding bound to the truncation bound. Execution rounding and underflow remain additional. A two-part coefficient representation does not make every accumulator double-double.

For a full finite polynomial use `compile_full(A, stages)` and the included `evaluate_sparse`. All fused coefficients are retained, with the exact zero-operator simplification. Skipping an unnecessary zero-tail bound improves preparation; it does not change the existing warm executor.

## Tests and evidence audit

From this directory, with the recorded dependencies and a real CUDA device:

```text
python run_tests.py
python audit_product.py
python verify_package.py
```

`run_tests.py` deliberately initializes PyTorch first and runs 29 core tests before 14 upgrade tests in the same process. This exercises the final compiler/header and cross-library stream repairs. It fails on errors or skipped checks, and records source hashes in `FINAL_LOCAL_TESTS.json`. `audit_product.py` checks frozen timed-source hashes and recomputes the six T4 gates, block intervals and comparison files without running another GPU benchmark.

`verify_package.py` checks extracted-file completeness, SHA-256 hashes and correspondence with the tested API. It requires only Python's standard library. Checksums verify consistency with this supplied manifest; they are not a digital signature. Regenerating reports can change their hashes, so check integrity before running commands that write outputs.

| Evidence | Contents |
|---|---|
| `PRODUCT_VERIFIED.json`, `product_selected.csv` | Selected comparisons, gate calculations and source-report paths |
| `product_all_methods.csv` | All confirmed profiles, including losses and accuracy failures |
| `BENCHMARK_HIGH_PRECISION.json` | Independent 256/384-bit checks on the first 16 rows of large fixtures |
| `source/product_v1`, `source/wide_v1` | Exact immutable timed versions |
| `kaggle/product-*` | Returned T4 results, manifests and tests |
| `source/repaired_api` | Final API snapshot, separately regression-tested after timed runs |
| `PACKAGE_SHA256.json` | All package file hashes, excluding caches and the manifest itself |

The current API uses a small finite-check kernel that avoids optional CUB header dependencies. Projected execution on the default stream explicitly orders CuPy and PyTorch through a named stream and events. These repairs passed combined tests; large latency experiments were not rerun on the repaired API. Exact timed code remains frozen separately.

For CUDA graph capture, prevalidate outside capture and use an explicit non-default stream with `checked=False`. That path returns reused workspace arrays; copy the output when ownership is required. Captured benchmark replays fixed `theta=0`; changing a Python parameter after capture does not update the captured computation. Timings include similarity scaling and an owned GPU output copy, but exclude validation, transfers, compilation and JIT.

Earlier iterations remain in `RESULTS.md` and their raw reports; they are not interchangeable with the final precision portfolio. The local RTX 3060 portfolio did not pass all six combined gates: its million-vertex compensated derivative profile lost on speed.

## Adoption and license

Apply this mechanism to repeated sparse linear graph-feature filters or diagonal-parameter sensitivity sweeps, then compare against the strongest equivalent backend on the intended workload. The tested graph family and coefficients are constructed. No new broad public-graph PageRank result, production savings, energy reduction or global SOTA certification is established.

New code and documentation use Apache-2.0. The eight historical numerical modules and their original tests retain MIT notices; see `LICENSE`, `LICENSES/MIT.txt` and `NOTICE`. Visible acknowledgment and software citation are requested in `NOTICE` and `CITATION.cff`; they do not add a license restriction. This experiment is published separately from the installed v0.1.0 API. Original private experiments and public preparation remain distinct source identities.
