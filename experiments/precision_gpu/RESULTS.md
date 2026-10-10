# K GraphFuse private mathematics upgrade: measured results

Two targeted changes are implemented and tested: similarity-aware polynomial/derivative planning, and skipping unnecessary range bounds when retaining every coefficient. These are improvements to the current implementation, not a claim of new mathematical priority or universal superiority.

## Local resident GPU results

Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB. Every case uses the same alternating asymmetric directed operator family, constructed features and degree-48 matrix-valued coefficients. Budget: 1e-8 output Frobenius error per input Frobenius norm. Five randomized held-out blocks of 31 synchronized samples per selected method, after seven pilot samples per accuracy-accepted configuration. Each method independently searches FP32/FP64, packed/unpacked and eager/captured configurations. Input/output similarity scaling is inside the timed query. These are prevalidated resident-compute timings; host-side input/output validation is outside the timed kernel loop. The checked functional API is separately regression-tested and is not the timed replay adapter.

| Vertices | Channels | Calculation | Fastest old control | Old ms | Metric ms | Speedup | 95% block interval | Gate |
|---:|---:|---|---|---:|---:|---:|---|---|
| 10,000 | 4 | Polynomial filter | older_bound | 1.107 | 0.799 | 1.39x | 1.30–1.50x | PASS |
| 100,000 | 4 | Polynomial filter | released | 5.840 | 3.771 | 1.55x | 1.54–1.59x | PASS |
| 1,000,000 | 4 | Polynomial filter | older_bound | 54.565 | 35.236 | 1.55x | 1.42–1.69x | PASS |
| 100,000 | 16 | Polynomial filter | older_bound | 12.903 | 7.755 | 1.66x | 1.65–1.69x | PASS |
| 100,000 | 4 | Value + derivative | older_bound | 16.444 | 9.319 | 1.76x | 1.58–1.91x | PASS |
| 1,000,000 | 4 | Value + derivative | released | 170.391 | 93.412 | 1.82x | 1.81–1.84x | PASS |

The old control is whichever is faster in confirmation: the released constant-two planner or the optimized older 1+sqrt(2) planner. It is not native cuGraph. Reported intervals resample the five paired whole-block medians 20,000 times; they do not express confidence over arbitrary graphs, hardware or customer workloads. Gate: at least 10% lower median, lower interval endpoint above 1, and no worse p95.

## Private matched Kaggle results

| Vertices | Channels | Calculation | Fastest old control | Old ms | Metric ms | Speedup | 95% block interval | Gate |
|---:|---:|---|---|---:|---:|---:|---|---|
| 10,000 | 4 | Polynomial filter | released | 1.330 | 0.905 | 1.47x | 1.18–1.47x | PASS |
| 100,000 | 4 | Polynomial filter | cupy_released | 4.302 | 3.914 | 1.10x | 1.10–1.11x | FAIL |
| 1,000,000 | 4 | Polynomial filter | cupy_released | 42.896 | 37.347 | 1.15x | 1.13–1.16x | PASS |
| 100,000 | 16 | Polynomial filter | released | 15.122 | 9.312 | 1.62x | 1.62–1.63x | PASS |
| 100,000 | 4 | Value + derivative | cupy_released | 12.647 | 9.605 | 1.32x | 1.31–1.32x | PASS |
| 1,000,000 | 4 | Value + derivative | cupy_released | 131.703 | 100.420 | 1.31x | 1.31–1.33x | PASS |

T4 follow-up uses PyTorch 2.11.0+cu128, NumPy 2.1.3, SciPy 1.16.3 and CuPy 14.0.1. It adds FP32/FP64 ordinary and packed CuPy Horner controls with the same polynomial/derivative semantics. Both original-coordinate and metric-coordinate CuPy adapters are recorded. The primary metric comparison in this table uses the PyTorch prototype against the fastest accepted original-coordinate control, including CuPy. CuPy cannot capture the tested sparse CSR operation in the installed version, so only its eager variants are searched. This is not a rerun of the 104 earlier ranking/filter cases or a native cuGraph PageRank comparison.

## Local cold full-polynomial preparation

Fixed ten-update, directed weighted column-stochastic propagation, FP64 eager execution, both methods retain degree 10. Eleven randomized samples/method from a loaded normalized CSR through validation, compilation, upload, first execution and full host output. File import and graph normalization are outside the timing. This is a diagnostic with fixed settings, not a held-out configuration competition.

| Vertices | Current total ms | Zero-tail fast path ms | Speedup | Current compile ms | Fast compile ms |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 12.661 | 7.644 | 1.66x | 5.358 | 0.902 |
| 100,000 | 75.246 | 17.588 | 4.28x | 62.154 | 4.544 |
| 1,000,000 | 905.381 | 124.666 | 7.26x | 803.061 | 45.096 |

The fast path reuses exactly the same executor and all coefficients. Its truncation error is zero because it discards no terms. It does not certify zero floating error or claim a warm-runtime reduction. Every measured output passed a full CPU reference with an observed normalized error <=1e-12.

## Meaning and boundaries

- Repeated asymmetric multi-feature propagation: fewer sparse passes at the same accepted numerical budget.
- Affine parameter sensitivity: compute a value and parameter derivative jointly with fewer retained terms; possible use in repeated network-response or simulation sweeps.
- Frequently rebuilt finite-propagation operators: reduce validation-to-first-output preparation overhead by omitting a mathematically unnecessary tail enclosure.
- All application and cloud-bill benefits still require workflow measurements. No customer deployment, energy reduction or money saved was measured.
- No automatic guarantee of improvement on already balanced graphs, huge condition penalties, real public graphs, or all earlier PageRank cases. Do not enable the metric blindly.
- Four earlier PageRank accuracy gaps, Graph500 native-cuGraph losses and unsupported graph algorithms remain unresolved.

## Reproduce and inspect

`LOCAL_RESULTS.json` and `COLD_RESULTS.json` preserve samples and rejected configurations. `VERIFIED_SUMMARY.json` and CSV files are derived by `audit_results.py`. `source/` contains the frozen prototype, regression checks, and local/T4 benchmark harnesses. `DERIVATION.md` contains the coordinate and derivative error transfer; `RESEARCH.md` links the selected research and classical prior art.

From the original workspace, run `work/.venv/Scripts/python.exe outputs/kgf_math_upgrade/audit_results.py`. Re-running timings requires the recorded numerical core, NumPy/SciPy/PyTorch and a CUDA device. T4 runs are independently frozen private Kaggle scripts; only returned completed evidence enters the report. Public repository and historical benchmark files are unchanged.
