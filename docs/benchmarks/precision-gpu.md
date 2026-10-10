---
title: Precision GPU graph filters — speed, accuracy, derivatives and reproducibility
description: Six audited FP64 sparse polynomial and sensitivity cases on Tesla T4: 1.25–3.74x speedups, sampled 384-bit accuracy checks, frozen sources, p95, and reproducibility.
---
# Precision GPU experiment

The experimental K GraphFuse portfolio passes speed and sampled-accuracy gates on **six of six constructed T4 cases**, with **1.25–3.74x speedups** against the fastest accepted equivalent FP64 CuPy/PyTorch control. This is a sparse polynomial and directional-sensitivity study. It does not compare native cuGraph PageRank or replace the installed `k_graphfuse` API.

## Latency and tail latency

All columns are resident compute. Selected configurations include an owned GPU output copy and input/output similarity scaling.

| Vertices | Channels | Calculation | Control | Control median ms | K median ms | Speedup | Block 95% interval | Control p95 ms | K p95 ms |
|---:|---:|---|---|---:|---:|---:|---|---:|---:|
| 10,000 | 4 | Filter | torch | 0.944 | 0.282 | 3.35x | 2.65–3.55x | 1.057 | 0.400 |
| 100,000 | 4 | Filter | cupy | 4.334 | 1.159 | 3.74x | 3.60–3.79x | 4.369 | 1.300 |
| 1,000,000 | 4 | Filter | cupy | 43.258 | 14.141 | 3.06x | 2.93–3.13x | 44.172 | 15.176 |
| 100,000 | 16 | Filter | torch | 14.216 | 9.179 | 1.55x | 1.54–1.55x | 14.336 | 9.290 |
| 100,000 | 4 | Value + derivative | cupy | 10.028 | 4.839 | 2.07x | 2.05–2.07x | 10.109 | 4.888 |
| 1,000,000 | 4 | Value + derivative | cupy | 104.502 | 83.618 | 1.25x | 1.24–1.28x | 106.124 | 85.518 |

![Six constructed T4 cases comparing resident speed and sampled numerical error; scope and excluded costs appear in the figure](../assets/precision_t4_comparison.png)

Median ratios use all 155 confirmation samples. The displayed interval bootstraps the ratio of median block medians, a different estimator; it need not contain the pooled-median ratio. Intervals describe within-run block variability, not an independently replicated hardware population.

## Numerical accuracy

Every sampled output/derivative Frobenius and maximum-absolute error must be no worse than the fastest control, with at least one improvement. The percentage below compares the largest sampled Frobenius component; it is not an improvement percentage for every vertex or for downstream model quality.

| Workload | Decrease in largest sampled Frobenius component |
|---|---:|
| 10,000 vertices, 4 channels, filter | 0.0077% |
| 100,000 vertices, 4 channels, filter | 1.9316% |
| 1,000,000 vertices, 4 channels, filter | 10.0281% |
| 100,000 vertices, 16 channels, filter | 1.4723% |
| 100,000 vertices, 4 channels, value + derivative | 28.7375% |
| 1,000,000 vertices, 4 channels, value + derivative | 19.9051% |

Value errors are already near the FP64 rounding floor. For example, the million-vertex four-channel filter moves from approximately `5.659e-17` to `5.091e-17` sampled normalized error. Small relative improvements at this floor should not be advertised as a large application-quality improvement. The million-vertex derivative component at theta zero improves from `9.003e-18` to `3.674e-18`; other components and endpoints are recorded separately.

All full outputs were checked against a degree-48 FP64 CPU computation with normalized difference at most `1e-13`. Independent 256/384-bit references validate the first 16 rows of the exact large fixtures. Two influence windows agreed below `1e-65`. The local finite influence cone proves those sampled rows are unaffected by the artificial window boundary; it does not provide arbitrary-precision truth for every graph vertex.

## Mechanism and mathematical contract

1. Fuse compatible linear stages in their original order, retaining noncommuting coefficient order.
2. Apply a fixed dyadic similarity `B = D^-1 A D`. Compile in the balanced coordinates with value and derivative tolerances divided by `kappa(D)`.
3. Fuse CSR edge products, coefficient injection and coupled Horner derivative updates into fewer kernels and reused state buffers.
4. Select FMA, compensated or mixed derivative accumulation according to measured accuracy, rather than assuming extra arithmetic always helps.
5. For the wide-feature case, use BLAS coefficient projection and a separately compiled compensated final step to avoid compensation register cost at every step.

The tested scale condition penalty is eight, not two. The smaller balanced operator enclosure nevertheless allows stricter tail targets with fewer passes. Main plans target `1e-16` truncation; the high-only wide follow-up targets `1e-15`. Exact rational coefficient-encoding bounds are additional to the truncation bounds. Execution rounding and underflow remain outside those certificates. Read the [derivation](../concepts/precision.md) and [research ledger](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/experiments/precision_gpu/RESEARCH.md).

Work is `O(d m q + d n q^2)` for degree d, stored edges m, vertices n and channels q. Narrow workspace is `O(m + n q + d q^2)`; dense wide projection adds `O(d n q)`. This is a constant-factor and numerical-evaluation improvement. The family-107 asymptotic matrix-multiplication accelerator is not implemented.

## Profiles and supported inputs

| Recorded profile | Workload | Preparation |
|---|---|---|
| FMA | Four-channel filter | `compensated=False` |
| Mixed | 100k four-channel sensitivity | `compensated=False, derivative_compensated=True` |
| Compensated | Million-vertex four-channel sensitivity | `compensated=True` |
| Wide final compensation | 100k sixteen-channel filter | `compensated=False, projected=True, coefficient_correction=False, final_compensated=True` |

These are calibrated profiles, not a universal automatic router. The fused backend accepts real finite canonical sparse operators, resident contiguous FP64 inputs, 1–16 channels, int32-indexable state/workspace and diagonal affine directions. General directions use the existing CPU/PyTorch fallback. One prepared workspace must run queries serially. Checked calls return owned outputs and reject invalid shape, dtype, device, parameters and nonfinite results.

## Experiment protocol

Controls: CuPy 14.2.0 and PyTorch 2.11.0+cu128 FP64 on the same Tesla T4, independently tuned for packing and capture where supported. Each main case tests 14 configurations; the separately frozen wide follow-up tests eight. Seven pilot samples precede five randomized confirmation blocks of 31 synchronized samples per selected method. The fastest accuracy-eligible pilot profile is selected; the denominator conservatively uses the fastest accepted held-out control.

The public audit enforces at least 10% lower median, interval lower endpoint above one, p95 no worse and the sampled accuracy screen. The selection policy was reconstructed after exploratory results; the wide follow-up was designed after initial wide profiles lost. All rejected/slower candidates remain available. Candidate selection and the small number of blocks limit inference.

The family is constructed: alternating asymmetric couplings 1 and 1/64, small negative diagonal entries, degree-48 rational matrix coefficients with geometric decay, and 4/16 channels. Sensitivity uses a diagonal direction and theta in `[-1/64, 1/64]`. Timing replays theta zero; accuracy additionally checks interval endpoints.

Timing excludes validation, host transfers, preparation and JIT. No new isolated peak-memory, integrated energy, end-to-end throughput or bill measurement was made. The earlier cold-compiler experiment is a separate source/timing contract.

## Source versions and robustness

| Version | Evidence meaning |
|---|---|
| `source/product_v1` | Exact main timed implementation |
| `source/wide_v1` | Exact separately confirmed wide implementation |
| Current experiment root / `source/repaired_api` | Later checked-API repair; 29 core plus 14 upgrade tests passed together |

The repaired API avoids an optional CUB header/compiler mismatch during finite validation and explicitly orders default-stream CuPy/PyTorch projection. It was regression-tested, not given another full large latency confirmation. CPU-only CI audits archived facts; it does not reproduce GPU timing or validate new GPU hardware.

The RTX 3060 portfolio did not pass all combined gates: its million-vertex compensated derivative profile lost on speed. The broad ranking suite, its four accuracy failures and native Graph500 losses were not rerun by this experiment.

## Evaluate an application

Best-fit hypotheses are repeated low-channel sparse linear graph features, asymmetric diffusion filters and diagonal-node-parameter sensitivity sweeps. Require matching operators, coefficients, channel count and budgets; then measure the complete workflow against your current production implementation. No fraud recall, recommendation nDCG, trading alpha or customer savings follows from this calculation benchmark alone.

## Reproduce and inspect evidence

- [Experiment code and checked usage](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/experiments/precision_gpu/README.md)
- [All confirmed methods, including losses](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/experiments/precision_gpu/product_all_methods.csv)
- [Audited selected results](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/experiments/precision_gpu/PRODUCT_VERIFIED.json)
- [Independent high-precision evidence](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/experiments/precision_gpu/BENCHMARK_HIGH_PRECISION.json)
- [Final API test record](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/experiments/precision_gpu/FINAL_LOCAL_TESTS.json)

Run the [offline audit](reproduce.md) before making a new GPU execution. Keep the frozen timed sources and current API results separate.
