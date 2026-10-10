# K GraphFuse precision GPU portfolio — verified results

The new implementation combines a similarity-aware mathematical plan with fused GPU execution and selective higher-precision accumulation. It is an experimental GPU upgrade; the released repository and historical evidence are unchanged.

## Independent Tesla T4 comparisons

Each row compares the selected accuracy-eligible profile with the fastest accepted original-coordinate FP64 control, independently tuned for packed/unpacked coefficients and eager/captured execution where supported. Controls are CuPy 14.2.0 and PyTorch 2.11.0+cu128 on the same T4. The wide row is a separately frozen confirmation after the initial wide-feature candidates lost.

| Vertices | Channels | Calculation | Control | Control ms | New ms | Speedup | 95% block interval | Worst sampled component error decrease | Speed + accuracy gate |
|---:|---:|---|---|---:|---:|---:|---|---:|---|
| 10,000 | 4 | Polynomial filter | torch | 0.944 | 0.282 | 3.35x | 2.65–3.55x | 0.0077% | PASS |
| 100,000 | 4 | Polynomial filter | cupy | 4.334 | 1.159 | 3.74x | 3.60–3.79x | 1.9316% | PASS |
| 1,000,000 | 4 | Polynomial filter | cupy | 43.258 | 14.141 | 3.06x | 2.93–3.13x | 10.0281% | PASS |
| 100,000 | 16 | Polynomial filter | torch | 14.216 | 9.179 | 1.55x | 1.54–1.55x | 1.4723% | PASS |
| 100,000 | 4 | Value + derivative | cupy | 10.028 | 4.839 | 2.07x | 2.05–2.07x | 28.7375% | PASS |
| 1,000,000 | 4 | Value + derivative | cupy | 104.502 | 83.618 | 1.25x | 1.24–1.28x | 19.9051% | PASS |

Five randomized confirmation blocks of 31 synchronized samples per selected method, following seven pilot samples/configuration. Fourteen FP64 configurations per main case and eight in the separate wide confirmation. Input/output similarity scaling and an owned GPU output copy are inside timing. Validation, preparation, JIT and host transfer are outside it. Every rejected or slower candidate is retained.

Accuracy acceptance compares **every** sampled output/derivative Frobenius error component and maximum absolute error against the fastest control; at least one must improve. The displayed percentage uses the largest Frobenius component, so it does not describe every component or every vertex. Values already lie near the FP64 rounding floor: small value-error improvements are real but modest. Derivative components can improve substantially more.

## What changed and why

- Transform the operator with a fixed dyadic similarity, paying its exact condition penalty in both value and derivative budgets. The smaller geometry permits a stricter truncation target without a very long polynomial.
- Fuse sparse products, coefficient injection and joint derivative updates into fewer GPU launches and fewer intermediate arrays.
- Store coefficients as high plus low parts where that helps; use FMA, compensated sums or mixed value/derivative precision according to the measured accuracy contract.
- For wide features, project coefficients with standard BLAS and use a separately compiled compensated final kernel. Compensation inside every iteration increased register pressure and latency; a final-only kernel avoids that cost.
- Reuse state buffers and perform final scaling in place internally. Checked API calls still return owned outputs and reject invalid states.

## Mathematical and numerical evidence

`DERIVATION.md` proves the coordinate error transfer. `coefficient_certificate.py` bounds coefficient-encoding error exactly using rational arithmetic and the existing complete-Crouzeix/derivative bounds. Main fused plans target 1e-16 truncation; the high-only wide follow-up targets 1e-15. These certificates cover exact-arithmetic truncation and encoding, not every floating operation.

Every full output was compared against full degree-48 FP64 CPU evaluation with an accepted normalized difference <=1e-13. Independent 256/384-bit references and two finite influence windows cross-check the first 16 rows of the exact large inputs. Window/precision discrepancies were below 1e-65. The oracle distinguishes real error from merely matching an already rounded FP64 answer. See `BENCHMARK_HIGH_PRECISION.json` and the raw product reports.

## Complexity and useful workloads

For n vertices, m stored edges, q channels and retained degree d, work remains O(d m q + d n q^2). This iteration improves constants, memory traffic and numerical evaluation; it does not implement a new matrix-multiplication exponent. Narrow fused workspace is O(m + n q + d q^2); wide dense projection adds O(d n q), matching the packed control class. No energy or cloud-bill reduction was measured.

Best-fit hypotheses are repeated sparse graph-feature refreshes, ranking/retrieval feature stages, linear diffusion filters and diagonal-parameter network/simulation sweeps that need both value and sensitivity. Downstream fraud detection, recommendation quality and production savings still require workflow validation. These are not BFS, shortest paths, graph transactions or HFT trading-alpha results.

## Robustness and reproduction

The current tests cover canonical sparse input checks, finite real states, dtype/shape/device/contiguity boundaries, parameter intervals, projection index limits, owned outputs, capture correctness, noncommuting coefficient fusion, exact zero operators, general asymmetric/symmetric controls and an independent high-precision rounding-floor check. The fused backend is calibrated for 1–16 channels and diagonal parameter directions. One prepared workspace must not run concurrent queries.

Use `README.md` for CPU and checked GPU examples. `source/product_v1` and `source/wide_v1` preserve exact timed versions; each returned Kaggle run has a source manifest, complete numerical results and test outcomes. `product_all_methods.csv` includes losses; `PRODUCT_VERIFIED.json` contains the gate calculations. Run `audit_product.py` to recompute claims without rerunning CUDA.

## Scope before adoption

This is one constructed asymmetric graph family at 10,000–1,000,000 vertices with 4/16 channels and constructed polynomial coefficients. It is not a broad public-graph validation or a global SOTA certification. The previous 104-case ranking/filter suite, four PageRank accuracy gaps, native Graph500 losses and unsupported graph algorithms have not been rerun or resolved by this package. JIT, cold startup, end-to-end transfers, resilience, energy and customer value remain separate gates.

The product uses classical prior art and a derived application of the cited mathematical premise. It makes no claim of new mathematical priority or absence of competitors. These experimental records are published separately from the installed package API.

## Final API repair and chart

The final checked API uses a self-contained finite-check kernel and explicit stream ordering for default-stream CuPy/PyTorch projection. The 29 core and 14 upgrade tests passed together after these repairs; see `FINAL_LOCAL_TESTS.json`. Exact performance versions remain frozen in `source/product_v1` and `source/wide_v1`. The large latency runs were not repeated on the repaired API.

![Six audited T4 comparisons](figures/precision_t4_comparison.png)

The chart is derived from `PRODUCT_VERIFIED.json`; `make_chart.py` regenerates the PNG and SVG without running a benchmark.
