---
title: K GraphFuse benchmark evidence — cuGraph, precision filters and sensitivity
description: Compare native NVIDIA cuGraph, tuned FP64 GPU controls, latency, sampled memory and numerical accuracy using raw records, audited gates, failures and reproducibility.
---
# Benchmark results and competitors

The [theorem-bound attribution analysis](https://utkarsh-sinha0.github.io/k-graphfuse/research/bound-attribution/) separates the new complete-Crouzeix bound from balancing and GPU fusion. **Two of 11 legacy T4 workload records reduce degree: 23→22 and 19→18.** Across all eight accepted matched FP64 configurations in those two records, the new-bound plan records **3.56–5.35% less median resident time**, with lower observed p95. This is single-run evidence below the separate 10% strong-speed gate. The exact CPU 9→8 boundary is untimed; the checked laptop old/new-bound cases retain equal degrees.

The new [controlled confirmation](https://utkarsh-sinha0.github.io/k-graphfuse/research/controlled-study/) uses the repaired implementation across six graph families and two GPUs. Matching-budget, matching-degree packed-balanced CuPy comparisons pass the strong speed gate in **18/20 pooled device–fixture comparisons**, with **36.88–83.71% less resident time in the wins**. R-MAT loses on both devices. MovieLens recommendation quality matches exactly, but specialized scalar-restart CuPy removes the application median speed advantage: K uses **7.13% more time on RTX** and practically ties T4. These are scoped experimental results, separate from native-cuGraph PageRank and the installed API.


K GraphFuse has measured advantages on specific repeated graph calculations. The published studies use different task contracts and source versions; select the matching comparison before reusing a result.

## Published results

| Study | Main result | Comparator and scope |
|---|---|---|
| [Broad native cuGraph study](cugraph.md) | **66.3% (59/89 attempted) strong speed wins**, including **29.2% (26/89) speed-and-error wins** | 85 comparable cases plus four K accuracy failures, counted as non-wins |
| [Strongest tested alternatives](cugraph.md#compare-against-the-strongest-tested-alternative) | **38.8% (33/85) overall**, **44.7% (38/85) GPU** speed-gate wins | Fastest accepted tested backend, including CPU where measured |
| [Precision GPU experiment](precision-gpu.md) | **Six of six** speed-and-sampled-accuracy gates; **~20.0–73.3% less median compute time** (1.25–3.74x), **~25.0–273.9% higher derived serial rate** | Equivalent tuned FP64 CuPy/PyTorch controls; constructed filters and derivatives |
| [Original reference experiment](historical-reference.md) | Eight favorable latency/memory/error observations out of nine | One same-T4 ten-update PageRank setup; aggregate L1 error worse |

For compute comparisons, `s = baseline median / K median`; time reduction is `100 × (1 − 1/s)`, and the derived reciprocal-latency serial rate increase is `100 × (s − 1)`. This rate comes from isolated median latency, not measured concurrent throughput. “Speed-and-error win” in the broad suite means the strong speed gate plus no worse recorded L1 and maximum-relative reference error. It does not include every resource or downstream quality metric. The precision study uses an independent high-precision sampled oracle and a different error contract.

![Six constructed Tesla T4 workloads: about 20.0–73.3% lower median compute time and 25.0–273.9% higher derived serial rate versus tuned FP64 CuPy or PyTorch, with sampled-accuracy scope](../assets/precision_percent_comparison.png)

![Native comparison coverage: 59 of 89 attempted cases pass the strong speed gate, 26 also pass both error comparisons, and four cases fail K accuracy confirmation](../assets/native_win_coverage.png)

## Where the evidence is strongest

- Repeated resident ranking stages reduce median compute time by 47.8–65.0% on several selected public graphs versus native cuGraph (`s=1.91–2.86x`, or 91.5–185.6% higher derived serial rate). Stronger CuPy controls still lead on some of those same graphs.
- The precision experiment records 20.0–73.3% lower median compute time and 25.0–273.9% higher derived serial rate, with sampled numerical accuracy gates, for constructed low-channel sparse polynomial filters and diagonal-parameter derivatives.
- The original reference run records lower sampled total GPU/process memory, with short-peak and inherited-allocation limitations.

Use [application fit](../guides/applications.md) to translate a calculation-stage result into a pilot. Production benefit depends on the time spent in that stage, graph updates, transfers, concurrency and downstream quality.

## Results that limit the claim

On Graph500-22, K’s `s=0.368x` relative to native means 172.0% more median compute time and 63.2% lower derived serial rate; native’s reciprocal-median rate was 2.72x K’s. A later datagen confirmation also favors native cuGraph, despite the original one-run 21.28% lower K median compute time (63.398 ms native /49.906 ms K; `s=1.2703x`, or 27.03% higher derived serial rate). Full file-processing and rebuild workflows favor native cuGraph. Four broad ranking cases fail K accuracy confirmation. K does not implement BFS, shortest paths, components, triangles or Louvain.

The older 19-workload study passed zero strongest GPU gates under its separate contracts. Its raw records remain published rather than replaced by newer wins. Neither study establishes global SOTA, absence of alternatives, integrated energy savings or customer value.

## Audit trail

Every study links raw samples, configurations, acceptance criteria, failed candidates and frozen source versions. Run `python benchmarks/verify_evidence.py` to check archived statistics, source hashes and claim counts without rerunning the GPU experiments. [Methodology](methodology.md) · [Reproduce and audit](reproduce.md) · [Claims ledger](../claims.json) · [Support limits](../reference/limitations.md).
