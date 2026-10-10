# Limitations and support

The new [controlled confirmation](https://utkarsh-sinha0.github.io/k-graphfuse/research/controlled-study/) uses the repaired implementation across six graph families and two GPUs. Matching-budget, matching-degree packed-balanced CuPy comparisons pass the strong speed gate in **18/20 pooled device–fixture comparisons**, with **36.88–83.71% less resident time in the wins**. R-MAT loses on both devices. MovieLens recommendation quality matches exactly, but specialized scalar-restart CuPy removes the application median speed advantage: K uses **7.13% more time on RTX** and practically ties T4. These are scoped experimental results, separate from native-cuGraph PageRank and the installed API.


K GraphFuse `0.1.0` is a research/evaluation release. The declared dependency floors are installation requirements, not proof that every compatible version or platform has been tested.

| Environment | Evidence |
|---|---|
| Python 3.11 and 3.13, Linux CPU | Repository CI runs unit tests, examples, build and evidence audit |
| Windows Python 3.13, RTX 3060 Laptop GPU | Local publication checks run CPU and CUDA tests against the installed wheel |
| Linux Kaggle, Tesla T4 | Archived benchmark experiments; PyTorch 2.11.0+cu128 |
| Native cuGraph comparator | `pylibcugraph-cu12 26.10.0`, CuPy 14.0.1; Linux benchmark fixtures |
| Other GPUs, architectures and mixed workloads | Not established by this release |

CI status is visible in the repository. CUDA tests skip on CPU-only CI; a green CPU job is not GPU benchmark replication. Benchmark packages and external datasets have their own licenses and environment requirements.

## Mathematical limits

- Fusion requires one operator, one channel size and linear stages without intervening nonlinear activations.
- Complete `K=2` certificates rely on a cited manuscript claim; classical bounds remain alternatives.
- Omitted-tail certificates exclude floating-point conversion and executor roundoff.
- The sensitivity model has one fixed affine direction and a fixed parameter interval.
- The symmetric resolvent requires exactly symmetric stored data and a valid spectral enclosure.
- The local backend can touch the whole graph.

## Performance limits

The original same-T4 reference measurement recorded 21.28% less median compute time, equivalent to 27.03% higher derived serial compute rate, on one selected ten-update PageRank setup. Its aggregate L1 error was worse, and a later longer confirmation favors cuGraph. It is a historical scoped result. It does not establish wider uniqueness, global SOTA, universal precision improvement, database-service replacement, revenue or bill savings. The earlier complete suite passed zero of 19 strongest GPU improvement gates. Best-candidate selection and within-run intervals do not replace independent runs.

The newer 15-suite study passes 59 native speed gates among 89 attempted cases (**66.3% coverage**), or 26/89 (**29.2%**) when both observed error measures must also be no worse. Four K accuracy failures remain. Against the strongest tested overall control it passes 33/85 comparable cases (**38.8%**). These are gate-coverage percentages, not latency reductions. Graph500 and complete file/rebuild workflows favor native. Read the [native comparison](../benchmarks/cugraph.md) before reusing its totals.

The [precision experiment](../benchmarks/precision-gpu.md) passes six constructed T4 cases against equivalent FP64 CuPy/PyTorch controls: **19.98–73.26% less median compute time**, equivalent to **24.98–273.95% higher derived serial compute rate**. These are not native cuGraph comparisons or measured service throughput. The API has 1–16-channel and diagonal-direction limits, sampled high-precision truth on 16 rows, and separate timed/repaired source versions. Its later repaired API did not receive a new full large latency run. Its local RTX 3060 million-vertex derivative does not pass the speed gate. These experimental functions are outside the installed package API. The [technical manuscript](../research/paper.md) gives the derivation and experiment; [metric definitions](../benchmarks/metric-definitions.md) explain the conversions.

Sampled memory may miss peaks and includes inherited/library allocations. GPU-preparation measurements exclude CPU normalization/compiler work. Sampled average board power was higher; integrated energy was not measured. Customer projections remain models until measured.

## Operational limits

Plans are not a secure boundary against hostile in-process mutation. One CUDA executor cannot be shared concurrently. Arbitrary graph changes require a new plan. Benchmark runners download vendor artifacts and datasets and use a disposable Linux workspace; they are manual research fixtures and do not execute during package installation.

Read the [accuracy contract](../concepts/contracts.md), [benchmark methodology](../benchmarks/methodology.md) and [application gates](../guides/applications.md) before production evaluation.

## Theorem-bound attribution update

The [theorem-bound attribution analysis](https://utkarsh-sinha0.github.io/k-graphfuse/research/bound-attribution/) separates the new complete-Crouzeix bound from balancing and GPU fusion. **Two of 11 legacy T4 workload records reduce degree: 23→22 and 19→18.** Across all eight accepted matched FP64 configurations in those two records, the new-bound plan records **3.56–5.35% less median resident time**, with lower observed p95. This is single-run evidence below the separate 10% strong-speed gate. The exact CPU 9→8 boundary is untimed; the checked laptop old/new-bound cases retain equal degrees.
