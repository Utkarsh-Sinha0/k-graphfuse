# Release notes

## Theorem-bound attribution and manuscript revision — 11 October 2026

- Adds a retrospective analysis of existing T4 measurements: two of eleven workload records reduce degree (23→22, 19→18), with 3.56–5.35% less observed median time across all eight matched FP64 settings.
- Records all medians, p95 and normalized output differences; pins source/sample provenance and separates the 17.1573% value-certificate factor from measured speed.
- Makes the untimed CPU 9→8 boundary and equal-degree laptop constant controls explicit. There is no new GPU run or algorithm implementation.
- Revises README, overview, benchmark methods/results, mathematical/contribution documentation, claims, agent navigation and the paper/PDF/LaTeX; adds a reproducible PNG/SVG chart and CPU audit.
- Retains independent-repeat/priority gaps, R-MAT and recommendation losses, all frozen sources and installed 0.1.0 API. These observations are below the 10% strong-speed bar and add no native-cuGraph wins.

## Frozen controlled study and revised manuscript — 10 October 2026

- Confirms repaired source on six graph families, ten fixtures, two fresh input/coefficient seeds per phase and RTX 3060 Laptop / Tesla T4; all source freezes precede their confirmation runs.
- Adds separate matched-degree packed-balanced controls: 18/20 strong pooled speed wins, 36.88–83.71% less resident time in the wins, and R-MAT losses on both devices.
- Adds a complete MovieLens recommendation demonstration with unchanged Recall@10, nDCG@10 and top-10 lists; an efficient scalar-restart control removes the end-to-end median speed advantage.
- Publishes all healthy raw results, an excluded-environment ledger, source/result manifests, 120 pooled component comparisons, same-estimator bootstrap intervals and two reproducible PNG/SVG graphics.
- Revises the [manuscript](../research/paper.md), PDF, closest-method review, applications, claims and agent navigation. [Controlled report](../research/controlled-study.md). Installed API and version remain 0.1.0.

## Research manuscript and presentation update — 10 October 2026

- Adds a technical research manuscript with a proof of similarity error transfer, coupled derivative recurrence, coefficient-encoding bound, reproducible methods, results and limitations; its status is an unreviewed preprint draft.
- Adds a downloadable PDF, standalone LaTeX source and three reproducible PNG/SVG figures: precision percentages, native win coverage and the evaluation pipeline.
- Presents latency reductions beside accurately labeled derived serial-rate increases. The six selected precision cases show 19.98–73.26% less median time and 24.98–273.95% higher derived rate against FP64 CuPy/PyTorch controls.
- Adds [metric definitions](../benchmarks/metric-definitions.md), figure source hashes, PDF provenance and offline presentation checks. [Detailed results](../benchmarks/results.md) and the [manuscript](../research/paper.md) retain negative results and comparator scope.
- Reuses the recorded evidence. This documentation update runs no new large GPU benchmark and makes no change to the installed API or package version.

## Evidence and documentation update — 10 October 2026

- Publishes 15 complete benchmark suites, 111 case records, 2,190 configurations and all accepted/rejected outcomes.
- Adds audited native-cuGraph totals: 59/89 strong speed wins and 26/89 speed-and-error wins, with four K accuracy failures retained.
- Publishes a separate precision GPU experiment: six T4 speed/accuracy passes against FP64 CuPy/PyTorch, with exact timed sources, sampled high-precision references and a later checked-API repair.
- Adds task-specific competitor, precision, derivation, application and reproduction pages, structured claims and offline CI audits.
- Preserves the original reference experiment, older zero-pass suite, MIT notices and installed `0.1.0` API. No production GPU API promotion or new package version is implied.

## 0.1.0 — 10 October 2026

- Introduces the public **K GraphFuse** package name and `k_graphfuse` namespace.
- Preserves eight byte-identical C2Fuse numerical modules and legacy imports.
- Adds an optional `k_graphfuse.gpu` import without loading PyTorch for CPU use.
- Packages professional technical documentation, runnable examples, citation and licenses.
- Publishes curated benchmark records, frozen run snapshots and offline audit tooling.

The rebrand does not change the algorithm or constitute a new performance experiment. Benchmarks were measured under **Sinha GraphFuse**. Existing manuscript-premise and numerical limits remain in effect. Version `0.1.0` is an evaluation release; plan mappings are not promised as a stable persisted format.
