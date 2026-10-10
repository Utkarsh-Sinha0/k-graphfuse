# Release notes

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
