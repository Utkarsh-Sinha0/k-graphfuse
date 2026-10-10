# Release notes

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
