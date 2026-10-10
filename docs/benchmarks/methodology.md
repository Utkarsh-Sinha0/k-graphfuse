# Benchmark methodology

## Choose the study contract

| Study | Task | Accuracy reference | Timing and uncertainty |
|---|---|---|---|
| [Theorem-bound attribution](../research/bound-attribution.md) | Same-backend old/new constant; packing/capture matched | Full FP64 CPU reference; both FP64 gates pass | 31 consecutive samples/configuration, shuffled configuration order; no paired repeated-block interval |
| [Broad native comparison](cugraph.md) | Fixed-update or converged rankings under each recorded graph contract | Full reference outputs; L1 and maximum relative error | Five paired blocks of 31 samples; bootstrap pooled medians |
| [Precision GPU experiment](precision-gpu.md) | Matrix-valued polynomial filters and affine derivatives | Full FP64 CPU check plus independent 256/384-bit truth on 16 rows | Five blocks of 31 samples; bootstrap block medians; seven pilot samples/configuration |
| [Original reference task](historical-reference.md) | Exactly ten global probability PageRank updates | Downloaded official Graphalytics per-vertex reference | 31 warm / 11 host-output samples; one-run sample bootstrap |

A graph vertex is one operator row; a channel is one feature column. FP32 and FP64 mean 32-bit and 64-bit floating-point execution. Acceptance, precision, output coordinates, graph normalization and timing boundaries must match the selected study. A rank calculation, an arbitrary polynomial filter and a shortest-path query are different tasks.

The broad study has 89 native-control cases: 85 with accepted K confirmations and four K accuracy failures. A strong speed win requires K median at most 90% of the comparator median, a descriptive interval above one and no p95 regression. The stricter native comparison additionally checks both observed reference error measures. Include failures in an attempted-case denominator; do not report only a favorable subset.

The precision study selects the fastest accuracy-eligible pilot profile and compares it with the fastest accepted held-out control. The policy was reconstructed after exploration, and the wide follow-up followed initial losses. Its published audit applies the 10% latency threshold; all six selected T4 results pass. An earlier exploratory audit used a 1.1 speedup threshold, equivalent to about 9.09% less latency. The public audit correction does not change the six selected T4 wins. Raw samples and timed source versions are unchanged.

Do not pool independent studies as additional native-cuGraph wins. A public graph or a large vertex count does not remove selection bias or establish a production-quality outcome.

The [metric definitions](metric-definitions.md) specify time reductions, derived serial rates, accuracy reductions and coverage percentages. Paired percentage units describe one latency ratio; they are not independent wins. The [technical manuscript](../research/paper.md) states the precision experiment's mathematical assumptions and method.

The documentation uses descriptive task names, explicit preconditions, consistent units and accessible figures, following the [Google developer documentation style guidance](https://developers.google.com/style/highlights). Separating profiling, correctness and timing boundaries follows the practical evaluation approach in the [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/). These are documentation and engineering references, not vendor endorsements of K GraphFuse.

## Original reference protocol

The public-reference protocol was saved before receiving the final timing result. The numerical core remains unchanged. The fixtures compare final original probability vectors; an independent SciPy iteration checks the polynomial before timed GPU execution. This is a Graphalytics reference task, not an officially audited benchmark submission.

## Workload contract

The source graph is undirected and unweighted for this task, with edge multiplicity preserved and no dangling vertices. The runner asserts these assumptions. Ten updates from a uniform state use damping `beta=0.85`:

```text
coeff[k] = (1-beta) * beta^k, k=0,...,9
coeff[10] = beta^10
```

A degree-based similarity transform produces a symmetric operator. Input is uniform probability divided by square-root degree; output is multiplied back inside the timed call. The compiler receives zero truncation. No top-k-only output, graph simplification or approximate early stopping substitutes for the reference.

The cuGraph call uses native probabilities, `alpha=0.85`, `epsilon=1e-30`, `max_iterations=10`, and `fail_on_nonconvergence=False`. It caches a transposed graph and outgoing sums. Epsilon zero is avoided because the vendor treats it as a default setting. The ten-step per-vertex reference, rather than a library convergence flag, determines acceptance.

## Accuracy gate

Vertices are aligned by original identifier. Each candidate must have finite scores and maximum per-vertex relative deviation no greater than `1e-4` versus the downloaded official reference. Aggregate L1 error and total mass are reported separately. Passing the gate is not an accuracy advantage on every metric.

## Timing boundaries

| Measurement | Included | Excluded |
|---|---|---|
| Warm resident compute | Accepted compute path, final probability coordinates, synchronization, validation/cloning as implemented | Initial graph download, CPU normalization/compiler work, host output |
| Host-output call | Compute plus output copy to host | Initial import and preparation |
| GPU preparation | Tested device graph/coefficient preparation | CPU normalization/compiler work; scopes differ by library |
| First call | One call after preparation | Repeated cold requests and process startup |
| CPU product compute | Released PageRank after graph analytical preparation | Import and graph-view construction |

Sample counts: 31 resident, 11 host-output, one first-call observation per GPU configuration. CPU product: one warm-up and 31 timed calls. Execution is sequential, not a randomized multi-run experiment. Candidate selection and host/GPU scheduling can affect observed results.

## Resource accounting

The monitor samples total process RSS, total device usage and board power every 20 ms during preparation and calls. It includes inherited allocations and library overhead and may miss short peaks. Board-power averages cover differing durations and are not integrated query energy. Do not infer cloud-bill savings, device capacity or isolated algorithm memory from these samples.

## Improvement gate and uncertainty

The scoped improvement gate is a median at least 10% lower than the accepted comparator, bootstrap median-ratio lower endpoint above one, and no p95 regression. The bootstrap uses 5,000 independent sample resamples with a fixed seed. The interval describes one run, is not corrected for selecting the fastest candidate, and does not measure cross-run variance. Independent hardware replication remains necessary.

## Provenance and earlier suite

Raw records retain **Sinha GraphFuse** names. Public K GraphFuse core files match the recorded module hashes. The earlier large/community/scale suite keeps all candidates, rejected outputs, exceptions and the incomplete original case. Its stronger controls and different accuracy/task contracts explain why its verdict differs from the scoped public-reference comparison. Refer to original protocol copies and candidate tables in the [evidence folder](https://github.com/Utkarsh-Sinha0/k-graphfuse/tree/main/evidence).

## Attribution denominators

The retrospective bound analysis examines eleven legacy workload records: two reduce degree, nine do not. It reports all four accepted matching FP64 settings for each changed-degree record. Eight positive timing comparisons are two workloads, not eight independent datasets. Their 3.56–5.35% observed changes do not pass the separate 10% strong-speed bar and add no native comparison wins.
