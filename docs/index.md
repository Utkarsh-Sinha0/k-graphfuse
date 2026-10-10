---
title: K GraphFuse — sparse GPU graph analytics and polynomial filters
description: Compile reusable graph polynomial plans, evaluate directional sensitivities, and inspect matched PageRank benchmarks against native NVIDIA cuGraph.
---
# K GraphFuse

**Compile compatible linear graph operations once. Reuse sparse CPU or CUDA execution across inputs.**

**Created by [Utkarsh Sinha](https://github.com/Utkarsh-Sinha0).** [Cite K GraphFuse](project/citation.md) when using its implementation or evidence.

K GraphFuse turns a sequence of compatible graph filters into an ordered matrix polynomial. It evaluates the plan without forming graph matrix powers or a dense graph operator. The package also provides a one-parameter directional derivative, a symmetric Chebyshev resolvent, local sparse propagation and an exact rational residual audit.

![K GraphFuse compiles compatible linear stages into an ordered polynomial, then reuses a sparse operator and plan for evaluation](assets/graphfuse_pipeline.png)

The intended opportunity is repeated graph computation on a reused operator. Compilation, graph conversion and GPU preparation are additional costs. Matching a mathematical operation alone does not establish a production speedup: include downstream quality, transfers, projection, updates and writeback in your evaluation.

## Audited performance evidence

| Study | Outcome | Read the matching contract |
|---|---|---|
| Native cuGraph ranking | 66.3% of 89 attempted cases passed the strong speed gate; 29.2% also had both reference errors no worse | [Native cuGraph comparison](benchmarks/cugraph.md) |
| Strongest tested controls | 38.8% overall and 44.7% GPU speed-gate passes | [Comparator selection](benchmarks/cugraph.md#compare-against-the-strongest-tested-alternative) |
| Precision GPU experiment | Six of six gates; about 20.0–73.3% less median compute time (1.25–3.74x), equivalent to 25.0–273.9% higher derived serial rate | [Filters, derivatives and sampled high-precision accuracy](benchmarks/precision-gpu.md) |

The native denominator includes four K accuracy failures, counted as non-wins. The precision study compares against equivalent tuned FP64 CuPy/PyTorch controls on a constructed asymmetric family, not native cuGraph. Its source is an experimental implementation outside the installed package API. Derived serial rate is based on reciprocal median latency, not measured concurrent throughput. These studies establish scoped measured advantages; they do not establish global leadership or customer savings.

![Audited native comparison coverage: 59 of 89 attempted cases pass the strong speed gate, 26 also pass both error comparisons, four cases fail K accuracy confirmation](assets/native_win_coverage.png)

![Six constructed Tesla T4 workloads: about 20.0–73.3% lower median compute time and 25.0–273.9% higher derived serial rate versus tuned FP64 CuPy or PyTorch](assets/precision_percent_comparison.png)

Inspect [all results](benchmarks/results.md), including Graph500, repeated-reference and complete-workflow losses. The original one-run result was 21.28% less median compute time (63.398 ms native / 49.906 ms K; 1.2703x), equivalent to 27.03% higher derived serial rate; the older zero-pass 19-workload study remains available with its original contract.

## Start with your task

| Task | Documentation |
|---|---|
| Install and check a small CPU example | [Installation](getting-started/install.md), [first graph filter](getting-started/quickstart.md) |
| Keep a prepared sparse operator on a GPU | [CUDA guide](guides/gpu.md) |
| Differentiate graph weights along one direction | [Sensitivity guide](guides/sensitivity.md) |
| Apply a symmetric restarted resolvent | [Resolvent guide](guides/resolvent.md) |
| Evaluate a real workflow | [Application fit](guides/applications.md), [limitations](reference/limitations.md) |
| Read the research and interpret benchmark units | [Research paper](research/paper.md), [metric definitions](benchmarks/metric-definitions.md) |
| Inspect signatures and output contracts | [Python API](reference/api.md) |
| Audit or rerun the experiment | [Methodology](benchmarks/methodology.md), [reproduction](benchmarks/reproduce.md) |

## Release status

Version **0.1.0** is a research and evaluation release. CPU usage requires Python 3.11+, NumPy and SciPy. GPU usage additionally requires a compatible CUDA-enabled PyTorch installation. The numerical core preserves its historical MIT license and original C2Fuse imports. New documentation and package infrastructure use Apache 2.0. [Release notes](project/releases.md) explain the naming and compatibility boundaries.
