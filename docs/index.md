---
title: K GraphFuse — sparse GPU graph analytics and polynomial filters
description: Compile reusable graph polynomial plans, evaluate directional sensitivities, and inspect matched PageRank benchmarks against native NVIDIA cuGraph.
---
# K GraphFuse

**Compile compatible linear graph operations once. Reuse sparse CPU or CUDA execution across inputs.**

**Created by [Utkarsh Sinha](https://github.com/Utkarsh-Sinha0).** [Cite K GraphFuse](project/citation.md) when using its implementation or evidence.

K GraphFuse turns a sequence of compatible graph filters into an ordered matrix polynomial. It evaluates the plan without forming graph matrix powers or a dense graph operator. The package also provides a one-parameter directional derivative, a symmetric Chebyshev resolvent, local sparse propagation and an exact rational residual audit.

The intended opportunity is repeated graph computation on a reused operator. Compilation, graph conversion and GPU preparation are additional costs. Matching a mathematical operation alone does not establish a production speedup: include downstream quality, transfers, projection, updates and writeback in your evaluation.

## What the measurement shows

On one public ten-update PageRank reference task, the best accepted configuration used **21.28% less median warm compute time than native NVIDIA cuGraph on the same Tesla T4**. Observed total memory peaks were lower and both outputs passed the reference accuracy check. Aggregate L1 error was higher. The benchmark used the full polynomial with zero truncation; it does not show a theorem-driven degree reduction.

![Same-device PageRank comparison, including measured latency, sampled memory and accuracy tradeoffs](assets/k_native_comparison.png)

Read the [results](benchmarks/results.md) before reusing a number. The earlier 19-workload suite did not pass its strongest GPU improvement gate on any complete workload. Independent replication, global SOTA, customer quality and cloud-bill savings remain open.

## Start with your task

| Task | Documentation |
|---|---|
| Install and check a small CPU example | [Installation](getting-started/install.md), [first graph filter](getting-started/quickstart.md) |
| Keep a prepared sparse operator on a GPU | [CUDA guide](guides/gpu.md) |
| Differentiate graph weights along one direction | [Sensitivity guide](guides/sensitivity.md) |
| Apply a symmetric restarted resolvent | [Resolvent guide](guides/resolvent.md) |
| Evaluate a real workflow | [Application fit](guides/applications.md), [limitations](reference/limitations.md) |
| Inspect signatures and output contracts | [Python API](reference/api.md) |
| Audit or rerun the experiment | [Methodology](benchmarks/methodology.md), [reproduction](benchmarks/reproduce.md) |

## Release status

Version **0.1.0** is a research and evaluation release. CPU usage requires Python 3.11+, NumPy and SciPy. GPU usage additionally requires a compatible CUDA-enabled PyTorch installation. The numerical core preserves its historical MIT license and original C2Fuse imports. New documentation and package infrastructure use Apache 2.0. [Release notes](project/releases.md) explain the naming and compatibility boundaries.
