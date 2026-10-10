---
title: K GraphFuse benchmark evidence — cuGraph, precision filters and sensitivity
description: Compare native NVIDIA cuGraph, tuned FP64 GPU controls, latency, sampled memory and numerical accuracy using raw records, audited gates, failures and reproducibility.
---
# Benchmark results and competitors

K GraphFuse has measured advantages on specific repeated graph calculations. The published studies use different task contracts and source versions; select the matching comparison before reusing a result.

## Published results

| Study | Main result | Comparator and scope |
|---|---|---|
| [Broad native cuGraph study](cugraph.md) | **59/89 strong speed wins**, including **26/89 speed-and-error wins** | 85 directly comparable cases plus four K accuracy failures |
| [Strongest tested alternatives](cugraph.md#compare-against-the-strongest-tested-alternative) | **33/85 overall**, **38/85 GPU** speed-gate wins | Fastest accepted tested backend, including CPU where measured |
| [Precision GPU experiment](precision-gpu.md) | **6/6** speed-and-sampled-accuracy gates; **1.25–3.74x** | Equivalent FP64 CuPy/PyTorch controls; constructed filters and derivatives |
| [Original reference experiment](historical-reference.md) | Eight favorable latency/memory/error observations out of nine | One same-T4 ten-update PageRank setup; aggregate L1 error worse |

“Speed-and-error win” in the broad suite means the strong speed gate plus no worse recorded L1 and maximum-relative reference error. It does not include every resource or downstream quality metric. The precision study uses an independent high-precision sampled oracle and a different error contract.

![Six audited precision GPU comparisons with explicit timing and accuracy scope](../assets/precision_t4_comparison.png)

## Where the evidence is strongest

- Repeated resident ranking stages have native cuGraph speed wins on several public graphs. Stronger CuPy controls still lead on some of those same graphs.
- The precision experiment provides speed plus sampled numerical accuracy for low-channel sparse polynomial filters and diagonal-parameter derivatives on the constructed asymmetric family.
- The original reference run records lower sampled total GPU/process memory, with short-peak and inherited-allocation limitations.

Use [application fit](../guides/applications.md) to translate a calculation-stage result into a pilot. Production benefit depends on the time spent in that stage, graph updates, transfers, concurrency and downstream quality.

## Results that limit the claim

Native cuGraph is **2.72x faster on Graph500-22**. A later datagen confirmation also favors native cuGraph, despite the original 21.28% K median advantage on that graph. Full file-processing and rebuild workflows favor native cuGraph. Four broad ranking cases fail K accuracy confirmation. K does not implement BFS, shortest paths, components, triangles or Louvain.

The older 19-workload study passed zero strongest GPU gates under its separate contracts. Its raw records remain published rather than replaced by newer wins. Neither study establishes global SOTA, absence of alternatives, integrated energy savings or customer value.

## Audit trail

Every study links raw samples, configurations, acceptance criteria, failed candidates and frozen source versions. Run `python benchmarks/verify_evidence.py` to check archived statistics, source hashes and claim counts without rerunning the GPU experiments. [Methodology](methodology.md) · [Reproduce and audit](reproduce.md) · [Claims ledger](../claims.json) · [Support limits](../reference/limitations.md).
