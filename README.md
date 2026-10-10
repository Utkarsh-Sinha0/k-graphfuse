# K GraphFuse

**Reusable sparse graph filters, directional sensitivities, and optional GPU execution.**

**Created and maintained by [Utkarsh Sinha](https://github.com/Utkarsh-Sinha0).**
If this project supports your product or research, please credit **K GraphFuse by Utkarsh Sinha** and [cite the software](CITATION.cff).

[![CI](https://github.com/Utkarsh-Sinha0/k-graphfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/Utkarsh-Sinha0/k-graphfuse/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-technical_reference-2e7d32)](https://utkarsh-sinha0.github.io/k-graphfuse/)
[![License](https://img.shields.io/badge/license-Apache_2.0_%2B_MIT-blue)](LICENSE)

[Documentation](https://utkarsh-sinha0.github.io/k-graphfuse/) · [Quickstart](docs/getting-started/quickstart.md) · [API reference](docs/reference/api.md) · [Benchmark evidence](docs/benchmarks/results.md) · [Reproduce](docs/benchmarks/reproduce.md)

K GraphFuse compiles compatible linear graph stages into one ordered matrix polynomial, then evaluates it with sparse Horner recurrences. Reuse a copied operator and compiled plan across inputs. Optional backends cover CUDA execution, symmetric restarted resolvents, sparse local propagation, and derivatives with respect to one affine graph parameter.

It is useful when repeated graph ranking or linear propagation consumes a meaningful part of a workload. It requires matching graph semantics, independent accuracy checks, and enough repeated calls to recover preparation costs. Nonlinear graph neural networks, shortest paths, general graph transactions, and arbitrary dynamic topology updates are separate tasks.

## Audited benchmark evidence

| Study | Measured result | Comparison contract |
|---|---|---|
| Native NVIDIA cuGraph ranking study | **59/89 strong speed wins**, including **26/89 speed-and-error wins** | 85 comparable cases and four K accuracy failures |
| Strongest tested alternatives | **33/85 overall**, **38/85 GPU** speed-gate wins | Fastest accepted recorded control, including CPU where measured |
| Experimental precision GPU filters | **6/6** combined speed/accuracy gates; **1.25–3.74x** | Equivalent tuned FP64 CuPy/PyTorch; constructed sparse filters and derivatives |

Strong native speed wins require at least 10% lower median, an interval above one, and no p95 regression. The 26 speed-and-error wins also have no worse recorded L1 and maximum-relative error; all 26 use matching precision. These counts do not establish better energy, memory or complete application runtime on every case.

[Native cuGraph metrics and all losses](docs/benchmarks/cugraph.md) · [Precision GPU latency, p95 and high-precision accuracy](docs/benchmarks/precision-gpu.md) · [Measurement contracts](docs/benchmarks/methodology.md)

![K GraphFuse precision GPU comparison: six constructed T4 workloads, 1.25–3.74x speedups, explicit sampled-accuracy and timing scope](docs/assets/precision_t4_comparison.png)

The original same-T4 `datagen-7_5-fb` experiment recorded 21.28% lower median compute and lower sampled total memory peaks, but aggregate L1 error was worse. A later longer confirmation on that graph favors cuGraph. Native cuGraph is 2.72x faster on Graph500-22, and complete file/rebuild workflows also favor native. The older 19-workload suite passed zero strongest GPU gates under its separate protocol. [Historical reference result](docs/benchmarks/historical-reference.md).

The new fused implementation lives in [experiments/precision_gpu](experiments/precision_gpu/README.md), separate from the installed `k_graphfuse` 0.1.0 API. Its repaired checked API passed 43 combined CPU/GPU tests; performance belongs to separately frozen timed versions. Raw samples, rejected configurations, code hashes and offline audits are included. Global SOTA, customer savings and integrated energy savings remain unestablished.

## Install and run

Python 3.11 or newer; CPU use requires NumPy and SciPy. Clone the repository and install locally:

```sh
git clone https://github.com/Utkarsh-Sinha0/k-graphfuse.git
cd k-graphfuse
python -m pip install .
python examples/quickstart.py
```

```python
import numpy as np
from scipy.sparse import csr_array
from k_graphfuse import compile_sparse, evaluate_sparse

a = csr_array([[0., 0.5], [0.25, 0.]])
x = np.array([[1.], [2.]])
stages = [[[[1]], [[1]]], [[[2]], [[1]]]]  # (I + A), then (2I + A)
plan = compile_sparse(a, stages, tolerance=0)
y = evaluate_sparse(plan, x)
```

For GPU use, install a CUDA-enabled PyTorch build compatible with your driver, then `python -m pip install ".[gpu]"`. Follow the [GPU guide](docs/guides/gpu.md). FP64 is the default; FP32 requires workload-specific acceptance. CUDA capture and packing can be slower or use more memory.

## Choose your path

| Need | Start here |
|---|---|
| Evaluate a linear graph-filter pipeline | [Quickstart](docs/getting-started/quickstart.md) |
| Integrate a reused CUDA operator | [GPU execution](docs/guides/gpu.md) |
| Differentiate a graph parameter | [Sensitivities](docs/guides/sensitivity.md) |
| Compute a symmetric restarted resolvent | [Resolvents and residual audit](docs/guides/resolvent.md) |
| Check a performance claim | [Results](docs/benchmarks/results.md), [methodology](docs/benchmarks/methodology.md), [raw evidence](evidence/README.md) |
| Assess ranking, fraud or retrieval fit | [Applications](docs/guides/applications.md) |
| Navigate with an AI coding agent | [AGENTS.md](AGENTS.md), [llms.txt](docs/llms.txt), [claims.json](docs/claims.json) |

## Verify and contribute

```sh
python -m unittest discover -s tests -v
python benchmarks/verify_evidence.py
python examples/sensitivity.py
python examples/resolvent.py
python -m pip install ".[docs]"
python -m mkdocs build --strict
```

CUDA tests skip when no CUDA device is available. Offline evidence verification recomputes reported timings and source hashes; it does not rerun large GPU experiments. See [contribution checks](CONTRIBUTING.md) and [support boundaries](docs/reference/limitations.md).

## License and citation

New packaging and documentation use **Apache 2.0**; the historical numerical modules retain their **MIT** license. Preserve the applicable licenses and notices when redistributing. Requested visible credit: **“Powered by K GraphFuse by Utkarsh Sinha”**, linked to this repository. The requested citation does not add a mandatory UI-attribution restriction. [NOTICE](NOTICE) · [Citation and reuse](docs/project/citation.md) · [CITATION.cff](CITATION.cff).
