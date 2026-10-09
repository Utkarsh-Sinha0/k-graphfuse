# Limitations and support

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

The same-T4 reference result is a scoped competitive advantage. It does not establish wider uniqueness, global SOTA, universal precision improvement, database-service replacement, revenue or bill savings. The earlier complete suite passed zero of 19 strongest GPU improvement gates. Best-candidate selection and within-run intervals do not replace independent runs.

Sampled memory may miss peaks and includes inherited/library allocations. GPU-preparation measurements exclude CPU normalization/compiler work. Sampled average board power was higher; integrated energy was not measured. Customer projections remain models until measured.

## Operational limits

Plans are not a secure boundary against hostile in-process mutation. One CUDA executor cannot be shared concurrently. Arbitrary graph changes require a new plan. Benchmark runners download vendor artifacts and datasets and use a disposable Linux workspace; they are manual research fixtures and do not execute during package installation.

Read the [accuracy contract](../concepts/contracts.md), [benchmark methodology](../benchmarks/methodology.md) and [application gates](../guides/applications.md) before production evaluation.
