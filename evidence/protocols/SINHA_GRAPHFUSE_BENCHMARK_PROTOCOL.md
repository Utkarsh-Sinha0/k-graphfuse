# Sinha GraphFuse: SOTA is the minimum acceptance bar

Project name: **Sinha GraphFuse**. Existing `c2fuse` module names remain compatible. The scope is certified finite linear graph filtering, one-parameter sensitivities and a separately scoped symmetric PageRank resolvent. Naming does not claim invention of the established graph filtering, Chebyshev, block derivative or CUDA techniques it uses.

This protocol was saved before the large Kaggle run. It prevents a favorable synthetic case, a CPU-to-GPU comparison or an unmatched published number from being called a SOTA result.

## Acceptance and rejection

1. **Correctness first:** all 27 implementation tests must pass on the actual Kaggle GPU with zero skips. Candidates must satisfy the same declared numerical metric and threshold; failed precision candidates remain in the raw evidence and cannot be speedup denominators.
2. **Strongest eligible measured baseline:** compare against the fastest accuracy-accepted CPU and GPU methods on the same operator, features, output contract and hardware. Full-degree optimized PyTorch controls, the older complete Crouzeix bound with the same optimized GPU executor, CuPy/cuSPARSE full and retained-degree controls, an independently implemented CuPy Chebyshev control with optional CUDA capture, independent SciPy CG and the actual NVIDIA `pylibcugraph` PageRank library are included where their mathematics matches. An optimized established Chebyshev control uses the same plan/executor because this resolvent algorithm predates the current project. A tie with it is meaningful evidence against a new algorithmic speedup claim. Retained-degree and Chebyshev controls reuse our plan and isolate execution benefits; they are not execution of every author implementation. The large run installs the latest verified stable `pylibcugraph-cu12` release, 26.10.0, in its disposable Kaggle environment rather than using the image's older 26.02 library.
3. **A useful measured improvement:** require at least 10% lower median latency and a bootstrap 95% confidence interval for the speedup entirely above 1. Report p95 separately. A win against a weaker CPU implementation alone does not clear the GPU bar.
4. **Complete economic picture:** include compilation, preparation/upload, first call, resident API latency, host round trips, throughput, allocator peaks, sampled total process/device memory and reuse break-even calls. State any additional memory or setup cost. No actual cloud cost or energy saving is inferred from latency alone.
5. **Honest scope:** report every completed case, rejection, failure and loss. Public graph topology plus constructed coefficients/features is a numerical kernel workload, not a production fraud or recommendation task. A SOTA claim stays **unproven** if a relevant stronger implementation is unavailable, a hardware requirement is unmet, or the task/error contract differs.

Even a scoped timing pass is not a universal SOTA certificate. Reliability tests here cover numerical and API behavior, not production uptime, distributed failover or an adversarial security assessment. The final report must distinguish measured baseline wins from the remaining SOTA validation gates.

## Workload matrix

Public SNAP graph families: ca-GrQc collaboration, DBLP collaboration, YouTube social connections, California road topology and directed Google web links. The California graph has approximately 1.97 million published vertices; YouTube approximately 1.13 million. Exact processed counts and compressed-file SHA-256 hashes are recorded by the run. Road topology is used for linear filtering, not shortest paths. [SNAP dataset collection](https://snap.stanford.edu/data/index.html).

The matrix includes multi-feature finite filters, matched `1e-4`, `1e-8` and `1e-10` polynomial precision on DBLP, symmetric personalized PageRank at two teleportation/accuracy settings, synthetic directed graphs with 1 million and 5 million vertices, a weak-tail case expected to offer no useful truncation, and a million-vertex value-and-derivative case. Public filters and feature values are constructed and reproducibly seeded, not customer models. Isolated PPR vertices retain probability through an explicitly recorded self-loop convention shared by both libraries.

Polynomial quality is error divided by input Frobenius norm, checked against full-degree SciPy/NumPy FP64 evaluation. Sensitivity endpoints receive additional checks. PPR quality uses L1 error against an independent CG solve and a floating achieved-L1 residual audit; top-100 overlap and probability mass are also reported. Those residual computations are not exact-rational or complete roundoff proofs.

PPR timings target the symmetric-coordinate resolvent output; the NVIDIA adapter reorders PageRank scores and converts them to that same coordinate system inside the timed call. Original probability scores are audited outside timing. A complete original-score customer pipeline must include Sinha's final coordinate conversion. Repeated sources are fixed, NVIDIA's sparse personalization seeds are cached, and Sinha's host path transfers its dense feature input. A source-only Sinha frontend was not built or measured.

GPU measurements use synchronized wall time and 31 samples; CPU measurements use 9 samples. Transfer measurements use 11 samples. Configuration order for finite GPU filters is seeded and shuffled. These are one-session repeated observations; bootstrap intervals describe observed variability, not variation across machines or production deployments. Total-memory/power sampling is every 20 ms and can miss short peaks; allocator peaks are reported separately. Board power samples are not whole-system energy measurements.

The run uses GPU 0 only, even if Kaggle supplies two devices. Static capture/workspace configurations estimated to exceed 65% of that device's VRAM are recorded as not run. This same rule applies to Sinha and full-degree controls. It is a predictable resource guard, not a demonstrated out-of-memory result.

## Published metrics and strongest relevant competitors

The final comparison ledger includes primary sources for NVIDIA's production PageRank metrics, the 2024 ChebyPower/ChebyPush work, Acc-SpMM (2025) and AsyncSparse (2026). Their published hardware, precision and output contract must accompany their performance figures. External numbers are background and cannot be divided into Kaggle timings to manufacture a speedup.

The [official NVIDIA PageRank interface](https://docs.nvidia.com/cugraph/latest/api_docs/api/cugraph/cugraph.pagerank/) and [lower-level personalized PageRank API](https://docs.nvidia.com/cugraph/26.10/api_docs/api/plc/pylibcugraph.personalized_pagerank/) identify an actual production comparator. We use cached transposed graph storage and precomputed out-weight sums, with the lower-level API to avoid unnecessary dataframe wrappers.

The [official Acc-SpMM implementation](https://github.com/Hyaloid/AccSpMM) lists A800, H100 and RTX 4090 as supported GPUs. [AsyncSparse](https://arxiv.org/html/2604.17834v1) evaluates Hopper-specific mechanisms on H100. Kaggle's verified preflight supplies Tesla T4 GPUs; that is an outstanding hardware gate for those comparisons. [ChebyPush's paper](https://arxiv.org/html/2412.10789v1) benchmarks optimized C++ local-query algorithms; our existing Python ChebyPower-style recurrence must not be labeled as execution of that author code.

All Kaggle submissions in this task are private. Account credentials stay on the user's machine and are not embedded in code or benchmark outputs. Preparing an open-source evidence bundle does not publish it to a public repository.
