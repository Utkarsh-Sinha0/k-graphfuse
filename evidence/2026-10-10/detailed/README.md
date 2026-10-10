# K GraphFuse detailed benchmark evidence

Start with [RESULTS.md](RESULTS.md). The report counts downloaded, audited measurements and distinguishes complete suites, partial checkpoints and jobs still running. The original release benchmark is preserved separately.

## Files and measurement contracts

| File | Purpose |
|---|---|
| `verification_summary.json` | Current execution state, verified source identities, test counts and coverage totals |
| `declared_case_coverage.csv` | Every predeclared ranking/filter case, including cases with no result or no accuracy-accepted K configuration |
| `case_coverage.csv` | All recorded cases, including workflow diagnostics and native-only calculations |
| `confirmed_comparisons.csv` | Held-out ranking comparisons, errors, median/p95/p99, host-output cost and block intervals |
| `all_candidates.csv` | Accepted and rejected configurations, exceptions, preparation and observed resource samples |
| `cold_rebuild_diagnostics.csv` | Complete jobs from loaded adjacency using configurations selected for warm computation |
| `workflow_diagnostics.csv` | Changing graphs and import-to-saved-score workflows; separate fixed-configuration protocols |
| `matrix_diagnostics.csv` | Channel polynomials and derivatives against measured full/retained/older-bound controls |
| `native_only_baselines.csv` | Algorithms that K GraphFuse does not implement; no K speedup is assigned |
| `figures/` | PNG and SVG charts with losses, uncertainty and missing accuracy-accepted results visible |
| `raw/<suite>/` | Original result JSON, 29-test record, frozen profile, source manifest and standalone runner |

Diagnostic files appear once the corresponding suite has supplied audited results. Their absence means no such measured result has been published in this package.

## Read a comparison correctly

The speed ratio is `comparator median / K GraphFuse median`. A ratio of 2 means half the measured K compute time. A ratio of 0.5 means twice the K compute time. Compare only methods solving the same recorded task and passing its declared accuracy gates.

Ranking uses five confirmation blocks of 31 samples per selected backend, following a separate nine-sample configuration pilot. The interval resamples the five paired blocks 5,000 times. It describes uncertainty within the observed worker, not hardware diversity or worldwide performance. A 155-sample p99 has limited tail resolution; production p99.9 behavior has not been measured.

The scoped performance bar requires at least 10% lower median latency, an interval entirely above ratio 1, and no higher observed p95. Passing this bar does not establish global SOTA. `strongest_tested_control` includes the CPU control where measured; `strongest_tested_gpu_control` considers GPU controls only. CPU comparisons are deliberately unavailable above the predeclared CPU size limit.

Timing synchronizes completed calls and includes the host overhead of invoking the implementation. Resident computation, first call, preparation, full host output and complete rebuilding are different contracts. Native host-output timing includes identifier alignment into reference order. Cold rebuilding uses warm-selected configurations and is not a search for the fastest cold implementation.

Matrix and derivative diagnostics use 31 GPU samples per configuration and select the best accepted measured configuration without independent held-out blocks. They are exploratory comparisons against the included implementations, not equivalent cuGraph PageRank workloads. Their error measure is normalized by input Frobenius norm. Historical frozen source labels beginning `Sinha GraphFuse` refer to the numerical implementation now packaged as K GraphFuse.

Observed memory is sampled total process/device use, with inherited and library allocations included. Ranking samples every 5 ms; the reused matrix harness samples every 20 ms. PyTorch allocator statistics omit other libraries. Peak board power is not query energy, and measured time or memory does not prove customer cloud-bill savings.

## Reproduce a suite

Use the frozen `raw/<suite>/run.py` in a separate Kaggle Python script notebook with GPU acceleration and internet enabled. The runner contains its exact profile and numerical/test source; it installs the pinned environment, checks dependencies, runs 29 tests with zero skips, then writes results under `/kaggle/working/kgf_detailed_results/`.

The tested environment is Linux, Python 3.13, Tesla T4, PyTorch 2.11.0+cu128, CuPy 14.0.1 and pylibcugraph 26.10.0. Environment details and source/data/input hashes are in each result. Installation is outside timed query boundaries. Account GPU limits control how many separate notebooks can execute together.

Private Kaggle metadata, API credentials and signed download links are excluded from this evidence package. Public datasets are referenced by URLs and hashes; their original archives are not republished here.

Run `python verify.py` with NumPy installed for an offline consistency audit of source payloads, profiles, regression counts, raw timing statistics and held-out comparison calculations. It makes no network calls and requires no GPU. This verifies the recorded evidence; it does not rerun numerical calculations or independently attest that an original cloud execution occurred.

## Known coverage boundaries

The frozen generator called `chain` creates a directed cycle by connecting the last vertex back to zero. It is not an open-ended path. Dynamic rebuilding measures 31 updates; an older raw scope string saying five is a labeling error. Native-only community validation checks structural coverage and finite modularity, not optimal community quality.

The new sparse dangling-state adapter belongs to benchmark integration; it is not a newly released API. Traversal, weighted shortest paths, components, triangles and community detection remain unimplemented in K GraphFuse. Genuine distributed/out-of-core execution, Hopper-only implementations, production availability, downstream customer quality, energy and actual costs remain unmeasured.

The full-polynomial degree is chosen from the aggregate L1 tail rule; it is not separately swept to enforce the per-vertex relative gate. A rejected tested degree does not prove that every degree or configuration fails. Likewise, the included controls are a finite candidate set, not every solver or custom CUDA implementation. Repeated timing samples reuse the same input; varying-input service throughput and the benefit of application-level result caching are not measured.
