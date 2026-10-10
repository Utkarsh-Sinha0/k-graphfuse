# Published evidence

The records retain the measured **Sinha GraphFuse** identity. K GraphFuse uses the same eight core modules. Generated CPU wrapper scores are included; third-party graph/reference data and vendor binaries are acquired separately.

| File/folder | Role |
|---|---|
| `SINHA_GRAPHFUSE_PUBLIC_REFERENCE_RAW.json` | Complete native-cuGraph / released-CPU task record and samples |
| `SINHA_GRAPHFUSE_PUBLIC_REFERENCE_COMPARISONS.csv` | Recomputed scoped comparisons |
| `SINHA_GRAPHFUSE_PUBLIC_REFERENCE_ALL_CANDIDATES.csv` | Complete public-task candidate table |
| `sinha_public_reference_evidence/` | Frozen bootstrap, decoded modules, CPU scores/timings, source hashes |
| `SINHA_GRAPHFUSE_KAGGLE_RAW.json` | Earlier three-run aggregate with private notebook-account URLs removed |
| `sinha_benchmark_evidence/` | Unchanged run snapshots, complete candidate records and original failures |
| `SINHA_GRAPHFUSE_COMPARISONS.csv` | Strongest accepted GPU control selection and zero-pass verdict |
| `SINHA_GRAPHFUSE_ALL_CANDIDATES.csv` | Earlier suite's full candidate catalogue |
| `SINHA_GRAPHFUSE_PUBLIC_INDUSTRY_METRICS.csv` | Public publisher metrics, lifecycle and hardware qualifications |
| `protocols/` | Original saved task and acceptance protocols |
| `MANIFEST.json` | SHA-256 and size for every published evidence file |
| `PUBLICATION_TRANSFORMATIONS.json` | Public-copy exclusions and redactions |

Run `python benchmarks/verify_evidence.py` from the repository root. This verifies saved facts, not independent performance replication. Use `--reference` with the separately acquired official reference for the complete CPU-output audit. Historical scripts are manual Linux/Kaggle fixtures that download external artifacts; package installation never runs them.

All benchmark publication is self-reported. It is not an independently certified Graphalytics submission. Negative results and selected-configuration limitations remain visible.

## 10 October 2026 extended evidence

| Location | Contents |
|---|---|
| `2026-10-10/comparison_summary.json` | Recomputed native counts, attempted/comparable denominators, strongest-control counts and source hashes |
| `2026-10-10/native_comparisons.csv` | All 85 direct native comparisons, including losses and both reference error measures |
| `2026-10-10/K_accuracy_failures.json` | Four additional cases with accepted native controls and no accepted K confirmation |
| `2026-10-10/detailed/` | All 15 complete suites, 111 case records, 2,190 configurations, diagnostics, frozen bootstraps and source manifests |
| `../experiments/precision_gpu/` | Separate experimental source and numerical evidence: six selected T4 speed/accuracy gates against FP64 CuPy/PyTorch |

The broad native study yields 59/89 strong speed wins and 26/89 speed-and-error wins when four accuracy failures count as non-wins. Its strongest-control count is different: 33/85 overall and 38/85 GPU. The precision experiment is not six more native-cuGraph wins. Exact timed implementations and the later checked-API repair remain separate.

Run the same repository audit command to verify these additions. The precision source package has its own file manifest. Private notebook-account metadata, installation/kernel logs and caches are excluded; the publication ledger records the derived audit/documentation changes. Raw numerical samples and historical timed-source bytes are preserved.
