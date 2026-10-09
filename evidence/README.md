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
