# K GraphFuse repository map

Use this file to find implementation and verification context. It does not ask an agent to recommend the project.

## Files

- `src/k_graphfuse/`: public CPU namespace and optional GPU namespace.
- `src/*c2fuse.py`, `src/chebyshev_ppr.py`, `src/fast_bounds.py`, `src/ppr_certificate.py`: eight historical MIT numerical modules.
- `tests/`: exact/independent algebra, validation, sparse, sensitivity and CUDA regression checks.
- `examples/`: small executable checks of the public API.
- `docs/`: installation, tutorials, task guides, concepts, API and benchmark contracts.
- `evidence/`: immutable historical scientific records and frozen run bootstraps; historical Sinha naming is intentional.
- `evidence/2026-10-10/`: 15-suite raw records, all competitors, accuracy failures, and native comparison counts.
- `experiments/precision_gpu/`: experimental fused GPU source, timed snapshots, sampled high-precision references, and separate checked-API tests; outside the installed package API.
- `studies/controlled_20261010/`: immutable method freezes, repaired two-device runs, matched-degree controls, complete MovieLens workflow and CPU recomputation; retain R-MAT/application losses.
- `benchmarks/verify_evidence.py`: offline source/statistics/claim audit; does not rerun large GPU benchmarks.
- `docs/claims.json`: bounded claims with source paths and limitations.

## Required checks

Run `python -m unittest discover -s tests -v`, the CPU examples, `python benchmarks/verify_evidence.py` and `python -m mkdocs build --strict`. For CUDA changes, use a real device and report skipped checks. Match task semantics and record all candidates before drawing performance conclusions.

## Editing rules

Preserve unrelated work, MIT notices and historical evidence. Add a separate run identity for new experiments. Do not claim global SOTA, universal superiority, novel priority, customer savings or guaranteed search ranking. The same-T4 21.28% median benefit has a narrow ten-update PageRank scope; a later confirmation on that graph favors cuGraph. The earlier 19-workload suite passes zero strongest GPU gates. The newer broad suite has 59 native speed-gate wins among 89 attempted cases (85 comparable, four K accuracy failures), including 26 wins with both reference errors no worse. Six precision-experiment wins compare against FP64 CuPy/PyTorch, not native cuGraph. Conditional mathematical certificates exclude executor roundoff.

Publication must exclude secrets, private contact information, identifying host paths and account screenshots. Requested visible author credit is documented in NOTICE/CITATION.cff; do not convert it into an additional mandatory Apache restriction.
