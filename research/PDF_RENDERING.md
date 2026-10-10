# PDF rendering record

`docs/research/k-graphfuse-paper.pdf` is rendered from `docs/research/paper.md` by `benchmarks/render_research_paper.py` with Python-Markdown 3.10.2, Matplotlib 3.10.3 MathText, ReportLab 4.5.1, and the bundled DejaVu fonts. It is an independent Markdown typesetting of the same technical manuscript; it is not compiled from the standalone LaTeX source.

- Source SHA-256: `4390103b3d0445693885b90626709c117165c843eab1c9be25c4dc0c417e051a`
- PDF SHA-256: `aeed9fba7e36bfae06b3cf5b19563e2277ce8e2d4e2f7ded35eba14d13407d19`
- PDF pages: 14
- Reference link annotations: 26
- Relative Markdown links use the public site URL from `mkdocs.yml`:
  - `k-graphfuse-paper.pdf` → `https://utkarsh-sinha0.github.io/k-graphfuse/research/k-graphfuse-paper.pdf`
  - `../benchmarks/metric-definitions.md` → `https://utkarsh-sinha0.github.io/k-graphfuse/benchmarks/metric-definitions/`
  - `closest-methods.md` → `https://utkarsh-sinha0.github.io/k-graphfuse/research/closest-methods/`
  - `controlled-study.md` → `https://utkarsh-sinha0.github.io/k-graphfuse/research/controlled-study/`
- Initial YAML frontmatter is omitted from page content and retained in the source hash.
- Result table: 6 data rows; all row cells verified in extracted PDF text
- Equations rendered with MathText: 132; monospaced fallback formulas: 1
- Fallback formulas (whitespace compacted here; exact source formulas remain visible in monospaced PDF text): `\square`
- Renderer font resources: DejaVu Serif, DejaVu Sans, DejaVu Sans Mono from Matplotlib's installed font bundle

Included figure inputs:
- `docs/assets/precision_percent_comparison.png` — SHA-256 `fd4b626de368627f3d2df30c30f1ebbc872d74a4028ad277a2d0bd5e7f36c444`
- `docs/assets/native_win_coverage.png` — SHA-256 `9b6dc4638979cf4fcb403aac6409d86f93efd4362241e44d0984f55b4cbeee35`
- `docs/assets/graphfuse_pipeline.png` — SHA-256 `1efc5a8b531382eebcaf74912dcd1d9af9d27abec7cca032a530759b1fba1d20`
- `docs/assets/controlled_matched_degrees.png` — SHA-256 `ecb8374e18daab9722bf34216c498c7321f26cbc719da668f998c29df588b705`
- `docs/assets/controlled_application.png` — SHA-256 `0bdf6ca2382b74a909594e60c704124c47e4e3a254ad6e14c741f583b48fb9af`

Run from any working directory with `python benchmarks/render_research_paper.py` using the repository's installed work environment. The script reads the Markdown and local PNG assets only, rejects missing or out-of-repository figures, checks the six result rows and scope/status text in the finished PDF, and writes this provenance record. ReportLab's invariant PDF mode makes metadata timestamps stable; the manuscript's own date remains visible in the document.
