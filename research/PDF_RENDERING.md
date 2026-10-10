# PDF rendering record

`docs/research/k-graphfuse-paper.pdf` is rendered from `docs/research/paper.md` by `benchmarks/render_research_paper.py` with Python-Markdown 3.10.2, Matplotlib 3.10.3 MathText, ReportLab 4.5.1, and the bundled DejaVu fonts. It is an independent Markdown typesetting of the same technical manuscript; it is not compiled from the standalone LaTeX source.

- Source SHA-256: `9410e51d65e651313a7017e3c113226ec28409cf22edcee1d802e2fea5286d52`
- PDF SHA-256: `4f811f3c0cf6da1d753d1d7a40f1418960a4d0f4d474634e875430628deb707a`
- PDF pages: 10
- Reference link annotations: 12
- Relative Markdown links use the public site URL from `mkdocs.yml`:
  - `k-graphfuse-paper.pdf` → `https://utkarsh-sinha0.github.io/k-graphfuse/research/k-graphfuse-paper.pdf`
  - `../benchmarks/metric-definitions.md` → `https://utkarsh-sinha0.github.io/k-graphfuse/benchmarks/metric-definitions/`
- Initial YAML frontmatter is omitted from page content and retained in the source hash.
- Result table: 6 data rows; all row cells verified in extracted PDF text
- Equations rendered with MathText: 134; monospaced fallback formulas: 1
- Fallback formulas (whitespace compacted here; exact source formulas remain visible in monospaced PDF text): `\square`
- Renderer font resources: DejaVu Serif, DejaVu Sans, DejaVu Sans Mono from Matplotlib's installed font bundle

Included figure inputs:
- `docs/assets/precision_percent_comparison.png` — SHA-256 `fd4b626de368627f3d2df30c30f1ebbc872d74a4028ad277a2d0bd5e7f36c444`
- `docs/assets/native_win_coverage.png` — SHA-256 `9b6dc4638979cf4fcb403aac6409d86f93efd4362241e44d0984f55b4cbeee35`
- `docs/assets/graphfuse_pipeline.png` — SHA-256 `1efc5a8b531382eebcaf74912dcd1d9af9d27abec7cca032a530759b1fba1d20`

Run from any working directory with `python benchmarks/render_research_paper.py` using the repository's installed work environment. The script reads the Markdown and local PNG assets only, rejects missing or out-of-repository figures, checks the six result rows and scope/status text in the finished PDF, and writes this provenance record. ReportLab's invariant PDF mode makes metadata timestamps stable; the manuscript's own date remains visible in the document.
