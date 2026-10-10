# Research manuscript

K GraphFuse research manuscript files:

- [Markdown manuscript](../docs/research/paper.md) — readable source with links to the generated figures and metric definitions.
- [Rendered PDF](../docs/research/k-graphfuse-paper.pdf) — linked paginated copy.
- [Standalone LaTeX source](k-graphfuse-paper.tex) — single-file article with inline references and no external figure dependency.
- [Public standalone LaTeX source](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/research/k-graphfuse-paper.tex) — canonical repository URL.

The manuscript is authored by Utkarsh Sinha and dated 10 October 2026. Its status is an unreviewed technical manuscript / preprint draft. It reports six constructed Tesla T4 polynomial cases against FP64 CuPy/PyTorch controls, separately from the native-cuGraph PageRank study.

The similarity-transfer inequality is proved in the manuscript. The complete Crouzeix constant-two result remains an explicitly stated premise from pinned OpenAI preprint 325-01. Benchmark limitations, source versions, confidence-interval estimators, sampled high-precision scope, and separate PageRank losses are recorded in the paper.

The standalone LaTeX source compiled successfully with the Codex desktop editor's compiler on 10 October 2026; [TEX_COMPILATION.json](TEX_COMPILATION.json) pins the checked source. The distributed PDF is independently rendered from Markdown with three figures; [PDF_RENDERING.md](PDF_RENDERING.md) records its inputs, hashes and layout checks. Compilation and artifact validation are engineering checks, not mathematical peer review.
