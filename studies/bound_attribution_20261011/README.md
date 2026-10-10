# Theorem-bound attribution analysis

This 11 October 2026 analysis extracts existing 10 October Tesla T4 measurements. It creates no new GPU timing, changes no numerical code and preserves every historical raw record. It is separate from the frozen controlled-study confirmation.

`extract.py` decodes the immutable run sources without executing them, checks source manifests, matches all accepted FP64 packing/capture pairs, recomputes raw medians/p95, and records the laptop equal-degree cases and untimed CPU boundary. Case identity includes the error budget and input/operator hashes because several fixtures share a display name.

```sh
python studies/bound_attribution_20261011/extract.py --verify
python studies/bound_attribution_20261011/figure.py
```

`SUMMARY.json` contains eight observed timing pairs in two constructed nonnormal workloads. `cpu_sparse_original.json` preserves the original CPU sparse record, including the 9→8 degree-only boundary. `MANIFEST.json` pins the analysis inputs/output; `VISUALS.json` pins the reproducible PNG/SVG figure. The main repository audit invokes the verification command without GPU or network access.

The corresponding [research report](../../docs/research/bound-attribution.md) distinguishes a supplied OpenAI theorem, a derived computational application and unestablished algorithmic priority. Certificates cover exact-arithmetic truncation, not execution rounding. All eight differences are below the separate 10% strong-speed threshold; no new native-cuGraph win is added. No exact degree-9-versus-8 GPU time or isolated laptop degree-change benefit is asserted.
