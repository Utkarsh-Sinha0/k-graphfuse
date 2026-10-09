# Troubleshoot common integration issues

| Symptom | Likely cause | Action |
|---|---|---|
| `torch.cuda.is_available()` is false | CPU-only PyTorch, incompatible driver or unavailable device | Use the official PyTorch selector and confirm device access before importing the GPU API |
| CUDA input validation fails | Shape, dtype or device differs from preparation | Reuse the example shape and dtype or create a new executor |
| Nonfinite output raises `ArithmeticError` | Operator growth, coefficient size or arithmetic overflow | Check normalization and input magnitude; validate in FP64 and compare a small independent case |
| `compile_ppr` rejects the graph | Stored operator is not exactly symmetric or the resolvent enclosure fails | Check semantics and normalization; do not silently symmetrize a directed task |
| A parameter endpoint is rejected | Binary float lies outside an exact compiled interval | Use a consistent rational endpoint or an interior parameter value |
| Approximation is no faster | Full degree retained or launch/transfer overhead dominates | Inspect selected degree, bound method, preparation and complete latency |
| Captured execution is slower | Small task, capture overhead or workspace pressure | Measure eager `capture=False` and unpacked execution |
| Local propagation expands widely | The source reaches much of the graph within the degree | Account for graph locality; the backend can fall back to global evaluation |
| CPU tests pass but GPU tests skip | No CUDA device is accessible | Report the skips and run the GPU tests on a real supported device |

## Report a reproducible issue

Include the package version or commit, Python/library versions, graph dimensions and stored entry count, selected degree, dtype, settings, minimal nonprivate input, expected result and actual error. For performance issues, include all synchronized samples and full timing boundaries. Do not include credentials, private customer data or system paths containing personal information.

Use [GitHub issues](https://github.com/Utkarsh-Sinha0/k-graphfuse/issues) for ordinary bugs. Follow [SECURITY.md](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/SECURITY.md) for security-sensitive reports.
