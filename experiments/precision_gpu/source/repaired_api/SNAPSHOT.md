# Final checked API snapshot — 10 October 2026

These files match the current root API and tests at packaging. Use the complete package root to execute them; this snapshot is a source record, not a standalone installation.

Compared with the frozen timed versions, the finite-validation reduction avoids optional CUB headers and projected default-stream execution explicitly orders CuPy/PyTorch with a named stream and events. `FINAL_LOCAL_TESTS.json` records the combined 29 core and 14 upgrade checks, without skipped tests, and source hashes.

The large recorded speedups belong to `source/product_v1` and `source/wide_v1`. The repaired API was regression-tested but did not receive a new full latency run. All versions remain separate so evidence cannot silently migrate to different code.
