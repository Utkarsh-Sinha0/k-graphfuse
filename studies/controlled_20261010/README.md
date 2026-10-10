# Controlled K GraphFuse study

This separate study uses the repaired implementation, one original-coordinate error budget, explicit coordinate/executor/compensation ablations, fresh feature seeds and a fixed application protocol. It does not replace historical measurements. Read `protocol.json` before execution. All outcomes, errors and losing methods are retained.

`FREEZE.json` pins the runner, protocol and reused numerical sources before confirmation. Two separate processes are required on each device:

```sh
python studies/controlled_20261010/study.py --selftest
python studies/controlled_20261010/study.py --seed 20261011 --output /your/results/run1.json --cache /your/data-cache
python studies/controlled_20261010/study.py --seed 20261012 --output /your/results/run2.json --cache /your/data-cache
```

The runtime needs NumPy, SciPy, Mpmath, CuPy, CUDA-enabled PyTorch, threadpoolctl and certifi. It uses the experimental root and its frozen core, without changing the installed API. A GPU and network access for the initial official MovieLens download are required. Archive identity is checked before parsing. Raw ratings and demographics are not redistributed; only aggregate study outcomes are published. MovieLens is a research demonstration under its own terms, independently of the software license.

The primary resident profile and comparator set are fixed in the protocol. Named same-backend pairs isolate effects; their results must not be pooled as independent algorithm victories. The iterated balancing control uses eight simultaneous dyadic sweeps of the existing update. It is an Osborne-inspired sparse control, not LAPACK DGEBAL or a claim to reproduce its sequential algorithm. Capture is tested with supported PyTorch controls; CuPy 14.2 cuBLAS capture is unavailable.

Large outputs use full FP64 CPU reference checks. The separate small-fixture check uses complete 64-row 256/384-bit references across six families. Neither supports an all-input floating-roundoff theorem. Application timing includes request seed construction, transfer, checked execution, host output, seen-item exclusion and ranking. Load/build, compile and GPU preparation are separate recorded costs. The model is a fixed finite graph-diffusion recommender; its quality metrics do not claim learned recommender state of the art.
