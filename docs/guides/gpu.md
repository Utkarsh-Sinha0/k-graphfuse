---
description: Prepare sparse CUDA operators, validate FP64 and FP32 outputs, and reuse K GraphFuse inference executors safely.
---
# Run a prepared graph operator on CUDA

Use CUDA when repeated sparse graph computations are large enough to recover operator preparation and transfer costs. Complete [GPU installation](../getting-started/install.md#enable-cuda) first. This guide starts with FP64 and eager execution.

## Prepare, evaluate and validate

```python
import numpy as np
import torch
from scipy.sparse import csr_array
from k_graphfuse import compile_sparse, evaluate_sparse
from k_graphfuse.gpu import prepare_cuda, CudaExecutor

a = csr_array([[0., 0.5], [0.25, 0.]])
x = np.array([[1.], [2.]])
plan = compile_sparse(a, [[[[1]], [[1]], [[0.5]]]])
prepared = prepare_cuda(plan, dtype=torch.float64, packed=False)
tensor = torch.tensor(x, dtype=torch.float64, device="cuda")
executor = CudaExecutor(prepared, tensor, capture=False)
y = executor(tensor)
np.testing.assert_allclose(y.cpu().numpy(), evaluate_sparse(plan, x), atol=1e-13)
```

Preparation transfers a copied CSR operator and coefficient actions to the CUDA device. Tensor shape, dtype, device and finite values must match the prepared operator. Returned values are tensors; sensitivity plans return `(value, derivative)`.

## Choose precision and launch mode

| Setting | Potential benefit | Required check |
|---|---|---|
| `torch.float64` | Higher arithmetic precision | Runtime and device throughput |
| `torch.float32` | Lower storage and often higher throughput | Reference scores and downstream quality |
| `packed=True` | Combine channel coefficient actions | Extra workspace, full job latency |
| `capture=True` | Replay a fixed CUDA launch schedule | Capture overhead, persistent buffers, concurrency |

The best matched public PageRank configuration was **FP32, unpacked, `capture=False`**. Enabling every option does not maximize performance. FP32 failed the stricter gates on several earlier workloads. Choose a configuration from accepted measurements on your task.

## Reuse with explicit boundaries

`CudaExecutor` is inference-only and binds an example shape. Returned outputs are cloned so that a later call does not overwrite an earlier result. **Do not share one executor between simultaneous callers.** Allocate separate executors or serialize access. Each executor has its own parameter buffer; additional instances add memory.

Use `evaluate_cuda(prepared, tensor)` for the functional path with supported input autograd. It does not differentiate through coefficient fusion or discrete degree selection. Coefficients and operator data are prepared constants; the directional derivative API covers one fixed affine direction separately.

For a sensitivity plan with radius `h`, pass `theta` only inside `[-h,h]`. Changing the graph arbitrarily requires recompilation. A CUDA graph capture reuses a launch schedule; it does not certify accuracy or accelerate every graph.

## Measure the complete call

Warm up first. Synchronize before and after wall-time measurements. Report preparation, first call, resident compute, input transfer and host output separately. Validate the final externally consumed score coordinates inside the timed call. Compare accepted outputs under identical graph semantics and tolerance.

CPU compilation and graph normalization are not included in the saved GPU preparation number. Sampled memory is total device/process use, not isolated operator allocation. Read the [benchmark methodology](../benchmarks/methodology.md) for the measured scope.

## Evaluate the precision experiment

The [precision GPU study](../benchmarks/precision-gpu.md) publishes a separate fused CuPy/CUDA implementation for 1–16 FP64 channels and diagonal affine directions. Its checked root API includes the compiler/header and stream-order repairs; its measured large-run versions are frozen separately. These functions are not exports of the installed `k_graphfuse.gpu` namespace.

Use the [experimental source and examples](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/experiments/precision_gpu/README.md) when evaluating that mechanism. Keep each prepared workspace serialized, validate before capture, and remember that `checked=False` returns reused buffers. Matching a captured input shape does not guarantee that a changed parameter or graph is reflected in a captured replay.
