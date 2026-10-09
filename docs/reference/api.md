---
description: K GraphFuse Python signatures, shapes, returned plan fields, numerical limits, exceptions and concurrency contracts.
---
# Python API reference

Public CPU functions are imported from `k_graphfuse`. CUDA functions are imported from `k_graphfuse.gpu`, which requires PyTorch. Historical flat C2Fuse imports remain available. Version `0.1.0` preserves the numerical implementation; plan mappings are implementation objects, not a stable serialized interchange format.

## Sparse polynomial plans

### `compile_sparse`

```python
compile_sparse(a, stages, tolerance=0, *, constant=Fraction(2), bounds="exact")
```

| Parameter | Contract |
|---|---|
| `a` | Nonempty square real finite SciPy sparse operator; copied to float64 CSR |
| `stages` | Nonempty sequence of stages; each is ascending-power `q × q` coefficients with common channel size |
| `tolerance` | Nonnegative finite rational-compatible operator budget |
| `constant` | Complete-inequality constant, at least `2`; default `2` uses the cited manuscript premise |
| `bounds` | `"exact"` or `"fast"`; fast mode has an exact overflow fallback |

Returns a read-only mapping with `a`, exact and float coefficients, `degree`, `fused_degree`, `original_degree`, `stage_count`, `error_bound`, `tolerance`, `constant`, `radius`, `norm_radius`, `bound_method` and `geometry_method`. It binds canonical stored values. See [certificate scope](../concepts/contracts.md).

Invalid graph shape, values, stages or budgets raise `ValueError`; arithmetic conversion/overflow may raise `ArithmeticError`. Input graph mutation after compilation does not update the plan.

### `evaluate_sparse`

```python
evaluate_sparse(plan, x)
```

Accepts a finite real `n × q` array-like state and returns a float64 `n × q` NumPy array. Uses sparse Horner evaluation. Invalid input raises `ValueError`; nonfinite output raises `ArithmeticError`. Independent concurrent CPU evaluations are tested. No graph matrix powers are materialized.

### `sparse_bounds`

```python
sparse_bounds(a)
```

Returns rational `(operator_norm_upper, numerical_range_disk_radius_upper)` for the supported stored sparse operator. Prefer compilation for canonicalization and validation. Conservative enclosures may prevent truncation.

## Directional sensitivities

### `compile_sensitivity`

```python
compile_sensitivity(a, direction, stages, value_tolerance, derivative_tolerance,
                    *, constant=Fraction(2), bounds="fast", parameter_radius=0)
```

`direction` is a real finite sparse matrix with the same shape as `a`. The two budgets and parameter radius are nonnegative rational-compatible values. The model is `A(theta)=A+theta*direction`, uniformly for `abs(theta)<=parameter_radius`.

Returns a sparse plan with additional fields `direction`, `direction_norm_bound`, `derivative_error_bound`, `derivative_tolerance`, `derivative_scale`, `derivative_bound_method`, `parameter_radius`, `parameter_norm_radius` and `parameter_numerical_radius`. Degree selection satisfies both implemented tail-budget checks. It is fixed after compilation.

### `evaluate_sensitivity`

```python
evaluate_sensitivity(plan, x, *, theta=0)
```

Returns `(value, derivative)`, two float64 `n × q` arrays. The derivative is of the retained polynomial along the fixed direction, with constant input and coefficients. Invalid state or out-of-interval `theta` raises `ValueError`; nonfinite results raise `ArithmeticError`. [Runnable guide](../guides/sensitivity.md).

## Symmetric resolvent and residual audit

### `compile_ppr`

```python
compile_ppr(s, alpha, tolerance, *, weights=None)
```

`s` must be exactly symmetric after canonicalization. `0<alpha<=1` is restart probability. Tolerance must be strictly positive. Optional enclosure weights must be positive, finite and match the vertex count. The certified spectral enclosure must keep the resolvent nonsingular. Returns a mapping containing `s`, `radius`, `alpha`, `degree`, `ratio`, `scale`, `error_bound`, `tolerance` and `backend`.

### `evaluate_ppr`

```python
evaluate_ppr(plan, x)
```

Accepts a finite real `n` vector or `n × q` state. Returns the same dimensionality as a float64 array, approximating `alpha*(I-(1-alpha)*s)^(-1)x`. This API is not the finite ten-update benchmark task.

### `certify_ppr`

```python
certify_ppr(p, source, estimate, alpha)
```

Audits a square nonnegative sparse transition with matching source/estimate vectors. Requires `0<alpha<=1` and a contraction bound below one. Returns `l1_error_bound`, `exact_residual_l1`, `contraction_bound` and `meaning`. The first three are exact rationals. It is an audit, not a solver; charge its rational cost. Invalid models or states raise `ValueError`. [Guide](../guides/resolvent.md).

## Local sparse propagation

```python
local_plan = prepare_local(plan)
output = evaluate_local(local_plan, x)
```

`prepare_local` adds a CSC index to a sparse polynomial plan. `evaluate_local` requires a finite real SciPy sparse state of shape `n × q` and returns sparse COO output. The reachable set can become global; locality does not imply a universal complexity improvement. Invalid or unprepared inputs raise `ValueError`.

## CUDA execution

### `prepare_cuda`

```python
prepare_cuda(plan, *, dtype=torch.float64, packed=False)
```

Requires CUDA availability and FP32 or FP64 dtype. Accepts a sparse polynomial, sensitivity or symmetric resolvent plan. Returns prepared device operators and coefficient actions bound to the source plan. Packing may allocate additional channel-action workspace. Default FP64; validate FP32 separately.

### `evaluate_cuda`

```python
evaluate_cuda(prepared, x, *, theta=0)
```

Requires a finite real CUDA tensor of compatible `n × q` shape, dtype and device. Returns a tensor or `(value, derivative)` for sensitivity. The functional path supports tested input autograd; operator and coefficient differentiation are outside the API. Invalid states or parameters raise `ValueError`; nonfinite outputs raise `ArithmeticError`.

### `CudaExecutor`

```python
executor = CudaExecutor(prepared, example, *, capture=True)
result = executor(x, theta=0)
```

Prepares a fixed-shape inference executor and optionally captures its launch schedule. Inputs must match the example; outputs are checked and cloned. One instance is **not safe for concurrent callers**. Captured inference does not preserve autograd. Preparation/capture overhead and persistent memory are part of deployment costs. [CUDA guide](../guides/gpu.md).

## Dense exact reference helpers

`compile_pipeline(a, stages, tolerance=0, *, constant=Fraction(2))`, `evaluate(plan, x, exact=False)`, `fuse(stages)` and `rational_matrix(values)` expose the historical small dense/rational reference path. Use it for small independent algebra checks. It is not the large-graph runtime. Exact mode can incur substantial rational bit-length costs.
