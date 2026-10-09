# Graph and accuracy contracts

## Compatible filter stages

The supported stage is `T_j(X)=sum_k A^k X B[j,k]^T`, with a real `n × n` operator, an `n × q` state and `q × q` channel coefficients. Stages share the operator and channel shape. Channel matrices need not commute; fusion preserves stage order. Interleaved nonlinear functions and different graph operators are outside this contract.

Sparse compilation copies the caller's matrix to canonical float64 CSR, sums duplicates and removes explicit zeros. Shape and finite real values are validated. Certificates concern the resulting stored matrix; conversion and duplicate aggregation can differ from a caller's intended exact mathematical model.

## Tolerance is an operator budget

For discarded polynomial terms with bound `delta`, the exact-arithmetic guarantee is:

```text
||full(X) - retained(X)||_F <= delta * ||X||_F
```

A requested absolute output tolerance `epsilon` for an input with norm `M>0` requires an operator budget no greater than `epsilon/M`. Zero tolerance retains the nonzero polynomial tail, but does not remove floating-point roundoff. For sensitivities, apply an independent derivative budget.

## What certificates cover

| Statement | Coverage |
|---|---|
| Rational coefficient fusion | Exact ordered algebra for represented coefficients |
| Polynomial tail certificate | Omitted terms in exact arithmetic for the canonical stored operator |
| Complete `K=2` bound | Conditional on the referenced manuscript claim |
| Classical norm bound | Alternative bound selected when smaller |
| `bounds="exact"` | Rational row-bound construction on stored values |
| `bounds="fast"` | Outward enclosure under the implementation's binary64 reduction model; exact overflow fallback |
| PPR residual audit | Exact rational residual-to-error bound for the represented nonnegative transition and supplied states |
| Empirical reference comparison | Observed agreement on tested inputs only |

Certificates do **not** include coefficient conversion, operator conversion, sparse reduction or executor roundoff. A customer requiring a full floating-point proof needs additional arithmetic-error analysis or validated computation. Read-only plan buffers prevent ordinary accidental mutation; they are not a security boundary against deliberate internal mutation.

## PageRank conventions

`compile_ppr` uses **restart** `alpha`; damping is `1-alpha`. The public benchmark uses damping `0.85`, so restart is `0.15`, and compares exactly ten global updates. It does not time an infinite personalized resolvent. Directedness, edge multiplicity, dangling treatment, uniform initialization and final probability coordinates are part of the task, not optional implementation details.

## Invalidation and concurrency

A plan binds copied operator data and fused coefficients. New topology or coefficients require a new plan. The sensitivity model permits `A+theta E` only in the compiled interval. CPU sparse evaluation is covered by concurrent-call tests. Each CUDA executor must be serialized or owned by one caller; returned outputs are cloned. Concurrency, memory capacity and throughput require deployment-specific measurements.
