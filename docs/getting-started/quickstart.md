---
description: Compile and verify two sparse graph polynomial stages using the public K GraphFuse Python API.
---
# Run your first graph filter

**Goal:** evaluate `(2I + A)(I + A)X` on a two-vertex operator and compare it with an independent calculation. Complete [installation](install.md) first. This tutorial uses CPU execution and requires no dataset or account.

## Define the operator and state

```python
import numpy as np
from scipy.sparse import csr_array
from k_graphfuse import compile_sparse, evaluate_sparse

a = csr_array([[0., 0.5], [0.25, 0.]])
x = np.array([[1.], [2.]])
```

The graph operator is `n × n`; the feature state is `n × q`. Here `n=2` and `q=1`. In the convention `Y=A@X`, entry `A[i,j]` carries a value from vertex `j` to vertex `i`. Check this orientation when importing directed graphs.

## Describe stages in ascending powers

```python
stages = [[[[1]], [[1]]], [[[2]], [[1]]]]
plan = compile_sparse(a, stages, tolerance=0)
y = evaluate_sparse(plan, x)
```

Each stage is a list of `q × q` channel matrices: constant coefficient first, then the coefficient of `A`, and so on. The first stage represents `(I+A)X`; the second applies `(2I+A)` to that result. All stages must use this same operator and channel count. A nonlinear activation between stages breaks the fusion contract.

The compiler copies and canonicalizes the operator, fuses coefficients in stage order, and retains degree `2`. Zero tolerance discards no nonzero tail. The floating evaluator still has arithmetic roundoff.

## Verify independently

```python
expected = (2*np.eye(2)+a.toarray()) @ (np.eye(2)+a.toarray()) @ x
np.testing.assert_allclose(y, expected)
print(plan["degree"], y.ravel())
```

Expected output:

```text
2 [5.125 5.   ]
```

Dense conversion is appropriate only for this tiny verification example. Production evaluation keeps the graph sparse.

## Reuse the plan

```python
y2 = evaluate_sparse(plan, 2*x)
np.testing.assert_allclose(y2, 2*y)
```

Reuse amortizes compilation. A new topology or new coefficients require a new plan. A copied plan does not track mutations to the caller's original sparse matrix.

For an approximation, choose a positive operator tolerance and inspect `plan["error_bound"]` and `plan["degree"]`. The omitted-tail guarantee scales with the input Frobenius norm and excludes executor roundoff. Read [accuracy contracts](../concepts/contracts.md) before using it as a customer acceptance threshold.

Next: [CUDA execution](../guides/gpu.md), [API reference](../reference/api.md), or [application fit](../guides/applications.md).
