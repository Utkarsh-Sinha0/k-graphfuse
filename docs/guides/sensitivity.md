---
description: Evaluate a linear graph filter and its derivative for one affine graph parameter with separate truncation budgets.
---
# Differentiate an affine graph parameter

**Goal:** return both `Y(theta)` and `dY/dtheta` for `A(theta)=A+theta E`. `E` is a fixed sparse direction; inputs and channel coefficients are constant. This is useful for parameter sweeps and sensitivity analysis of compatible linear graph filters.

```python
import numpy as np
from scipy.sparse import csr_array
from k_graphfuse import compile_sensitivity, evaluate_sensitivity

a = csr_array([[0., 0.5], [0.25, 0.]])
e = csr_array([[0.1, 0.], [0., -0.1]])
x = np.array([[1.], [2.]])
stages = [[[[1]], [[2]], [[1]]]]
plan = compile_sensitivity(a, e, stages, 0, 0, parameter_radius=0.25)
y, dy = evaluate_sensitivity(plan, x, theta=0.125)
```

The retained polynomial is `I+2A+A²`. An independent derivative is `(2E + EA(theta) + A(theta)E)X`. Run `python examples/sensitivity.py` for this check. For GPU execution, pass this plan to `prepare_cuda` and evaluate using the same `theta`.

## Budget value and derivative separately

`value_tolerance` controls the discarded output operator; `derivative_tolerance` controls the discarded derivative operator. Both are nonnegative. A zero budget retains the corresponding nonzero tail. Inspect `error_bound`, `derivative_error_bound`, `degree`, `derivative_bound_method` and `parameter_radius` in the returned plan.

A small value tail can hide a large derivative tail: for `P(z)=1+z^100` at `z=0.9`, the tail value is about `2.66e-5`, while its derivative is about `2.95e-3`. A `1e-4` value budget alone is insufficient for the derivative. The regression tests include this case.

## Keep the parameter within the compiled interval

The plan is uniform over `abs(theta) <= parameter_radius`. A parameter outside the interval is rejected. Exact rational endpoints and floating values can differ slightly; choose interior values or represent the endpoint consistently. Arbitrary topology changes and multiple independently varying directions require a different implementation.

The coupled Horner recurrence differentiates the retained polynomial exactly in exact arithmetic. Its certificates bound omitted terms, with a manuscript-dependent complete-inequality premise when selected. They do not cover floating-point roundoff, graph canonicalization changes or differentiation through degree selection. See the [derivation](../concepts/mathematics.md) and [API](../reference/api.md#directional-sensitivities).
