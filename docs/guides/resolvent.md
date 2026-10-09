---
description: Compute a symmetric restarted graph resolvent with Chebyshev evaluation and audit candidate probability scores with exact rational residuals.
---
# Compute a symmetric restarted resolvent

This backend approximates `alpha * (I - (1-alpha)S)^(-1) X` using a Chebyshev expansion. `alpha` is the **restart probability**, so damping is `1-alpha`. The stored sparse operator `S` must be exactly symmetric; a certified spectral enclosure must keep the resolvent nonsingular.

```python
from fractions import Fraction as F
from scipy.sparse import csr_array
from k_graphfuse import compile_ppr, evaluate_ppr

s = csr_array([[0., 1.], [1., 0.]])
plan = compile_ppr(s, alpha=F(1, 5), tolerance=F(1, 10**10))
y = evaluate_ppr(plan, [1., 0.])
```

Run `python examples/resolvent.py` to compare this output with a dense linear solve. Positive `weights` can tighten the spectral enclosure for an irregular symmetric operator. They are enclosure weights, not personalization probabilities.

## Convert graph coordinates deliberately

An undirected graph may admit a symmetric similarity transform. Inputs and outputs then need the matching degree-based conversion. Arbitrary directed transitions are not accepted by `compile_ppr`. This guide's symmetric resolvent and the benchmark's global **ten-update** PageRank polynomial are distinct tasks: do not compare their timings without matching semantics and quality.

## Audit an existing candidate

`certify_ppr(p, source, estimate, alpha)` audits a nonnegative transition operator, including eligible directed operators, with an exact rational L1 residual bound. It checks that `(1-alpha)||p||_1 < 1`. It does not solve for the estimate.

```python
from k_graphfuse import certify_ppr
certificate = certify_ppr(s, [1., 0.], y, F(1, 5))
print(float(certificate["l1_error_bound"]))
```

This audit interprets canonical stored values and the supplied state as exact rationals. Its guarantee concerns that represented model. Exact rational computation can be expensive and must be charged to the workflow. A residual-to-error bound can be much larger than the residual, especially for a small restart probability.

A returned resolvent plan has a strictly positive truncation tolerance. Neither its certificate nor agreement with one CPU solution proves universal floating-point accuracy. See [contracts](../concepts/contracts.md) and [API details](../reference/api.md#symmetric-resolvent-and-residual-audit).
