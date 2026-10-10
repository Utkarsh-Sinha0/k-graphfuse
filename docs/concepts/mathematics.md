# Mathematical derivation

This document packages the derivation underlying the existing code. The complete-inequality argument is conditional on the cited OpenAI manuscript claim. It does not assert worldwide novelty, a new shortest-path algorithm, or superiority over every existing GPU method.

The premise is [A direct proof of the complete Crouzeix inequality, Theorem 1.1](https://github.com/openai/math/blob/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints/A-direct-proof-of-the-complete-Crouzeix-inequality-September-26-2026/paper.pdf), at repository snapshot `fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb`. The complete constant is taken to be `K = 2`. The code also retains classical operator-norm bounds and selects the smaller applicable bound.

## 1. Model and exact ordered fusion

Let `A` be a real `n x n` matrix, `X` a real `n x q` feature state, and all `B[j,k]` be `q x q` channel matrices. Stage `j` is

`T_j(X) = sum_k A^k X B[j,k]^T`.

Write `P_j(z) = sum_k B[j,k] z^k`. Applying stages 1 through L gives

`P(z) = P_L(z) ... P_1(z) = sum_{k=0}^D C_k z^k`,

`Y = sum_{k=0}^D A^k X C_k^T`.

Proof: each two-stage term is `A^l A^k X B[1,k]^T B[2,l]^T = A^(k+l) X (B[2,l] B[1,k])^T`. Repeated convolution preserves that order. Channel matrices need not commute. Every stage must share the same graph operator, and no intervening nonlinear activation is allowed.

For row-major vectorization, `vec_row(A X C^T) = (A tensor C) vec_row(X)`. Thus define `P[A] = sum_k A^k tensor C_k`; its induced Euclidean norm controls the Frobenius norm of output states. No Kronecker matrix needs to be materialized.

The retained polynomial is `Q_m(z) = sum_{k=0}^m C_k z^k`; the discarded tail is `R_m = P - Q_m`. Let `a_k` be any outward upper bound on `||C_k||_2`. The implementation uses exact rational upper bounds, which can be conservative.

## 2. Uniform value certificate

The numerical range is `W(A) = {u* A u : ||u||_2 = 1}`, allowing complex vectors even for real A. Suppose it lies in the origin-centered disk of radius `r_A`, and `rho_A >= ||A||_2`.

For a fixed sparse direction `E`, let `eta >= ||E||_2` and define

`A_theta = A + theta E`, with `|theta| <= h`.

The triangle inequality gives uniform enclosures

`r = r_A + h eta`, and `rho = rho_A + h eta`.

Indeed `|u* A_theta u| <= r_A + |theta| eta` and `||A_theta||_2 <= rho_A + |theta| eta`.

Under the assumed complete inequality,

`||R_m[A_theta]||_2 <= K max_{z in W(A_theta)} ||R_m(z)||_2 <= K sum_{k>m} a_k r^k`.

The classical bound is `||R_m[A_theta]||_2 <= sum_{k>m} a_k rho^k`. Therefore

`delta_value(m) = min(K sum_{k>m} a_k r^k, sum_{k>m} a_k rho^k)`.

For every feature state X and every theta in the interval,

`||P[A_theta] X - Q_m[A_theta] X||_F <= delta_value(m) ||X||_F`,

where the notation means the corresponding vectorized operator action. This is an exact-arithmetic truncation guarantee. To impose an absolute output tolerance epsilon for an input of norm M, choose the operator budget no larger than `epsilon/M` when M is positive.

## 3. Why the derivative requires its own budget

The directional derivative of a power is

`D(A^k)[E] = sum_{j=0}^{k-1} A^j E A^(k-1-j)`.

Consequently the discarded derivative has the classical bound

`delta_derivative_classic(m) = eta sum_{k>m} k a_k rho^(k-1)`.

A small value tail need not have a small derivative tail. Consider the exact scalar model `P(z) = 1 + z^100`, `A = 9/10`, `E = 1`, `X = 1`, `h = 0` and both budgets `1e-4`. Retaining degree 0 discards a value of about `2.656e-5`, which meets the value budget. Its discarded derivative is `100 (9/10)^99`, about `0.0029513`, or 29.5 times the derivative budget. A value-only compiler returns a zero derivative for the retained constant, even though the full derivative matters. The sensitivity compiler retains degree 100 for these budgets. This counterexample is included in the tests.

## 4. A scaled lift certifies the derivative tail

For any `t > 0`, form a proof-only block matrix

```text
B_t = [ A_theta   t E     ]
      [ 0         A_theta ]
```

Its kth power has upper-right block `t D(A_theta^k)[E]`. Hence, up to the harmless permutation that groups block and channel indices,

```text
R_m[B_t] = [ R_m[A_theta]   t D R_m[A_theta][E] ]
           [ 0              R_m[A_theta]        ]
```

For a unit block vector `(u,v)`, the diagonal contribution is a convex combination of points in `W(A_theta)`. The off-diagonal contribution satisfies

`|t u* E v| <= t eta ||u|| ||v|| <= t eta/2`.

Therefore `W(B_t)` lies in the disk of radius `r + t eta/2`. Apply the complete inequality to B_t and take its upper-right block; the block extraction maps have norm one. This yields

`||D R_m[A_theta][E]||_2 <= (K/t) sum_{k>m} a_k (r + t eta/2)^k`.

For any chosen positive scale t, the usable derivative certificate is

`delta_derivative(m,t) = min(delta_derivative_classic(m), (K/t) sum_{k>m} a_k (r+t eta/2)^k)`.

It follows uniformly over the interval that

`||dY_full/dtheta - dY_retained/dtheta||_F <= delta_derivative(m,t) ||X||_F`.

The lift is a mathematical argument, not a matrix constructed by the evaluator. Both graph and channel dimensions are covered by the complete inequality.

## 5. Choosing the scale with derivatives

Set `S(u) = sum_{k>m} a_k u^k` and `c = eta/2`. Minimize the scalar function

`phi(t) = K S(r+c t)/t`, for `t > 0`.

Its derivative is

`phi'(t) = K [c t S'(r+c t) - S(r+c t)] / t^2`.

Each expanded term `(r+c t)^k/t` is a sum of a nonnegative multiple of `t^-1`, a constant, a linear term, and higher nonnegative powers of t. It is convex on `t > 0`. Equivalently, for the derivative numerator `H(t) = c t S'(r+c t) - S(r+c t)`,

`H'(t) = c^2 t S''(r+c t) >= 0`.

For a tail containing one positive degree `k > 1` and `r,eta > 0`, the optimum is `t = 2r/((k-1)eta)`. If all positive tail degrees lie between `k_min` and `k_max`, with `k_min >= 2`, their individual roots bracket a root for the sum:

`2r/((k_max-1)eta) <= t_opt <= 2r/((k_min-1)eta)`.

The implementation uses these endpoints, doubles the upper endpoint if a degree-1 term requires a larger bracket, and performs 24 exact-rational bisection steps on H. Any positive returned t still proves a valid bound. Incomplete minimization can make the bound less tight, but cannot invalidate it.

The implementation returns the classical derivative bound for zero radius, a purely degree-1 tail, or `r >= rho`. In the last case the lifted bound cannot improve the classical one: for `k >= 1`, `(rho+c t)^k/t >= k c rho^(k-1)` and K=2 gives `eta k rho^(k-1)`. A zero direction or zero derivative tail returns bound zero.

## 6. Compilation and coupled evaluation

Compilation first fuses the coefficients and obtains geometry bounds for A and E. It selects a value-admissible retained degree, then increases the degree until the derivative budget also passes. Zero coefficient gaps can be skipped. It is the first degree accepted by this implemented selection procedure, not an optimal approximation over all possible polynomials or representations.

Evaluate both quantities with the coupled Horner recurrence:

```text
Y = X C_m^T
Z = 0
for k = m-1 down to 0:
    Z = A_theta Z + E Y       # use Y from the previous iteration
    Y = A_theta Y + X C_k^T
return Y, Z
```

Proof by differentiating each Horner step: `d(A_theta Y)/dtheta = E Y + A_theta dY/dtheta`, while X and C_k are fixed. Z is the exact derivative of the retained polynomial. The degree choice is fixed after compilation; the evaluator does not differentiate through that discrete selection.

With `N_A` and `N_E` nonzero graph entries, the general evaluation arithmetic cost is `O(m (N_A q + N_E q + n q^2))`, up to fixed multiplicative factors. Storage is `O(N_A + N_E + n q + m q^2)` without packed workspace. Optional packed GPU coefficient evaluation adds workspace on the order of `(m+1)nq`. Diagonal E reduces direction application to row scaling. Nonzero theta may require additional direction applications; it is not free.

Exact rational coefficient fusion and bound construction have bit-length costs beyond arithmetic operation counts. Compilation is therefore additional work that must be amortized. Fusion can increase polynomial degree; strict tolerances can require the full degree, leaving no truncation saving.

## 7. Sparse bounds, tiny coefficients and arithmetic assumptions

The implementation canonicalizes a copied real float64 sparse operator, checks shape and finiteness, and interprets its stored finite values as exact rationals for the certificate. Input conversion and duplicate-entry aggregation can already change a caller's intended mathematical matrix; the certificate concerns the stored canonical matrix. Direction matrices receive the same treatment.

For `H = (A+A^T)/2` and `S = (A-A^T)/2`, symmetric/skew row-sum enclosures give h_H and h_S. A valid origin-centered numerical-range enclosure is `min(rho_A, sqrt(h_H^2+h_S^2))`. Operator bounds use `sqrt(||A||_1 ||A||_infinity)`. `bounds="fast"` supplies outward enclosures under the documented binary64 reduction model, with an exact fallback on overflow. It is not a hardware-independent proof of every possible reduction implementation.

Square-root bounds are enclosed by integer arithmetic. For rational `v=n/d >= 0`, select a dyadic scale s, compute `b=isqrt(floor(n s^2/d))`, then return lower `b/s` and upper `(b+1)/s` unless the lower square is exactly v. Their squares straddle v. The implemented scale uses a relative 42-bit target, avoiding the former fixed absolute floor for very small positive coefficient norms. This affects certificate tightness, not a new arithmetic complexity exponent.

Read-only mappings and sparse buffers prevent ordinary accidental mutation; they are not a security boundary against code deliberately mutating internal object attributes. Plans bind copied A and E. Arbitrary topology or coefficient changes require a new plan. A requested theta outside the compiled interval is rejected, including cases where a floating endpoint lies just beyond an exact rational boundary.


## 8. Prior art and attribution

The complete `K=2` premise replaces the older `1+sqrt(2)` constant from [Crouzeix and Palencia](https://arxiv.org/abs/1702.00668). For a fixed disk and tail it tightens the bound by approximately 17.16%; a tighter bound need not reduce an integer selected degree.

Block triangular derivative identities and numerical-range derivative analysis are established techniques. [Crouzeix and Kressner](https://arxiv.org/abs/2007.09784) provide relevant bivariate matrix-function prior art. The scaled application and implementation are not a proof of worldwide priority. The older complete constant also permits a weaker lifted bound.

The measured public PageRank result uses zero truncation and demonstrates no manuscript-driven degree reduction. Family 107 rectangular multiplication is not implemented. See [results](../benchmarks/results.md), [accuracy contracts](contracts.md) and [support limits](../reference/limitations.md).

## Theorem-bound attribution update

The [theorem-bound attribution analysis](https://utkarsh-sinha0.github.io/k-graphfuse/research/bound-attribution/) separates the new complete-Crouzeix bound from balancing and GPU fusion. **Two of 11 legacy T4 workload records reduce degree: 23→22 and 19→18.** Across all eight accepted matched FP64 configurations in those two records, the new-bound plan records **3.56–5.35% less median resident time**, with lower observed p95. This is single-run evidence below the separate 10% strong-speed gate. The exact CPU 9→8 boundary is untimed; the checked laptop old/new-bound cases retain equal degrees.
