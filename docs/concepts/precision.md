---
title: Similarity-aware graph polynomial precision — derivation and CUDA execution
description: Coordinate error transfer, dyadic balancing, coupled derivatives, coefficient encoding, selective precision, complexity and floating arithmetic limits.
---
# Similarity-aware precision and fused evaluation

This experimental implementation is a derived application of a stated research theorem and classical matrix balancing. It is not a claim of a new balancing theorem, an optimal similarity construction, or global state of the art. The tested released numerical core is unchanged.

The [technical manuscript](../research/paper.md) presents this derivation as a research document with proofs, methods, historical results, new controlled two-device confirmation, application outcomes, limitations and references. Its status is an unreviewed preprint draft.

## Contract

For a real stored sparse operator A, features X in R^(n x q), and exactly fused ordered matrix coefficients C_k, compute

    P_A(X) = sum_{k=0}^d A^k X C_k^T.

The coefficient matrices may not commute. Fusion and execution reuse the released implementation. The accuracy budget is output Frobenius error divided by input Frobenius norm. It is not PageRank aggregate L1 error or maximum relative error per vertex.

## Similarity and original-coordinate error

Choose a fixed positive diagonal D and let B = D^(-1) A D. Since A^k = D B^k D^(-1),

    P_A(X) = D P_B(D^(-1) X).

This follows term by term; right coefficient multiplication is unchanged. If a retained polynomial Q has induced feature-space error at most delta_B in the balanced coordinates, then

    ||P_A(X) - D Q_B(D^(-1) X)||_F
      <= ||D||_2 delta_B ||D^(-1) X||_F
      <= kappa_2(D) delta_B ||X||_F.

Therefore compile B with delta_B <= tolerance / kappa_2(D). The prototype records the penalty and the resulting bound in the original coordinates. It does not silently compare the balanced norm with the unbalanced norm.

The complete constant-two Crouzeix inequality in OpenAI manuscript 325-01 is taken as the supplied theorem premise. It bounds matrix-valued polynomial tails using the numerical range of the operator. The existing planner already uses the constant two. The new strategy changes the operator geometry before applying that bound; it does not obtain its performance improvement merely by replacing the old 1+sqrt(2) constant.

## Smallest implemented metric

Let r_i and c_i be the absolute off-diagonal row and column sums. For indices with both positive, choose

    e_i = ceil((log2(r_i) - log2(c_i)) / 4),
    D_ii = 2^(e_i - max_j(e_j)).

Use e_i=0 when either sum is zero. This is one parallel balancing step, not an iterative optimizer. The exponent is a heuristic; the validity of the error bound does not depend on it finding an optimum.

The condition penalty is computed exactly as 2^(max(e)-min(e)). Stored edges are scaled with binary exponents and then scaled back. Reject the transform if any entry is nonfinite or fails exact round-trip equality. Reject scale underflow. CPU evaluation validates incoming state, scaled state and output finiteness. Runtime floating rounding remains additional to the truncation certificate.

For the tested alternating directed family, outgoing couplings are 1 and 1/64. D alternates between 1 and 1/8, so both transformed couplings are 1/8 and kappa(D)=8. The prototype pays this eightfold error-budget penalty. A smaller numerical-range enclosure nevertheless reduces the retained degree and the number of sparse passes.

## Parameter derivative

For A(theta)=A+theta E, hold D fixed. Then

    B(theta)=D^(-1) A D + theta D^(-1) E D,
    d P_A(theta)(X) / d theta
      = D [d P_B(theta)(D^(-1) X) / d theta].

Divide both the value and derivative error budgets by the same condition penalty. Reuse the released joint value/derivative planner, including its certified parameter interval. The benchmark checks actual floating outputs at theta=0 and both interval endpoints. These three sampled checks do not prove all floating outputs throughout the interval; the conditional exact-arithmetic tail bound covers the interval.

This is not an adjoint gradient engine, nonlinear graph training, or a proof that fewer iterations improve downstream model quality.

## Full evaluation fast path

If every fused coefficient is retained, the discarded tail is exactly zero. Computing a numerical-range enclosure to bound that zero tail is unnecessary. `compile_full` preserves canonical input validation, exact ordered fusion, finite coefficient conversion and read-only plan arrays, then reuses the same Horner executor.

For A=0, all positive powers vanish and only the constant coefficient is needed. Other operators retain the full fused degree. The fast path does not drop tiny coefficients, claim a zero floating error, or accelerate the unchanged warm executor.

## Applicability and adoption

Balancing may help strongly asymmetric polynomial graph filters and repeated affine sensitivity calculations. It may hurt already balanced matrices, increase preparation time, or create large condition penalties. Keep the released plan as a control and adopt a metric plan only after accuracy acceptance and a held-out speed comparison on the intended workload. No automatic universal fallback router is implemented in this prototype.

The cold fast path targets full finite polynomials, including fixed-step PageRank-style propagation and repeatedly rebuilt operators. It excludes graph loading, initial normalization and unrelated native graph algorithms.

An asymptotic matrix multiplication exponent theorem is not a finite GPU kernel. No algorithm from family 107 is implemented here. No new BFS, shortest path, triangle counting, community detection, distributed storage or out-of-core backend is implemented.

## Coupled fused Horner evaluation

Write Z=D^(-1)X and B(theta)=B+theta F, where F=D^(-1)ED. For retained degree d initialize

    H_d = Z C_d^T,   J_d = 0.

For k=d-1,...,0, using the previous value before overwriting it, evaluate

    J_k = B(theta) J_(k+1) + F H_(k+1),
    H_k = B(theta) H_(k+1) + Z C_k^T.

The product rule proves J_k=dH_k/dtheta by induction, because Z and C_k are fixed. Return D H_0 and D J_0. For the implemented diagonal direction F, the extra directional product is one elementwise multiply per channel; the general case requires another sparse multiply and uses the existing fallback. The recurrence preserves noncommuting ordered coefficient products.

The narrow CUDA kernel assigns one row/channel output to each thread and accumulates the CSR edge products, optional derivative and coefficient term together. Alternating state buffers avoid allocating a new state per degree. Work remains O(d m q + d n q^2). Narrow resident storage is O(m+n q+d q^2), including operator, scales, coefficients and reused state. This is a constant-factor and memory-traffic improvement, not a better asymptotic graph algorithm.

## Encoding and selective precision

For exact retained rational coefficient C_k, encode a high part H_k=round64(C_k). Optionally encode L_k=round64(C_k-H_k). Let R_k=C_k-H_k-L_k, or C_k-H_k for high-only execution. `coefficient_certificate.py` computes an exact rational upper bound r_k on the matrix norm of R_k. In balanced coordinates the induced coefficient error is bounded by

    min(sum r_k rho^k, 2 sum r_k radius^k),

where rho is the stored operator-norm enclosure and radius is the numerical-range enclosure, enlarged to cover the parameter interval when applicable. Multiply by kappa(D) to return to original coordinates. The derivative encoding error uses the existing directional derivative polynomial bound with all degrees included. Add these bounds to the truncation certificate. This is exact-arithmetic accounting for the encoded polynomial; GPU execution roundoff is additional.

Plain FMA forms a product-plus-sum with one final rounding. Compensated accumulation additionally tracks product residuals with FMA and addition residuals. Mixed execution uses plain value accumulation and compensated derivative accumulation. The double-double backend carries two parts for intermediate states; the cheaper compensated backend rounds its completed per-thread sum to FP64 at each stage. None eliminates overflow, underflow or all propagation of rounding.

For wide features, standard BLAS computes all Z C_k^T terms in a dense O(d n q) projection workspace before sparse Horner execution. A high-only coefficient projection avoids a second contraction when the encoding bound and actual accuracy check permit it. Compensation at every sparse step can increase register pressure; a separately compiled compensated final-step kernel preserves the last-step accuracy benefit without that cost at every degree. The 16-channel follow-up uses this profile. It trades additional workspace for throughput rather than establishing a memory gain against every competitor.

The selection gate requires all sampled value/derivative Frobenius and maximum-absolute errors to be no worse than the fastest control and at least one to improve. Choose among eligible candidates by pilot latency, then evaluate held-out speed and p95. A tighter truncation certificate alone does not guarantee a more accurate floating result. The million-vertex T4 derivative case illustrates why: the faster mixed profile failed an error component, so the fully compensated profile was selected.

## Independent truth and integration boundaries

The full degree-48 CPU computation checks every output but is already rounded. Independent high-precision sparse Horner evaluation supplies the accuracy reference for the first 16 rows. In the tested operator family each multiply reaches at most the next row; degree 48 therefore needs at most 48 additional rows. The 64-row and 128-row influence windows contain that cone without wrapping into their artificial boundary. Evaluations at 256 and 384 bits agreed below 1e-65. This justifies those sampled rows, not arbitrary-precision truth at every vertex or for unrelated operators. Errors near the nearest-FP64 floor can improve only modestly in a stored FP64 result.

After the historical frozen timed experiments, a same-process integration check exposed optional CuPy CUB headers mismatching PyTorch's preloaded CUDA compiler, and a default-stream dependency ambiguity in projected execution. The final checked API replaces the finite reduction with a self-contained flag kernel and uses a named stream plus events to order cross-library projection. The combined 29 core and 14 upgrade checks passed. Historical large-run performance applies to its frozen sources. The new controlled study confirms the repaired implementation under a separate frozen protocol, including checked application requests; it does not retime the historical six cases.

## Controlled component findings

The new [controlled confirmation](https://utkarsh-sinha0.github.io/k-graphfuse/research/controlled-study/) uses the repaired implementation across six graph families and two GPUs. Matching-budget, matching-degree packed-balanced CuPy comparisons pass the strong speed gate in **18/20 pooled device–fixture comparisons**, with **36.88–83.71% less resident time in the wins**. R-MAT loses on both devices. MovieLens recommendation quality matches exactly, but specialized scalar-restart CuPy removes the application median speed advantage: K uses **7.13% more time on RTX** and practically ties T4. These are scoped experimental results, separate from native-cuGraph PageRank and the installed API.

Read the [closest-method comparison](../research/closest-methods.md) before attributing novelty to balancing, graph polynomials, compensation or fusion.
