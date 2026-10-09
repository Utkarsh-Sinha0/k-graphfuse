"""Run the new planning, local-query, resolvent and residual-audit APIs."""
from fractions import Fraction as F
import numpy as np
from scipy.sparse import coo_array, csr_array

from chebyshev_ppr import compile_ppr, evaluate_ppr
from local_c2fuse import prepare_local, evaluate_local
from ppr_certificate import certify_ppr
from sparse_c2fuse import compile_sparse


def main():
    vertices = np.arange(1000)
    a = csr_array((np.full(1000, 0.25), ((vertices+1) % 1000, vertices)), shape=(1000, 1000))
    coefficients = [[[F(1, 4**k)]] for k in range(33)]
    plan = prepare_local(compile_sparse(a, [coefficients], F(1, 10**8), bounds="fast"))
    source = coo_array(([1.], ([0], [0])), shape=(1000, 1))
    result = evaluate_local(plan, source)
    assert len(np.unique(result.row)) == plan["degree"]+1
    print(f"Finite filter: 32 -> {plan['degree']} passes; {result.nnz} output entries across 1000 vertices")
    p = csr_array([[0., 1.], [1., 0.]])
    resolvent = compile_ppr(p, F(1, 5), F(1, 10**10))
    estimate = evaluate_ppr(resolvent, [1, 0])
    audit = certify_ppr(p, [1, 0], estimate, F(1, 5))
    assert audit["l1_error_bound"] < F(1, 10**8)
    print(f"Symmetric PPR estimate: {estimate}")
    print(f"Exact achieved L1 error bound: {float(audit['l1_error_bound']):.12g}")


if __name__ == "__main__":
    main()
