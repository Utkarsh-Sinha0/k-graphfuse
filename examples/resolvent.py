"""Check a symmetric restarted resolvent against a dense linear solve."""
from fractions import Fraction as F
import numpy as np
from scipy.sparse import csr_array
from k_graphfuse import compile_ppr, evaluate_ppr, certify_ppr

s = csr_array([[0., 1.], [1., 0.]])
alpha = F(1, 5)
source = np.array([1., 0.])
plan = compile_ppr(s, alpha, F(1, 10**10))
estimate = evaluate_ppr(plan, source)
expected = np.linalg.solve(np.eye(2)-(1-float(alpha))*s.toarray(), float(alpha)*source)
np.testing.assert_allclose(estimate, expected, atol=1e-10)
certificate = certify_ppr(s, source, estimate, alpha)
print("Scores:", estimate, "Exact residual L1 bound:", float(certificate["l1_error_bound"]))
