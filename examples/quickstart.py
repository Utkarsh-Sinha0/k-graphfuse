"""Run two compatible filters and check the result independently."""
import numpy as np
from scipy.sparse import csr_array
from k_graphfuse import compile_sparse, evaluate_sparse

a = csr_array([[0., 0.5], [0.25, 0.]])
x = np.array([[1.], [2.]])
stages = [[[[1]], [[1]]], [[[2]], [[1]]]]
plan = compile_sparse(a, stages, tolerance=0)
y = evaluate_sparse(plan, x)
expected = (2*np.eye(2)+a.toarray()) @ (np.eye(2)+a.toarray()) @ x
np.testing.assert_allclose(y, expected)
print("Retained degree:", plan["degree"])
print("Output:", y.ravel())
