"""Compare the coupled derivative with an independent analytic derivative."""
import numpy as np
from scipy.sparse import csr_array
from k_graphfuse import compile_sensitivity, evaluate_sensitivity

a = csr_array([[0., 0.5], [0.25, 0.]])
direction = csr_array([[0.1, 0.], [0., -0.1]])
x = np.array([[1.], [2.]])
plan = compile_sensitivity(a, direction, [[[[1]], [[2]], [[1]]]], 0, 0,
                           parameter_radius=0.25)
theta = 0.125
y, dy = evaluate_sensitivity(plan, x, theta=theta)
current = a.toarray()+theta*direction.toarray()
e = direction.toarray()
np.testing.assert_allclose(y, (np.eye(2)+2*current+current@current)@x)
np.testing.assert_allclose(dy, (2*e+e@current+current@e)@x)
print("Output:", y.ravel(), "Directional derivative:", dy.ravel())
