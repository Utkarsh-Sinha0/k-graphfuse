from fractions import Fraction as F
import unittest
import numpy as np
from scipy.sparse import csr_array

from c2fuse import evaluate, compile_pipeline, rational_matrix, norm_upper
from sensitivity_c2fuse import compile_sensitivity, evaluate_sensitivity, derivative_tail_bound
from sparse_c2fuse import compile_sparse, evaluate_sparse


class SensitivityChecks(unittest.TestCase):
    def test_exact_block_derivative_and_truncation_bound(self):
        rng = np.random.default_rng(121010)
        for _ in range(12):
            a = rational_matrix(rng.integers(-2, 3, (3, 3)))/8
            e = rational_matrix(rng.integers(-2, 3, (3, 3)))/8
            coeffs = [rational_matrix(rng.integers(-2, 3, (2, 2)))/4**k for k in range(10)]
            x = rational_matrix(rng.integers(-3, 4, (3, 2)))
            block = np.block([[a, e], [np.zeros((3, 3), dtype=int), a]])
            state = np.vstack((np.zeros((3, 2), dtype=int), x))
            plan = compile_sensitivity(csr_array(np.asarray(a, dtype=float)),
                csr_array(np.asarray(e, dtype=float)), [coeffs], F(1, 1000), F(1, 1000), bounds="exact")
            full = evaluate(compile_pipeline(block, [coeffs]), state, exact=True)
            retained = evaluate(compile_pipeline(block, [coeffs[:plan["degree"]+1]]), state, exact=True)
            norm_squared = sum(v*v for v in x.ravel())
            value_error, derivative_error = full[3:]-retained[3:], full[:3]-retained[:3]
            self.assertLessEqual(sum(v*v for v in value_error.ravel()), plan["error_bound"]**2*norm_squared)
            self.assertLessEqual(sum(v*v for v in derivative_error.ravel()), plan["derivative_error_bound"]**2*norm_squared)
            y, dy = evaluate_sensitivity(plan, np.asarray(x, dtype=float))
            np.testing.assert_allclose(y, np.asarray(retained[3:], dtype=float), atol=1e-13)
            np.testing.assert_allclose(dy, np.asarray(retained[:3], dtype=float), atol=1e-13)

    def test_scaled_derivative_certificate_improves_classical(self):
        norms = [F(0)]*13+[F(1)]
        bound, scale, method = derivative_tail_bound(norms, 12, F(2), F(1), F(1))
        self.assertEqual(method, "scaled complete Crouzeix derivative")
        self.assertIsNotNone(scale)
        self.assertLess(bound, 13*2**12)
        self.assertAlmostEqual(float(scale), 2/12, places=6)

    def test_zero_direction_and_invalid_budgets(self):
        a = csr_array([[0., 0.25], [0., 0.]])
        stages = [[[[1]], [[1]], [[F(1, 10)]]]]
        plan = compile_sensitivity(a, csr_array((2, 2)), stages, F(1, 10), F(0))
        self.assertEqual(plan["derivative_error_bound"], 0)
        np.testing.assert_array_equal(evaluate_sensitivity(plan, [[1], [2]])[1], [[0], [0]])
        with self.assertRaises(ValueError): compile_sensitivity(a, a, stages, -1, 0)
        with self.assertRaises(ValueError): compile_sensitivity(a, csr_array((1, 1)), stages, 0, 0)

    def test_uniform_certificate_covers_changing_weights(self):
        a = rational_matrix([[F(-1, 8), 1], [0, F(1, 8)]])
        e = rational_matrix([[F(1, 4), 0], [0, F(-1, 4)]])
        coefficients = [rational_matrix([[F(1, 3**k), F((-1)**k, 4**k)], [0, F(1, 5**k)]])
                        for k in range(20)]
        plan = compile_sensitivity(csr_array(np.asarray(a, dtype=float)),
            csr_array(np.asarray(e, dtype=float)), [coefficients], F(1, 10**6), F(1, 10**6),
            parameter_radius=F(1, 8), bounds="exact")
        x = rational_matrix([[1, 2], [-1, 3]])
        state = np.vstack((np.zeros((2, 2), dtype=int), x))
        for theta in (F(-1, 8), F(0), F(1, 8)):
            current = a+theta*e
            block = np.block([[current, e], [np.zeros((2, 2), dtype=int), current]])
            full = evaluate(compile_pipeline(block, [coefficients]), state, exact=True)
            retained = evaluate(compile_pipeline(block, [coefficients[:plan["degree"]+1]]), state, exact=True)
            value_error, gradient_error = full[2:]-retained[2:], full[:2]-retained[:2]
            norm = sum(v*v for v in x.ravel())
            self.assertLessEqual(sum(v*v for v in value_error.ravel()), plan["error_bound"]**2*norm)
            self.assertLessEqual(sum(v*v for v in gradient_error.ravel()), plan["derivative_error_bound"]**2*norm)
            y, dy = evaluate_sensitivity(plan, np.asarray(x, dtype=float), theta=theta)
            np.testing.assert_allclose(y, np.asarray(retained[2:], dtype=float), atol=1e-13)
            np.testing.assert_allclose(dy, np.asarray(retained[:2], dtype=float), atol=1e-13)
        with self.assertRaises(ValueError): evaluate_sensitivity(plan, [[1, 2], [3, 4]], theta=1)

    def test_accurate_values_can_hide_a_bad_gradient(self):
        a, e = csr_array([[0.9]]), csr_array([[1.]])
        coefficients = [[[1]]] + [[[0]]]*99 + [[[1]]]
        value_only = compile_sparse(a, [coefficients], F(1, 10_000), bounds="fast")
        aware = compile_sensitivity(a, e, [coefficients], F(1, 10_000), F(1, 10_000))
        true_value_term = F(float(a.data[0]))**100
        true_gradient = 100*F(float(a.data[0]))**99
        self.assertEqual(value_only["degree"], 0)
        self.assertLess(true_value_term, F(1, 10_000))
        self.assertGreater(true_gradient, F(1, 10_000))
        self.assertEqual(aware["degree"], 100)
        y, dy = evaluate_sensitivity(aware, [[1.]])
        self.assertAlmostEqual(float(dy[0, 0]), float(true_gradient), places=14)


if __name__ == "__main__": unittest.main()
