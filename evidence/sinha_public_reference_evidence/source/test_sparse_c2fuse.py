from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import coo_array, csr_array, csr_matrix

from c2fuse import compile_pipeline, evaluate, norm_upper, rational_matrix, sqrt_upper
from sparse_c2fuse import compile_sparse, evaluate_sparse, sparse_bounds


class SparseChecks(unittest.TestCase):
    def test_sparse_bounds_equal_dense_bounds_exactly(self):
        rng = np.random.default_rng(1009)
        for n in range(1, 10):
            for _ in range(4):
                a = rng.integers(-3, 4, size=(n, n)) / 8
                a[rng.random((n, n)) < 0.65] = 0
                exact = rational_matrix(a)
                rho = norm_upper(exact)
                h, s = norm_upper((exact + exact.T)/2), norm_upper((exact - exact.T)/2)
                self.assertEqual(sparse_bounds(csr_array(a)),
                                 (rho, min(rho, sqrt_upper(h*h+s*s))))

    def test_ordered_stages_and_exact_residual_certificate(self):
        rng = np.random.default_rng(1010)
        for _ in range(20):
            a = rng.integers(-2, 3, (4, 4)) / 8
            stages = [[rational_matrix(rng.integers(-2, 3, (3, 3))) / 8**k
                       for k in range(5)] for _ in range(2)]
            x = rng.integers(-3, 4, (4, 3))
            sparse = compile_sparse(csr_array(a), stages, F(1, 1000))
            truncated = compile_pipeline(a, stages, F(1, 1000))
            full = compile_pipeline(a, stages, 0)
            self.assertEqual(sparse["degree"], truncated["degree"])
            self.assertEqual(sparse["error_bound"], truncated["error_bound"])
            error = evaluate(full, x, exact=True) - evaluate(truncated, x, exact=True)
            self.assertLessEqual(sum(v*v for v in error.ravel()),
                                 sparse["error_bound"]**2 * int(np.sum(x*x)))
            np.testing.assert_allclose(evaluate_sparse(sparse, x),
                                       evaluate(truncated, x), atol=1e-14, rtol=1e-14)

    def test_canonicalization_and_caller_mutation(self):
        a = coo_array(([0.5, 0.25, -0.125], ([0, 0, 1], [1, 1, 0])), shape=(2, 2))
        plan = compile_sparse(a, [[[[1]], [[1]]]])
        expected = np.array([[2.5], [1.875]])
        a.data[:] = 9
        np.testing.assert_array_equal(evaluate_sparse(plan, [[1], [2]]), expected)
        with self.assertRaises(TypeError):
            plan["degree"] = 0
        with self.assertRaises(ValueError):
            plan["a"].data[0] = 0

    def test_zero_cancellation_and_no_reduction(self):
        identity = np.eye(3, dtype=int)
        shear = np.zeros((3, 3), dtype=int); shear[0, 2] = 1
        plan = compile_sparse(csr_matrix([[2., 7.], [-3., 1.]]),
                              [[identity, shear], [identity, -shear]], 0)
        self.assertEqual(plan["degree"], 0)
        x = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(evaluate_sparse(plan, x), x)
        zero = compile_sparse(csr_array((3, 3)), [[[[1]], [[20]]]], 0)
        self.assertEqual(zero["degree"], 0)
        full = compile_sparse(csr_array([[1.]]), [[[[1]], [[1]], [[1]]]], 0)
        self.assertEqual(full["degree"], 2)
        self.assertEqual(full["error_bound"], 0)

    def test_no_dense_materialization_and_concurrent_evaluation(self):
        a = csr_array(([0.25]*100, (np.arange(100), np.arange(100))), shape=(100, 100))
        with patch.object(csr_array, "toarray", side_effect=AssertionError("No dense conversion")), \
             patch.object(csr_array, "todense", side_effect=AssertionError("No dense conversion")):
            plan = compile_sparse(a, [[[[1]], [[1]], [[F(1, 100)]]]], F(1, 1000))
            x = np.ones((100, 1))
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: evaluate_sparse(plan, x), range(12)))
        for result in results:
            np.testing.assert_array_equal(result, results[0])

    def test_invalid_inputs_and_overflow(self):
        stages = [[[[1]], [[1]]]]
        for a in (np.eye(2), csr_array((0, 0)), csr_array((2, 3)),
                  csr_array([[float("nan")]]), csr_array([[1j]])):
            with self.assertRaises(ValueError): compile_sparse(a, stages)
        for budget in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compile_sparse(csr_array([[1.]]), stages, budget)
        with self.assertRaises(ValueError): compile_sparse(csr_array([[1.]]), [])
        with self.assertRaises(ValueError):
            compile_sparse(csr_array([[1.]]), stages, constant=1)
        plan = compile_sparse(csr_array([[2.]]), stages)
        for x in ([[1, 2]], [[float("nan")]], [[1j]]):
            with self.assertRaises(ValueError): evaluate_sparse(plan, x)
        with self.assertRaises(ArithmeticError): evaluate_sparse(plan, [[1e308]])


if __name__ == "__main__":
    unittest.main()
