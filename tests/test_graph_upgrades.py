from fractions import Fraction as F
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import coo_array, csr_array, dok_array, lil_array

from c2fuse import rational_matrix
from chebyshev_ppr import compile_ppr, evaluate_ppr
from fast_bounds import fast_sparse_bounds
from local_c2fuse import prepare_local, evaluate_local
from ppr_certificate import certify_ppr
from sparse_c2fuse import compile_sparse, evaluate_sparse, sparse_bounds


class Upgrades(unittest.TestCase):
    def test_fast_bounds_enclose_exact_bounds(self):
        rng = np.random.default_rng(1009)
        for n in range(1, 12):
            for _ in range(8):
                a = rng.normal(size=(n, n))
                a[rng.random((n, n)) < 0.6] = 0
                a = csr_array(a)
                fast, exact = fast_sparse_bounds(a), sparse_bounds(a)
                self.assertGreaterEqual(fast[0], exact[0])
                self.assertGreaterEqual(fast[1], exact[1])
                self.assertLess(float(fast[0]-exact[0]), 1e-8)
        for value in (0., np.nextafter(0., 1.), 1e-300, 1e200):
            a = csr_array([[value]])
            fast, exact = fast_sparse_bounds(a), sparse_bounds(a)
            self.assertGreaterEqual(fast[0], exact[0])
            self.assertGreaterEqual(fast[1], exact[1])
        overflow = compile_sparse(csr_array([[1e308]]), [[[[1]]]], bounds="fast")
        self.assertEqual(overflow["geometry_method"], "exact rational rows")

    def test_fast_plan_and_exact_fallback(self):
        coeffs = [[[[1]], [[F(1, 3)]], [[F(1, 9)]]]]
        a = csr_array([[0., 0.125], [0.25, 0.125]])
        exact = compile_sparse(a, coeffs, F(1, 100))
        fast = compile_sparse(a, coeffs, F(1, 100), bounds="fast")
        self.assertEqual(exact["degree"], fast["degree"])
        self.assertGreaterEqual(fast["error_bound"], exact["error_bound"])
        np.testing.assert_array_equal(evaluate_sparse(exact, [[1], [2]]),
                                      evaluate_sparse(fast, [[1], [2]]))
        with patch("fast_bounds.fast_sparse_bounds", side_effect=OverflowError):
            fallback = compile_sparse(a, coeffs, bounds="fast")
        self.assertEqual(fallback["geometry_method"], "exact rational rows")
        with self.assertRaises(ValueError): compile_sparse(a, coeffs, bounds="guess")
        for a in (dok_array([[1.]]), lil_array([[1.]])):
            self.assertEqual(compile_sparse(a, coeffs)["degree"], 2)

    def test_local_matches_global_with_directed_boundary_paths(self):
        rng = np.random.default_rng(1109)
        for n in (5, 17, 100):
            for degree in (0, 1, 4):
                rows, cols = rng.integers(0, n, (2, n*2))
                a = csr_array((rng.choice([-0.125, 0.125], n*2), (rows, cols)), shape=(n, n))
                coefficients = [rational_matrix(rng.integers(-2, 3, (2, 2))) / 8**k
                                for k in range(degree+1)]
                plan = prepare_local(compile_sparse(a, [coefficients], bounds="fast"))
                x = coo_array(([1., 2., -1.], ([0, 0, n-1], [0, 0, 1])), shape=(n, 2))
                actual = evaluate_local(plan, x)
                np.testing.assert_allclose(actual.toarray(), evaluate_sparse(plan, x.toarray()),
                                           rtol=1e-14, atol=1e-14)
        # Movement is by columns of A, and returning cycles must be retained.
        a = csr_array(([0.25, 0.5, 0.5], ([1, 2, 1], [0, 1, 2])), shape=(100, 100))
        plan = prepare_local(compile_sparse(a, [[[[0]], [[0]], [[0]], [[1]]]]))
        x = coo_array(([1.], ([0], [0])), shape=(100, 1))
        np.testing.assert_array_equal(evaluate_local(plan, x).toarray(),
                                      evaluate_sparse(plan, x.toarray()))

    def test_local_empty_validation_and_no_dense_input(self):
        plan = compile_sparse(csr_array((100, 100)), [[[[1]], [[1]]]])
        x = coo_array((100, 1))
        with self.assertRaises(ValueError): evaluate_local(plan, x)
        local = prepare_local(plan)
        with patch.object(coo_array, "toarray", side_effect=AssertionError):
            self.assertEqual(evaluate_local(local, x).nnz, 0)
        for x in (np.zeros((100, 1)), coo_array((100, 2)),
                  coo_array(([float("nan")], ([0], [0])), shape=(100, 1))):
            with self.assertRaises(ValueError): evaluate_local(local, x)

    def test_exact_residual_certificate_and_norm_transfer_counterexample(self):
        alpha, epsilon = F(1, 5), F(1, 10_000)
        p = csr_array([[0., 1.], [1., 0.]])
        # The exact PPR vector for source [1,0] is [5/9,4/9].
        x = np.array([1, 0])
        estimate = np.array([F(5, 9)-epsilon, F(4, 9)+epsilon], dtype=object)
        # Public API accepts numeric finite vectors; rational audit uses binary64 estimates.
        certificate = certify_ppr(p, x, np.asarray(estimate, dtype=float), alpha)
        error = abs(F(5, 9)-F(float(estimate[0]))) + abs(F(4, 9)-F(float(estimate[1])))
        self.assertLessEqual(error, certificate["l1_error_bound"])
        residual = (2-alpha)*epsilon/alpha
        self.assertEqual(residual/epsilon, F(9))
        self.assertGreater(residual, epsilon)
        exact_audit = certify_ppr(p, x, estimate, alpha)
        self.assertEqual(exact_audit["exact_residual_l1"], 2*(2-alpha)*epsilon)
        self.assertEqual(exact_audit["l1_error_bound"], 18*epsilon)
        identity = csr_array([[1.]])
        self.assertEqual(certify_ppr(identity, np.array([1]), np.array([1]), F(1, 2))["l1_error_bound"], 0)

    def test_certificate_rejects_invalid_models(self):
        p = csr_array([[1.]])
        for alpha in (0, -1, 2, float("nan")):
            with self.assertRaises(ValueError): certify_ppr(p, [1], [1], alpha)
        for model in (csr_array([[-1.]]), csr_array([[3.]])):
            with self.assertRaises(ValueError): certify_ppr(model, [1], [1], F(1, 2))
        with self.assertRaises(ValueError): certify_ppr(p, [1, 2], [1], F(1, 2))
        with self.assertRaises(ValueError): certify_ppr(p, [float("nan")], [1], F(1, 2))

    def test_chebyshev_resolvent_and_spectral_enclosure(self):
        rng = np.random.default_rng(1210)
        for n in (1, 3, 8):
            a = rng.integers(0, 5, (n, n))/32
            a = (a+a.T)/2
            for alpha in (F(1, 5), F(1, 100), F(1)):
                plan = compile_ppr(csr_array(a), alpha, F(1, 10**10))
                x = rng.normal(size=(n, 3))
                expected = np.linalg.solve(np.eye(n)-(1-float(alpha))*a, float(alpha)*x)
                error = np.linalg.norm(evaluate_ppr(plan, x)-expected)/np.linalg.norm(x)
                self.assertLessEqual(error, float(plan["error_bound"])+1e-13)
                self.assertLessEqual(plan["error_bound"], F(1, 10**10))
                self.assertGreaterEqual(float(plan["radius"]), max(abs(np.linalg.eigvalsh(a)))-1e-14)
        zero = compile_ppr(csr_array((2, 2)), F(1, 2), F(1, 100))
        np.testing.assert_array_equal(evaluate_ppr(zero, [1, 2]), [0.5, 1.])
        # Weighted rows certify an irregular symmetric operator whose unweighted bound is loose.
        a = csr_array([[0., 0.5, 0.5], [0.5, 0., 0.], [0.5, 0., 0.]])
        weighted = compile_ppr(a, F(1, 10), F(1, 10**8), weights=[1.5, 1, 1])
        self.assertEqual(weighted["radius"], F(3, 4))
        self.assertLess(weighted["radius"], compile_ppr(a, F(1, 10), F(1, 10**8))["radius"])

    def test_chebyshev_rejects_invalid_spectrum_and_inputs(self):
        for s in (csr_array([[0., 1.], [0., 0.]]), csr_array([[2.]])):
            with self.assertRaises(ValueError): compile_ppr(s, F(1, 10), F(1, 100))
        for weights in ([0], [-1], [float("nan")], [1, 2]):
            with self.assertRaises(ValueError):
                compile_ppr(csr_array([[0.]]), F(1, 10), F(1, 100), weights=weights)
        with self.assertRaises(ValueError): compile_ppr(csr_array([[0.]]), 0, 1e-8)
        with self.assertRaises(ValueError): compile_ppr(csr_array([[0.]]), 0.2, 0)
        plan = compile_ppr(csr_array([[0.]]), 0.2, 1e-8)
        with self.assertRaises(ValueError): evaluate_ppr(plan, [float("nan")])


if __name__ == "__main__":
    unittest.main()
