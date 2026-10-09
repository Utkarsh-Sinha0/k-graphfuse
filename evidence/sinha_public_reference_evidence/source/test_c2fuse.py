from fractions import Fraction as F
import unittest
import numpy as np
from c2fuse import compile_pipeline, evaluate, fuse, rational_matrix, sqrt_upper, sqrt_bounds


def explicit_operator(a, coefficients):
    a = rational_matrix(a)
    q = len(coefficients[0])
    power = rational_matrix(np.eye(len(a), dtype=int))
    operator = np.full((len(a)*q, len(a)*q), F(0), dtype=object)
    for b in coefficients:
        operator += np.kron(power, rational_matrix(b))
        power = power @ a
    return operator


class Checks(unittest.TestCase):
    def test_ordered_fusion_against_explicit_operators(self):
        a = [[0, F(1, 2)], [0, F(1, 3)]]
        stages = [[[ [1, 0], [0, 1] ], [ [0, 1], [0, 0] ]],
                  [[ [1, 0], [0, 1] ], [ [0, 0], [1, 0] ]]]
        x = rational_matrix([[1, 2], [3, 4]])
        expected = x.ravel()
        for stage in stages:
            expected = explicit_operator(a, stage) @ expected
        plan = compile_pipeline(a, stages)
        np.testing.assert_array_equal(evaluate(plan, x, exact=True).ravel(), expected)
        reversed_coeffs = fuse(list(reversed(stages)))
        self.assertFalse(np.array_equal(explicit_operator(a, reversed_coeffs),
                                        explicit_operator(a, fuse(stages))))

    def test_exact_cancellation_for_every_base_matrix(self):
        identity = np.eye(4, dtype=int)
        shear = np.zeros((4, 4), dtype=int); shear[0, 3] = 1
        a = [[2, 7], [-3, 1]]
        plan = compile_pipeline(a, [[identity, shear], [identity, -shear]])
        self.assertEqual(plan["degree"], 0)
        self.assertEqual(plan["error_bound"], 0)
        x = rational_matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
        np.testing.assert_array_equal(evaluate(plan, x, exact=True), x)

    def test_truncation_bound_with_exact_rational_arithmetic(self):
        rng = np.random.default_rng(20261009)
        for _ in range(20):
            a = rational_matrix(rng.integers(-2, 3, size=(3, 3))) / 10
            coeffs = [rational_matrix(rng.integers(-2, 3, size=(4, 4))) / 2**k
                      for k in range(9)]
            x = rational_matrix(rng.integers(-4, 5, size=(3, 4)))
            plan = compile_pipeline(a, [coeffs], F(1, 10))
            full = (explicit_operator(a, coeffs) @ x.ravel()).reshape(x.shape)
            error = full - evaluate(plan, x, exact=True)
            self.assertLessEqual(sum(v*v for v in error.ravel()),
                                 plan["error_bound"]**2 * sum(v*v for v in x.ravel()))
            self.assertLessEqual(plan["error_bound"], F(1, 10))

    def test_outward_square_root(self):
        for x in [F(0), F(2), np.int64(2), F(np.int64(2)),
                  F(1, 10**40), F(10**80, 3), F(9, 4)]:
            expected = F(x)
            expected = F(int(expected.numerator), int(expected.denominator))
            self.assertGreaterEqual(sqrt_upper(x)**2, expected)
            lower, upper = sqrt_bounds(x)
            self.assertLessEqual(lower**2, expected)
            self.assertGreaterEqual(upper**2, expected)
            if lower:
                self.assertLessEqual((upper-lower)/lower, F(1, 2**42))
        tiny = F(1, 10**80)
        self.assertLessEqual(sqrt_upper(tiny), F(1000000000001, 10**52))

    def test_validation_and_zero_matrix(self):
        with self.assertRaises(ValueError): compile_pipeline([[1]], [], 0)
        with self.assertRaises(ValueError): compile_pipeline([[1, 2]], [[[ [1] ]]], 0)
        with self.assertRaises(ValueError): compile_pipeline([[float('nan')]], [[[ [1] ]]], 0)
        with self.assertRaises(ValueError): compile_pipeline([[1]], [[[ [1] ]]], -1)
        with self.assertRaises(ValueError): compile_pipeline([[1]], [[[ [1] ]]], constant=1)
        plan = compile_pipeline([[0]], [[[ [1] ], [ [5] ]]], 0)
        self.assertEqual(plan["degree"], 0)
        self.assertEqual(evaluate(plan, [[2]], exact=True)[0, 0], 2)
        with self.assertRaises(ValueError): evaluate(plan, [[1, 2]])


if __name__ == "__main__":
    unittest.main()
