from fractions import Fraction as F
import unittest
import numpy as np
from scipy.sparse import csr_array

try:
    import torch
    from cuda_c2fuse import prepare_cuda, evaluate_cuda, CudaExecutor
    AVAILABLE = torch.cuda.is_available()
except ImportError:
    AVAILABLE = False
from sensitivity_c2fuse import compile_sensitivity, evaluate_sensitivity
from sparse_c2fuse import compile_sparse, evaluate_sparse
from chebyshev_ppr import compile_ppr, evaluate_ppr


@unittest.skipUnless(AVAILABLE, "CUDA-enabled PyTorch and GPU required")
class CudaChecks(unittest.TestCase):
    def test_cuda_matches_cpu_and_captured_outputs_are_independent(self):
        a = csr_array([[0.1, 0.5, 0.], [0., -0.1, 0.25], [0.1, 0., 0.]])
        stages = [[[[1, 0], [0, 1]], [[0, 1], [-1, 0]], [[F(1, 8), 0], [0, F(1, 8)]]]]
        plan = compile_sparse(a, stages, 0, bounds="fast")
        x = np.arange(6).reshape(3, 2)/5
        tensor = torch.tensor(x, dtype=torch.float64, device="cuda")
        for packed in (False, True):
            prepared = prepare_cuda(plan, packed=packed)
            np.testing.assert_allclose(evaluate_cuda(prepared, tensor).cpu().numpy(), evaluate_sparse(plan, x), atol=1e-13)
            executor = CudaExecutor(prepared, tensor)
            first = executor(tensor)
            saved = first.clone()
            second = executor(tensor*2)
            torch.testing.assert_close(first, saved)
            torch.testing.assert_close(second, saved*2)
            with self.assertRaises(ValueError): executor(torch.full_like(tensor, float("nan")))

    def test_cuda_directional_derivative_and_input_autograd(self):
        a = csr_array([[0., 0.25], [0.125, 0.]])
        e = csr_array([[0.25, 0.125], [0., -0.125]])
        coefficients = [[[1.]], [[0.5]], [[0.125]], [[0.0625]]]
        plan = compile_sensitivity(a, e, [coefficients], 0, 0)
        prepared = prepare_cuda(plan)
        x = torch.tensor([[1.], [2.]], dtype=torch.float64, device="cuda", requires_grad=True)
        y, dy = evaluate_cuda(prepared, x)
        cpu_y, cpu_dy = evaluate_sensitivity(plan, x.detach().cpu().numpy())
        np.testing.assert_allclose(y.detach().cpu().numpy(), cpu_y, atol=1e-13)
        np.testing.assert_allclose(dy.detach().cpu().numpy(), cpu_dy, atol=1e-13)
        y.sum().backward()
        operator = sum(float(c[0][0])*np.linalg.matrix_power(a.toarray(), k) for k, c in enumerate(coefficients))
        np.testing.assert_allclose(x.grad.cpu().numpy().ravel(), operator.T @ np.ones(2), atol=1e-13)
        executor = CudaExecutor(prepared, x.detach())
        actual_y, actual_dy = executor(x.detach())
        np.testing.assert_allclose(actual_y.cpu().numpy(), cpu_y, atol=1e-13)
        np.testing.assert_allclose(actual_dy.cpu().numpy(), cpu_dy, atol=1e-13)
        robust = compile_sensitivity(a, e, [coefficients], 0, 0, parameter_radius=F(1, 4))
        prepared_robust = prepare_cuda(robust)
        executor = CudaExecutor(prepared_robust, x.detach())
        for theta in (-0.25, 0.125, 0.25):
            actual_y, actual_dy = executor(x.detach(), theta=theta)
            expected_y, expected_dy = evaluate_sensitivity(robust, x.detach().cpu().numpy(), theta=theta)
            np.testing.assert_allclose(actual_y.cpu().numpy(), expected_y, atol=1e-13)
            np.testing.assert_allclose(actual_dy.cpu().numpy(), expected_dy, atol=1e-13)
        with self.assertRaises(ValueError): executor(x.detach(), theta=0.5)

    def test_cuda_resolvent_and_float32_opt_in(self):
        plan = compile_ppr(csr_array([[0., 1.], [1., 0.]]), F(1, 5), F(1, 10**10))
        x = np.array([[1.], [0.]])
        for dtype in (torch.float32, torch.float64):
            prepared = prepare_cuda(plan, dtype=dtype)
            tensor = torch.tensor(x, dtype=dtype, device="cuda")
            actual = evaluate_cuda(prepared, tensor).cpu().numpy()
            np.testing.assert_allclose(actual, evaluate_ppr(plan, x), atol=2e-7 if dtype == torch.float32 else 1e-13)


if __name__ == "__main__": unittest.main()
