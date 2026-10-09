"""A GPU demo of one plan reused across certified changing graph weights."""
from fractions import Fraction as F
import numpy as np
from scipy.sparse import csr_array
import torch

from cuda_c2fuse import prepare_cuda, CudaExecutor
from sensitivity_c2fuse import compile_sensitivity, evaluate_sensitivity


def main():
    a = csr_array([[-0.2, 1., 0.], [0., -0.1, 0.125], [0.125, 0., -0.2]])
    direction = csr_array(np.diag([1., -1., 1.]))
    coefficients = [np.eye(2, dtype=int)] + [
        [[F(1, 3**k), F((-1)**k, 4**k)], [F(1, 5**k), F(1, 3**k)]]
        for k in range(1, 33)]
    plan = compile_sensitivity(a, direction, [coefficients], F(1, 10**8), F(1, 10**8),
                               parameter_radius=F(1, 64))
    x = np.array([[1., 2.], [-1., 3.], [2., 0.]])
    tensor = torch.tensor(x, device="cuda", dtype=torch.float64)
    executor = CudaExecutor(prepare_cuda(plan, packed=True), tensor)
    for theta in (-1/64, 0., 1/64):
        y, derivative = executor(tensor, theta=theta)
        cpu_y, cpu_derivative = evaluate_sensitivity(plan, x, theta=theta)
        np.testing.assert_allclose(y.cpu().numpy(), cpu_y, atol=1e-13)
        np.testing.assert_allclose(derivative.cpu().numpy(), cpu_derivative, atol=1e-13)
        print(f"theta={theta:+.6f}; output norm={float(torch.linalg.norm(y)):.8f}; derivative norm={float(torch.linalg.norm(derivative)):.8f}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Degree: {plan['original_degree']} -> {plan['degree']}")
    print(f"Uniform exact value error/input norm <= {float(plan['error_bound']):.12g}")
    print(f"Uniform exact derivative error/input norm <= {float(plan['derivative_error_bound']):.12g}")


if __name__ == "__main__": main()
