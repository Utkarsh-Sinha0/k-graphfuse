"""Validate resident FP64 GPU evaluation before reusing the operator."""
import numpy as np
import torch
from scipy.sparse import csr_array
from k_graphfuse import compile_sparse, evaluate_sparse
from k_graphfuse.gpu import prepare_cuda, CudaExecutor

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA-enabled PyTorch installation and NVIDIA GPU are required")
a = csr_array([[0., 0.5], [0.25, 0.]])
x = np.array([[1.], [2.]])
plan = compile_sparse(a, [[[[1]], [[1]], [[0.5]]]])
prepared = prepare_cuda(plan, dtype=torch.float64, packed=False)
tensor = torch.tensor(x, dtype=torch.float64, device="cuda")
executor = CudaExecutor(prepared, tensor, capture=False)
actual = executor(tensor).cpu().numpy()
np.testing.assert_allclose(actual, evaluate_sparse(plan, x), atol=1e-13)
print("GPU result:", actual.ravel())
