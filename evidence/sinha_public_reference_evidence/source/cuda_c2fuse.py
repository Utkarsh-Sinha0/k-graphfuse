"""Optional CUDA execution: resident sparse operators, packed coefficients, graphs.

Certificates remain exact-arithmetic certificates. GPU rounding is additional.
Functional evaluation supports input autograd; captured executors are inference-only.
"""
from types import MappingProxyType
import warnings
from fractions import Fraction

import numpy as np
import torch


def _csr(a, dtype):
    index_dtype = np.int32 if max(a.shape[0], a.shape[1], a.nnz) < 2**31 else np.int64
    rows = torch.as_tensor(np.array(a.indptr, dtype=index_dtype), device="cuda")
    columns = torch.as_tensor(np.array(a.indices, dtype=index_dtype), device="cuda")
    values = torch.as_tensor(np.array(a.data, copy=True), dtype=dtype, device="cuda")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        # PyTorch reads its unset global default before honoring the explicit check.
        warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
        return torch.sparse_csr_tensor(rows, columns, values, size=a.shape,
                                       device="cuda", check_invariants=True)


def prepare_cuda(plan, *, dtype=torch.float64, packed=False):
    """Upload once. Float32 is opt-in and requires independent error acceptance."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU and CUDA-enabled PyTorch are required")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32 or float64")
    key = "a" if "a" in plan else "s"
    prepared = {"operator": _csr(plan[key], dtype), "dtype": dtype,
                "source_plan": plan, "kind": "polynomial" if key == "a" else "resolvent"}
    if "direction" in plan:
        diagonal = np.asarray(plan["direction"].diagonal())
        if plan["direction"].nnz == np.count_nonzero(diagonal):
            prepared["direction"] = None
            prepared["direction_diagonal"] = torch.as_tensor(diagonal.copy(), dtype=dtype, device="cuda").reshape(-1, 1)
        else:
            prepared["direction"] = _csr(plan["direction"], dtype)
        prepared["kind"] = "sensitivity"
        prepared["theta"] = torch.tensor(0., dtype=dtype, device="cuda")
    if key == "a":
        coefficients, actions = [], []
        for b in plan["float_coefficients"]:
            tensor = torch.as_tensor(np.array(b, copy=True), dtype=dtype, device="cuda")
            coefficients.append(tensor.T.contiguous())
            if not np.any(b):
                actions.append(("zero", None))
            elif not np.any(b-np.diag(np.diag(b))):
                actions.append(("diagonal", torch.as_tensor(np.diag(b).copy(), dtype=dtype, device="cuda")))
            else:
                actions.append(("dense", coefficients[-1]))
        prepared.update({"coefficients": tuple(coefficients), "actions": tuple(actions),
                         "channels": coefficients[0].shape[0],
                         "packed": torch.cat(coefficients, dim=1) if packed else None})
    elif packed:
        raise ValueError("Packing applies only to matrix-valued polynomial coefficients")
    return MappingProxyType(prepared)


def _validate(prepared, x):
    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.device.index != prepared["operator"].device.index:
        raise ValueError("Input must be a tensor on the prepared CUDA device")
    if x.dtype != prepared["dtype"] or x.ndim != 2 or x.shape[0] != prepared["operator"].shape[0]:
        raise ValueError("Input dtype/shape must match the prepared operator")
    if x.shape[1] == 0 or ("channels" in prepared and x.shape[1] != prepared["channels"]):
        raise ValueError("Invalid channel count")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Input must be finite")


def _kernel(prepared, x):
    """No host synchronization inside this function; safe for graph capture."""
    operator, plan = prepared["operator"], prepared["source_plan"]
    if prepared["kind"] == "resolvent":
        y = plan["scale"]*x
        if plan["degree"]:
            previous, current = x, torch.sparse.mm(operator, x)/float(plan["radius"])
            y = y + 2*plan["scale"]*plan["ratio"]*current
            for k in range(2, plan["degree"]+1):
                previous, current = current, 2*torch.sparse.mm(operator, current)/float(plan["radius"])-previous
                y = y + 2*plan["scale"]*plan["ratio"]**k*current
        return y
    packed = prepared["packed"]
    if packed is not None:
        terms = (x @ packed).reshape(x.shape[0], len(prepared["actions"]), x.shape[1])
        def term(k):
            return terms[:, k, :]
    else:
        def term(k):
            kind, coefficient = prepared["actions"][k]
            if kind == "zero":
                return torch.zeros_like(x)
            if kind == "diagonal":
                return x*coefficient
            return x @ coefficient
    y = term(plan["degree"])
    derivative = None
    def directional_product(value):
        if "direction_diagonal" in prepared:
            return prepared["direction_diagonal"]*value
        return torch.sparse.mm(prepared["direction"], value)
    for k in range(plan["degree"]-1, -1, -1):
        if "direction" in prepared:
            directional = directional_product(y)
            if derivative is None:
                derivative = directional
            elif plan["parameter_radius"]:
                derivative = torch.sparse.mm(operator, derivative)+prepared["theta"]*directional_product(derivative)+directional
            else:
                derivative = torch.sparse.mm(operator, derivative)+directional
        y = torch.sparse.mm(operator, y)+term(k)
        if prepared["kind"] == "sensitivity" and plan["parameter_radius"]:
            y = y+prepared["theta"]*directional
    if "direction" in prepared:
        return y, torch.zeros_like(y) if derivative is None else derivative
    return y


def _check_output(result):
    values = result if isinstance(result, tuple) else (result,)
    if any(not bool(torch.isfinite(v).all()) for v in values):
        raise ArithmeticError("CUDA execution produced nonfinite values")


def _theta(prepared, theta):
    try:
        theta = float(theta)
        valid = abs(Fraction(theta)) <= prepared["source_plan"].get("parameter_radius", 0)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError("Theta must be finite and certified") from exc
    if not valid:
        raise ValueError("Theta is outside the certified interval")
    return theta


def evaluate_cuda(prepared, x, *, theta=0):
    """Checked functional execution, including input-autograd support."""
    _validate(prepared, x)
    theta = _theta(prepared, theta)
    if prepared["kind"] == "sensitivity" and prepared["source_plan"]["parameter_radius"]:
        prepared = {**prepared, "theta": torch.tensor(theta, dtype=x.dtype, device=x.device)}
    result = _kernel(prepared, x)
    _check_output(result)
    return result


class CudaExecutor:
    """Fixed-shape captured inference. One instance must not run concurrently."""
    def __init__(self, prepared, example, *, capture=True):
        _validate(prepared, example)
        if example.requires_grad:
            raise ValueError("Use evaluate_cuda for autograd")
        if "theta" in prepared:
            prepared = {**prepared, "theta": prepared["theta"].clone()}
        self.prepared = prepared
        self.input = torch.empty_like(example)
        self.input.copy_(example)
        self.graph = None
        with torch.inference_mode():
            if capture:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _kernel(prepared, self.input)
                torch.cuda.current_stream().wait_stream(stream)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.output = _kernel(prepared, self.input)
            else:
                self.output = _kernel(prepared, self.input)

    def replay_prevalidated(self):
        """Benchmark/internal entry point: input already validated and resident."""
        with torch.inference_mode():
            if self.graph is not None:
                self.graph.replay()
            else:
                self.output = _kernel(self.prepared, self.input)
        return self.output

    def __call__(self, x, *, theta=0):
        _validate(self.prepared, x)
        if x.shape != self.input.shape or x.requires_grad:
            raise ValueError("Captured input must have the fixed shape and no autograd")
        with torch.inference_mode():
            theta = _theta(self.prepared, theta)
            if self.prepared["kind"] == "sensitivity" and self.prepared["source_plan"]["parameter_radius"]:
                self.prepared["theta"].fill_(theta)
            self.input.copy_(x)
            result = self.replay_prevalidated()
            _check_output(result)
            if isinstance(result, tuple):
                return tuple(v.clone() for v in result)
            return result.clone()
