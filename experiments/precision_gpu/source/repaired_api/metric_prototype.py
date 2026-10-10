"""Private research prototype; no public API or numerical-roundoff certificate."""
from fractions import Fraction as F
from types import MappingProxyType
import numpy as np
from scipy.sparse import csr_array
from c2fuse import fuse, normalize
from sparse_c2fuse import _canonical_sparse, compile_sparse, evaluate_sparse
from sensitivity_c2fuse import compile_sensitivity, evaluate_sensitivity

def compile_full(a, stages):
    """Retain every fused coefficient: the discarded tail is identically zero."""
    a=_canonical_sparse(a);stages=[normalize(stage) for stage in stages];coefficients=fuse(stages)
    fused_degree=len(coefficients)-1
    if not a.nnz:coefficients=coefficients[:1]
    floats=tuple(np.asarray(b,dtype=np.float64) for b in coefficients)
    if any(not np.isfinite(b).all() for b in floats):raise ValueError('Coefficients must fit finite float64')
    for array in (a.data,a.indices,a.indptr,*coefficients,*floats):array.setflags(write=False)
    return MappingProxyType(dict(a=a,coefficients=tuple(coefficients),float_coefficients=floats,
        degree=len(coefficients)-1,fused_degree=fused_degree,
        original_degree=sum(len(s)-1 for s in stages),stage_count=len(stages),
        error_bound=F(0),tolerance=F(0),constant=F(2),norm_radius=None,radius=None,
        bound_method='No discarded terms',geometry_method='Not needed for full evaluation'))

def _similarity(a, exponents):
    coo=a.tocoo();shift=exponents[coo.col]-exponents[coo.row]
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        data=np.ldexp(coo.data,shift);restored=np.ldexp(data,-shift)
    if not np.isfinite(data).all() or not np.array_equal(restored,coo.data):
        raise ValueError('Dyadic similarity must preserve stored entries exactly')
    return csr_array((data,(coo.row,coo.col)),shape=a.shape)

def _metric(a):
    off=a.copy();off.setdiag(0);off.eliminate_zeros();off.data=np.abs(off.data)
    with np.errstate(over='ignore',invalid='ignore'):
        rows=np.asarray(off.sum(axis=1)).ravel();cols=np.asarray(off.sum(axis=0)).ravel()
    if not np.isfinite(rows).all() or not np.isfinite(cols).all():raise ValueError('Balancing reduction overflow')
    exponents=np.zeros(a.shape[0],dtype=np.int32);valid=(rows>0)&(cols>0)
    # One parallel balancing step; dyadic scaling avoids new edge-rounding error.
    exponents[valid]=np.ceil((np.log2(rows[valid])-np.log2(cols[valid]))/4).astype(np.int32)
    exponents-=int(np.max(exponents))
    with np.errstate(under='ignore'):
        scales=np.ldexp(np.ones(a.shape[0]),exponents)
    if (scales==0).any():raise ValueError('Balancing scale underflow')
    penalty=F(2)**int(np.max(exponents)-np.min(exponents))
    return exponents,scales,penalty

def compile_metric(a, stages, tolerance, *, direction=None, derivative_tolerance=None, parameter_radius=0):
    """D^-1 A D with budgets divided by kappa(D); scale input/output explicitly."""
    a=_canonical_sparse(a);exponents,scales,penalty=_metric(a);balanced=_similarity(a,exponents)
    tolerance=F(tolerance)
    if tolerance<0:raise ValueError('Tolerance must be nonnegative')
    if direction is None:
        plan=compile_sparse(balanced,stages,tolerance/penalty,bounds='fast')
    else:
        direction=_canonical_sparse(direction)
        if direction.shape!=a.shape:raise ValueError('Direction must match operator')
        if derivative_tolerance is None:raise ValueError('Derivative budget is required')
        plan=compile_sensitivity(balanced,_similarity(direction,exponents),stages,tolerance/penalty,
            F(derivative_tolerance)/penalty,parameter_radius=parameter_radius,bounds='fast')
    scales.setflags(write=False)
    return MappingProxyType(dict(plan=plan,scales=scales,condition_penalty=penalty,
        original_error_bound=penalty*plan['error_bound'],original_budget=tolerance,
        original_derivative_error_bound=penalty*plan['derivative_error_bound'] if direction is not None else None))

def evaluate_metric(compiled, x, *, theta=0):
    if np.iscomplexobj(x):raise ValueError('State must be real')
    x=np.asarray(x,dtype=np.float64);scales=compiled['scales'][:,None]
    if x.ndim!=2 or x.shape[0]!=len(scales) or not np.isfinite(x).all():raise ValueError('Invalid state')
    with np.errstate(over='ignore',invalid='ignore'):
        incoming=x/scales
    if not np.isfinite(incoming).all():raise ArithmeticError('Scaled input overflow')
    plan=compiled['plan']
    if 'direction' in plan:
        values=evaluate_sensitivity(plan,incoming,theta=theta)
        result=tuple(scales*v for v in values)
    else:result=scales*evaluate_sparse(plan,incoming)
    if any(not np.isfinite(v).all() for v in (result if isinstance(result,tuple) else (result,))):
        raise ArithmeticError('Scaled output overflow')
    return result

def prepare_metric_cuda(compiled, *, dtype=None, packed=False):
    """Optional checked CUDA path; scales and operator are uploaded once."""
    import torch
    from cuda_c2fuse import prepare_cuda
    prepared=prepare_cuda(compiled['plan'],dtype=torch.float64 if dtype is None else dtype,packed=packed)
    scales=torch.as_tensor(compiled['scales'].copy(),device='cuda',dtype=prepared['dtype']).reshape(-1,1)
    return MappingProxyType(dict(prepared=prepared,scales=scales))

def evaluate_metric_cuda(prepared, x, *, theta=0):
    """Reuse original input validation and validate the scaled state/output."""
    from cuda_c2fuse import _validate,evaluate_cuda
    _validate(prepared['prepared'],x)
    result=evaluate_cuda(prepared['prepared'],x/prepared['scales'],theta=theta)
    if isinstance(result,tuple):return tuple(v*prepared['scales'] for v in result)
    return result*prepared['scales']
