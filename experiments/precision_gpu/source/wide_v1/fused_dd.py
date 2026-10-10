"""Private fused inference executor; conditional tail bounds exclude GPU rounding."""
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import cupy as cp

module=cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text(),options=('--std=c++14','--fmad=false'),name_expressions=('scale_input','horner','finish'))

def _prepare_fused(compiled,kernels,*,low_state):
    from sparse_c2fuse import _canonical_sparse
    plan=compiled['plan'];a=_canonical_sparse(plan['a']);q=plan['float_coefficients'][0].shape[0]
    if max(a.shape[0]*q,a.nnz)>=2**31:raise ValueError('Prototype requires int32 indexing')
    if not 0<q<=16:raise ValueError('Prototype is calibrated for 1 to 16 channels')
    coefficient=plan['coefficients'];high=np.array(coefficient,dtype=np.float64)
    if high.shape!=(plan['degree']+1,q,q) or not np.isfinite(high).all():raise ValueError('Invalid coefficient shape/value')
    low=np.array([[[float(F(v)-F(float(h))) for v,h in zip(row,hrow)] for row,hrow in zip(b,hb)] for b,hb in zip(coefficient,high)])
    if not np.isfinite(high).all() or not np.isfinite(low).all():raise ValueError('Nonfinite coefficient expansion')
    sensitivity='direction' in plan
    e=_canonical_sparse(plan['direction']) if sensitivity else None
    direction=np.asarray(e.diagonal()) if sensitivity else np.zeros(a.shape[0])
    if sensitivity and (e.shape!=a.shape or e.nnz!=np.count_nonzero(direction)):raise ValueError('Prototype direction must be matching and diagonal')
    scale=np.asarray(compiled['scales'],dtype=np.float64)
    if scale.shape!=(a.shape[0],) or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid similarity scales')
    shape=(a.shape[0],q)
    buffers=[cp.empty(1,dtype=cp.float64) for _ in range(11)]
    for i in ((0,1,5)+(2,6) if low_state else (0,1,5)):
        buffers[i]=cp.empty(shape,dtype=cp.float64)
    if sensitivity:
        for i in ((3,7)+(4,8) if low_state else (3,7)):buffers[i]=cp.empty(shape,dtype=cp.float64)
    # Final scaling is in place; the next query overwrites its initial state.
    buffers[9]=buffers[1 if (plan['degree']+1)%2==0 else 5]
    if sensitivity:buffers[10]=buffers[3 if (plan['degree']+1)%2==0 else 7]
    return dict(plan=plan,shape=shape,rows=cp.asarray(a.indptr,dtype=cp.int32),columns=cp.asarray(a.indices,dtype=cp.int32),edges=cp.asarray(a.data),ch=cp.asarray(high),cl=cp.asarray(low),scale=cp.asarray(scale),direction=cp.asarray(direction),buffers=buffers,sensitivity=sensitivity,
        kernels=kernels)

def prepare_fused(compiled):
    return _prepare_fused(compiled,{name:module.get_function(name) for name in ('scale_input','horner','finish')},low_state=True)

def evaluate_fused(prepared,x,*,theta=0,checked=True):
    """One prepared workspace must not execute concurrent queries."""
    if checked:
        if cp.cuda.runtime.getDevice()!=prepared['edges'].device.id:raise ValueError('Select the prepared CUDA device before execution')
        if not isinstance(x,cp.ndarray) or x.dtype!=cp.float64 or x.shape!=prepared['shape'] or x.device.id!=prepared['edges'].device.id or not x.flags.c_contiguous:
            raise ValueError('Input must be contiguous resident FP64 with the prepared shape/device')
        if not bool(cp.isfinite(x).all()):raise ValueError('Input must be finite')
        try:parameter=F(float(theta))
        except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Invalid parameter') from exc
        if abs(parameter)>prepared['plan'].get('parameter_radius',0):raise ValueError('Parameter outside certified interval')
    count=int(x.size);q=prepared['shape'][1];block=(128,);grid=((count+127)//128,)
    incoming,yh,yl,dh,dl,nh,nl,ndh,ndl,output,doutput=prepared['buffers'];kernel=prepared['kernels']
    kernel['scale_input'](grid,block,(x,prepared['scale'],incoming,np.int32(count),np.int32(q)))
    if checked and not bool(cp.isfinite(incoming).all()):raise ArithmeticError('Scaled input overflow')
    if 'projection' in prepared:
        import torch
        with torch.cuda.stream(torch.cuda.ExternalStream(cp.cuda.get_current_stream().ptr,device=prepared['edges'].device.id)),torch.inference_mode():
            torch.matmul(prepared['incoming_torch'],prepared['packed_high_torch'],out=prepared['projection_torch'])
            for k in range(prepared['plan']['degree']+1) if prepared.get('coefficient_correction',True) else ():
                torch.matmul(prepared['incoming_torch'],prepared['packed_low_torch'][:,k*q:(k+1)*q],out=prepared['correction_torch'])
                target=prepared['projection_torch'][:,k*q:(k+1)*q]
                torch.add(target,prepared['correction_torch'],out=target)
        operand=prepared['projection']
    else:operand=incoming
    for k in range(prepared['plan']['degree'],-1,-1):
        step=kernel['horner_final'] if k==0 and 'horner_final' in kernel else kernel['horner']
        step(grid,block,(prepared['rows'],prepared['columns'],prepared['edges'],operand,prepared['ch'],prepared['cl'],yh,yl,dh,dl,nh,nl,ndh,ndl,prepared['direction'],np.int32(count),np.int32(q),np.int32(k),np.int32(k==prepared['plan']['degree']),np.int32(prepared['sensitivity']),np.float64(theta)))
        yh,nh=nh,yh;yl,nl=nl,yl;dh,ndh=ndh,dh;dl,ndl=ndl,dl
    kernel['finish'](grid,block,(yh,yl,prepared['scale'],output,np.int32(count),np.int32(q)))
    if prepared['sensitivity']:kernel['finish'](grid,block,(dh,dl,prepared['scale'],doutput,np.int32(count),np.int32(q)))
    values=(output,doutput) if prepared['sensitivity'] else (output,)
    if checked and any(not bool(cp.isfinite(value).all()) for value in values):raise ArithmeticError('Fused execution produced nonfinite output')
    # Benchmark callers may use the reused workspace; public callers get owned output.
    result=tuple(value.copy() for value in values) if checked else values
    return result if prepared['sensitivity'] else result[0]
