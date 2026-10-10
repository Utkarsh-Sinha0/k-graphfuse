"""Reuse the validated fused workspace with lower-cost FP64 accumulators."""
from pathlib import Path
import cupy as cp
from fused_dd import _prepare_fused,module as base_module,evaluate_fused

def prepare_acc(compiled,*,compensated=True,derivative_compensated=None,projected=False,coefficient_correction=True,final_compensated=False):
    if projected and compiled['plan']['a'].shape[0]*compiled['plan']['float_coefficients'][0].shape[0]*(compiled['plan']['degree']+1)>=2**31:
        raise ValueError('Projected prototype requires int32 workspace indexing')
    derivative_compensated=compensated if derivative_compensated is None else derivative_compensated
    source=f'#define COMPENSATE {int(bool(compensated))}\n#define DERIVATIVE_COMPENSATE {int(bool(derivative_compensated))}\n#define FINAL_COMPENSATE 0\n#define PROJECTED {int(bool(projected))}\n#define STAGES {compiled["plan"]["degree"]+1}\n'+Path(__file__).with_suffix('.cu').read_text()
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false'),name_expressions=('horner','finish'))
    kernels={name:module.get_function(name) for name in ('horner','finish')}
    kernels.update({name:base_module.get_function(name) for name in ('scale_input','check_finite')})
    if final_compensated:
        final_module=cp.RawModule(code=source.replace(f'#define COMPENSATE {int(bool(compensated))}', '#define COMPENSATE 1'),options=('--std=c++14','--fmad=false'),name_expressions=('horner',))
        kernels['horner_final']=final_module.get_function('horner')
    prepared=_prepare_fused(compiled,kernels,low_state=False)
    prepared['accumulator']='mixed' if bool(compensated)!=bool(derivative_compensated) else 'compensated' if compensated else 'fma'
    if projected:
        import torch
        prepared['interop_stream']=cp.cuda.Stream(non_blocking=True)
        prepared['interop_ready']=cp.cuda.Event(disable_timing=True)
        prepared['interop_done']=cp.cuda.Event(disable_timing=True)
        n,q=prepared['shape'];count=compiled['plan']['degree']+1
        prepared['packed_high']=cp.concatenate([b.T for b in prepared['ch']],axis=1)
        prepared['packed_low']=cp.concatenate([b.T for b in prepared['cl']],axis=1)
        prepared['projection']=cp.empty((n,count*q),dtype=cp.float64)
        prepared['projection_correction']=cp.empty((n,q),dtype=cp.float64) if coefficient_correction else prepared['buffers'][0]
        prepared['coefficient_correction']=bool(coefficient_correction)
        for name,array in (('incoming_torch',prepared['buffers'][0]),('packed_high_torch',prepared['packed_high']),('packed_low_torch',prepared['packed_low']),('projection_torch',prepared['projection']),('correction_torch',prepared['projection_correction'])):
            prepared[name]=torch.from_dlpack(array)
    return prepared

evaluate_acc=evaluate_fused
