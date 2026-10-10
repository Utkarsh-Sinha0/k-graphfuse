"""Reuse the validated fused workspace with lower-cost FP64 accumulators."""
from pathlib import Path
import cupy as cp
from fused_dd import _prepare_fused,module as base_module,evaluate_fused

def prepare_acc(compiled,*,compensated=True,derivative_compensated=None,projected=False):
    derivative_compensated=compensated if derivative_compensated is None else derivative_compensated
    source=f'#define COMPENSATE {int(bool(compensated))}\n#define DERIVATIVE_COMPENSATE {int(bool(derivative_compensated))}\n#define PROJECTED {int(bool(projected))}\n#define STAGES {compiled["plan"]["degree"]+1}\n'+Path(__file__).with_suffix('.cu').read_text()
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false'),name_expressions=('horner','finish'))
    kernels={name:module.get_function(name) for name in ('horner','finish')};kernels['scale_input']=base_module.get_function('scale_input')
    prepared=_prepare_fused(compiled,kernels,low_state=False)
    prepared['accumulator']='mixed' if bool(compensated)!=bool(derivative_compensated) else 'compensated' if compensated else 'fma'
    if projected:
        import torch
        n,q=prepared['shape'];count=compiled['plan']['degree']+1
        prepared['packed_high']=cp.concatenate([b.T for b in prepared['ch']],axis=1)
        prepared['packed_low']=cp.concatenate([b.T for b in prepared['cl']],axis=1)
        prepared['projection']=cp.empty((n,count*q),dtype=cp.float64)
        prepared['projection_correction']=cp.empty((n,q),dtype=cp.float64)
        for name,array in (('incoming_torch',prepared['buffers'][0]),('packed_high_torch',prepared['packed_high']),('packed_low_torch',prepared['packed_low']),('projection_torch',prepared['projection']),('correction_torch',prepared['projection_correction'])):
            prepared[name]=torch.from_dlpack(array)
    return prepared

evaluate_acc=evaluate_fused
