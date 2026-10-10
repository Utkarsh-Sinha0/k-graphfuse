"""Reuse the validated fused workspace with lower-cost FP64 accumulators."""
from pathlib import Path
import cupy as cp
from fused_dd import prepare_fused,evaluate_fused

def prepare_acc(compiled,*,compensated=True):
    prepared=prepare_fused(compiled)
    source=f'#define COMPENSATE {int(bool(compensated))}\n'+Path(__file__).with_suffix('.cu').read_text()
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false'),name_expressions=('horner','finish'))
    prepared['kernels'].update({name:module.get_function(name) for name in ('horner','finish')})
    prepared['accumulator']='compensated' if compensated else 'fma'
    # The accumulator kernels never access four low-state planes.
    for i in (2,4,6,8):prepared['buffers'][i]=cp.empty(1,dtype=cp.float64)
    return prepared

evaluate_acc=evaluate_fused
