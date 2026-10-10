"""Matched FP64 probe/confirmation of fused higher-precision GPU accumulation."""
from pathlib import Path
import sys
base=Path(__file__).parent;sys.path[:0]=[str(base),str(base/'source/core')]
import gc,hashlib,json,os
from fractions import Fraction as F
from time import perf_counter
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cp_csr
import torch
from scipy.sparse import diags
from threadpoolctl import threadpool_limits
from sparse_c2fuse import compile_sparse,evaluate_sparse
from metric_prototype import compile_full
from sensitivity_c2fuse import compile_sensitivity,evaluate_sensitivity
from cuda_c2fuse import prepare_cuda,CudaExecutor
from metric_prototype import compile_metric
from fused_dd import prepare_fused,evaluate_fused
from fused_acc import prepare_acc
from test_metric_prototype import operator
from check_high_precision import oracle,metrics

torch.set_num_threads(1);threadpool_limits(1);torch.backends.cuda.matmul.allow_tf32=False
report=dict(status='running',gpu=torch.cuda.get_device_name(0),cupy=cp.__version__,torch=torch.__version__,scope='Wide-feature final compensated high-only projection; 1e-15 truncation target, high-only coefficient encoding; same input and full CPU/sampled 384-bit oracle; eight configurations, 5x31 held-out samples/method',cases=[],source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),base/'fused_dd.py',base/'fused_dd.cu',base/'fused_acc.py',base/'fused_acc.cu',base/'metric_prototype.py')})
def save():(Path(os.environ['KGF_PRODUCT_OUTPUT'])/'RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False))
def clean():gc.collect();torch.cuda.empty_cache();cp.get_default_memory_pool().free_all_blocks()
def timing(fn,count):
    for _ in range(2):fn();cp.cuda.Device().synchronize()
    samples=[]
    for _ in range(count):
        cp.cuda.Device().synchronize();start=perf_counter();v=fn();cp.cuda.Device().synchronize();samples.append(1000*(perf_counter()-start));del v
    return dict(samples_ms=samples,median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)))
def host(v):return tuple(t.get() if isinstance(t,cp.ndarray) else t.cpu().numpy() for t in (v if isinstance(v,tuple) else (v,)))
for n,q,derivative in ((100000,16,False),):
    rng=np.random.default_rng(20261010+n+q+int(derivative));x=rng.normal(size=(n,q))
    bs=[np.eye(q,dtype=int)]+[np.array([[F(int(v),10*3**k) for v in row] for row in rng.integers(-2,3,(q,q))],dtype=object) for k in range(1,49)]
    a=operator(n);direction=diags(np.where(np.arange(n)%2,-1.,1.),format='csr') if derivative else None
    old=compile_sensitivity(a,direction,[bs],F(1,10**8),F(1,10**8),parameter_radius=F(1,64)) if derivative else compile_sparse(a,[bs],F(1,10**8),bounds='fast')
    new=compile_metric(a,[bs],F(1,10**15),direction=direction,derivative_tolerance=F(1,10**16) if derivative else None,parameter_radius=F(1,64) if derivative else 0)
    reference={theta:tuple(target[:16] for target in oracle(operator(128),bs,x[:128],derivative,theta,384)) for theta in ((0,-F(1,64),F(1,64)) if derivative else (0,))}
    full=compile_sensitivity(a,direction,[bs],0,0,parameter_radius=F(1,64)) if derivative else compile_full(a,[bs])
    full_reference={theta:evaluate_sensitivity(full,x,theta=theta) if derivative else (evaluate_sparse(full,x),) for theta in reference}
    full_norm=float(np.linalg.norm(x))
    row=dict(n=n,q=q,sensitivity=derivative,old_degree=old['degree'],new_degree=new['plan']['degree'],candidates=[],confirmation=[]);report['cases'].append(row);save()
    def factory(spec):
        if spec['method'].startswith('fused_'):
            prepared=prepare_acc(new,compensated=False,projected=True,coefficient_correction=False,final_compensated=True);tensor=cp.asarray(x)
            def raw(theta=0):return evaluate_fused(prepared,tensor,theta=theta,checked=False)
        elif spec['method']=='cupy':
            matrix=cp_csr((cp.asarray(a.data),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.indptr,dtype=cp.int32)),shape=a.shape);tensor=cp.asarray(x);c=[cp.asarray(b.T) for b in old['float_coefficients']];packed=cp.concatenate(c,axis=1) if spec['packed'] else None
            diag=cp.asarray(direction.diagonal())[:,None] if derivative else None
            def raw(theta=0):
                theta=float(theta)
                incoming=tensor.copy();terms=(incoming@packed).reshape(n,len(c),q) if packed is not None else None
                def term(k):return terms[:,k,:] if terms is not None else incoming if k==0 else incoming@c[k]
                y=term(len(c)-1);dy=None
                for k in range(len(c)-2,-1,-1):
                    if derivative:
                        previous_derivative=dy
                        directional=diag*y;dy=directional if dy is None else matrix@dy+directional
                        if theta and dy is not directional:dy=dy+theta*diag*previous_derivative
                    y=matrix@y+term(k)
                    if derivative and theta:y=y+theta*directional
                return (y,dy) if derivative else y
        else:
            prepared=prepare_cuda(old,packed=spec['packed']);tensor=torch.tensor(x,device='cuda',dtype=torch.float64);executor=CudaExecutor(prepared,tensor,capture=spec['capture'])
            def raw(theta=0):
                if derivative:executor.prepared['theta'].fill_(float(theta))
                executor.input.copy_(tensor);return executor.replay_prevalidated()
        graph=None;static=None
        if spec['capture'] and spec['method']!='torch':
            stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                for _ in range(3):raw()
            stream.synchronize()
            with stream:
                stream.begin_capture()
                try:static=raw()
                finally:graph=stream.end_capture()
        def fn():
            if graph is not None:graph.launch();v=static
            else:v=raw()
            if isinstance(v,tuple):return tuple(t.copy() if isinstance(t,cp.ndarray) else t.clone() for t in v)
            return v.copy() if isinstance(v,cp.ndarray) else v.clone()
        def quality():
            result=dict(sampled_errors=[],sampled_max_absolute=[],global_reference_errors=[],finite=True)
            for theta,target in reference.items():
                v=host(fn() if not theta else raw(theta));item=metrics(tuple(t[:16] for t in v),target,x[:16])
                result['sampled_errors'].extend(item['error_per_input_norm']);result['sampled_max_absolute'].extend(item['max_absolute_error'])
                result['global_reference_errors'].extend(float(np.linalg.norm(t-expected)/full_norm) for t,expected in zip(v,full_reference[theta]))
                result['finite'] &= all(np.isfinite(t).all() for t in v)
            result['accepted']=result['finite'] and max(result['global_reference_errors'])<=1e-13
            return result
        return fn,quality
    specs=[dict(method=m,packed=p,capture=c) for m in ('torch','cupy') for p in (False,True) for c in ((False,True) if m=='torch' else (False,))]
    specs+=[dict(method=m,packed=False,capture=c) for m in ('fused_wide_final',) for c in (False,True)]
    rng.shuffle(specs)
    for spec in specs:
        clean();fn=quality=None;record={**spec};row['candidates'].append(record)
        try:
            fn,quality=factory(spec);record['quality']=quality();record['pilot']=timing(fn,7)
        except Exception as error:record['error']=str(error)
        finally:fn=quality=None;clean();save()
    selected=[]
    for method in ('torch','cupy','fused_wide_final'):
        available=[c for c in row['candidates'] if c['method']==method and 'pilot' in c and c['quality']['accepted']]
        if available:
            selected.append(min(available,key=lambda c:c['pilot']['median_ms']));row['confirmation'].append(dict(method=method,configuration={k:selected[-1][k] for k in ('packed','capture')},blocks=[]))
    for block in range(5):
        for i in rng.permutation(len(selected)):
            clean();fn,quality=factory(selected[i]);stats=timing(fn,31);stats.update(block=block,quality=quality());row['confirmation'][i]['blocks'].append(stats);fn=quality=None;save()
    print('FUSED',n,q,derivative,[(r['method'],round(float(np.median([v for b in r['blocks'] for v in b['samples_ms']])),3),r['blocks'][0]['quality']['sampled_errors']) for r in row['confirmation']],flush=True)
    del a,x,bs,old,new,direction,reference,full,full_reference;clean()
report['status']='complete';save()
