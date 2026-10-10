"""Private matched local GPU experiment; original industry evidence is immutable."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'outputs/kgf_math_upgrade'),str(root/'outputs/k-graphfuse/src')]
import gc,json,hashlib
from fractions import Fraction as F
from datetime import datetime,timezone
from time import perf_counter
import numpy as np
import torch
from scipy.sparse import diags,coo_array
from threadpoolctl import threadpool_limits
from sparse_c2fuse import compile_sparse,evaluate_sparse
from sensitivity_c2fuse import compile_sensitivity,evaluate_sensitivity
from cuda_c2fuse import prepare_cuda,CudaExecutor
from metric_prototype import compile_metric,compile_full
from test_metric_prototype import operator,coefficients
from c2fuse import sqrt_upper

torch.set_num_threads(1);threadpool_limits(1);torch.backends.cuda.matmul.allow_tf32=False
out=root/'outputs/kgf_math_upgrade';out.mkdir(exist_ok=True)
report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
    gpu=torch.cuda.get_device_name(0),torch=torch.__version__,scope='Accuracy/speed Pareto search against the two original-coordinate GPU controls; full CPU reference errors are empirical, not an arbitrary-precision oracle',
    protocol=dict(pilot_samples=7,blocks=5,samples_per_block=31,precision_search=['float64'],capture_search=[False,True],packed_search=[False,True]),cases=[],cold=[])
report['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'metric_prototype.py',Path(__file__),root/'outputs/k-graphfuse/src/sparse_c2fuse.py',root/'outputs/k-graphfuse/src/cuda_c2fuse.py')}
def save():(out/'ACCURACY_FIRST_RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
def cleanup():gc.collect();torch.cuda.empty_cache()
def timing(fn,count):
    for _ in range(2):fn();torch.cuda.synchronize()
    times=[]
    for _ in range(count):
        torch.cuda.synchronize();start=perf_counter();value=fn();torch.cuda.synchronize();times.append(1000*(perf_counter()-start));del value
    return dict(samples_ms=times,sample_count=count,median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)))
def host(result):return tuple(v.detach().cpu().numpy() for v in result) if isinstance(result,tuple) else (result.detach().cpu().numpy(),)

for n,q,sensitivity in ((10000,4,False),(100000,4,False),(1000000,4,False),(100000,16,False),(100000,4,True),(1000000,4,True)):
    a=operator(n);rng=np.random.default_rng(20261010+n+q+int(sensitivity));x=rng.normal(size=(n,q))
    bs=[np.eye(q,dtype=int)]+[np.array([[F(int(v),10*3**k) for v in row] for row in rng.integers(-2,3,(q,q))],dtype=object) for k in range(1,49)]
    direction=diags(np.where(np.arange(n)%2,-1.,1.),format='csr') if sensitivity else None
    budget=F(1,10**8);start=perf_counter();plans={}
    for name,constant in (('released',F(2)),('older_bound',1+sqrt_upper(2))):
        begin=perf_counter()
        plans[name]=compile_sensitivity(a,direction,[bs],budget,budget,parameter_radius=F(1,64),constant=constant) if sensitivity else compile_sparse(a,[bs],budget,bounds='fast',constant=constant)
        plans[name+'_compile_ms']=1000*(perf_counter()-begin)
    begin=perf_counter();metric=compile_metric(a,[bs],budget,direction=direction,derivative_tolerance=budget if sensitivity else None,parameter_radius=F(1,64) if sensitivity else 0)
    plans['metric']=metric['plan'];plans['metric_compile_ms']=1000*(perf_counter()-begin)

    for name,new_budget in (('metric_12',F(1,10**12)),('metric_14',F(1,10**14)),('metric_16',F(1,10**16)),('metric_full',F(0))):
        begin=perf_counter()
        extra=compile_metric(a,[bs],new_budget,direction=direction,derivative_tolerance=new_budget if sensitivity else None,parameter_radius=F(1,64) if sensitivity else 0)
        plans[name]=extra['plan'];plans[name+'_compile_ms']=1000*(perf_counter()-begin)
    if sensitivity:
        full=compile_sensitivity(a,direction,[bs],0,0,parameter_radius=F(1,64));reference=evaluate_sensitivity(full,x)
    else:reference=(evaluate_sparse(compile_full(a,[bs]),x),)
    reference_by_theta={theta:evaluate_sensitivity(full,x,theta=theta) for theta in (-1/64,1/64)} if sensitivity else {}
    norm=float(np.linalg.norm(x));row=dict(n=n,q=q,sensitivity=sensitivity,budget=float(budget),
        degrees={key:plans[key]['degree'] for key in ('released','older_bound','metric','metric_12','metric_14','metric_16','metric_full')},
        compile_ms={key:plans[key+'_compile_ms'] for key in ('released','older_bound','metric','metric_12','metric_14','metric_16','metric_full')},
        condition_penalty=float(metric['condition_penalty']),metric_original_bound=float(metric['original_error_bound']),
        metric_derivative_bound=float(metric['original_derivative_error_bound']) if sensitivity else None,candidates=[],confirmation=[])
    row['tail_bounds']={name:float(plans[name]['error_bound']*(metric['condition_penalty'] if name.startswith('metric') else 1)) for name in ('released','older_bound','metric','metric_12','metric_14','metric_16','metric_full')}
    report['cases'].append(row);save()
    def factory(spec):
        prepared=prepare_cuda(plans[spec['method']],dtype=getattr(torch,spec['precision']),packed=spec['packed'])
        tensor=torch.as_tensor(x,dtype=prepared['dtype'],device='cuda');scales=torch.as_tensor(metric['scales'].copy(),dtype=prepared['dtype'],device='cuda').reshape(-1,1) if spec['method'].startswith('metric') else None
        incoming=tensor if scales is None else tensor/scales
        executor=CudaExecutor(prepared,incoming,capture=spec['capture'])
        def fn():
            executor.input.copy_(tensor if scales is None else tensor/scales)
            values=executor.replay_prevalidated()
            if isinstance(values,tuple):return tuple(v.clone() if scales is None else v*scales for v in values)
            return values.clone() if scales is None else values*scales
        def quality():
            errors=[float(np.linalg.norm(actual-expected)/norm) for actual,expected in zip(host(fn()),reference)]
            if sensitivity:
                for theta in (-1/64,1/64):
                    expected=reference_by_theta[theta]
                    result=executor(incoming,theta=theta)
                    if scales is not None:result=tuple(v*scales for v in result)
                    errors.extend(float(np.linalg.norm(actual-target)/norm) for actual,target in zip(host(result),expected))
                executor.prepared['theta'].fill_(0)
            return dict(errors=errors,accepted=all(np.isfinite(e) and e<=float(budget) for e in errors))
        return fn,quality
    specs=[dict(method=m,precision=p,packed=b,capture=c) for m in ('released','older_bound','metric','metric_12','metric_14','metric_16','metric_full') for p in ('float64',) for b in (False,True) for c in (False,True)]
    rng.shuffle(specs)
    for spec in specs:
        cleanup();record={**spec};row['candidates'].append(record);fn=check=None
        try:
            estimated=x.size*(4 if spec['precision']=='float32' else 8)*(plans[spec['method']]['degree']+1)*3 if spec['capture'] or spec['packed'] else x.size*8*6
            if estimated>.75*torch.cuda.get_device_properties(0).total_memory:record.update(accepted=False,not_run='Conservative workspace estimate');continue
            fn,check=factory(spec);record.update(check());record['accepted']=record['accepted']
            if record['accepted']:record['pilot']=timing(fn,7)
        except Exception as error:record.update(accepted=False,error=repr(error))
        finally:fn=check=None;cleanup();save()
    selected=[]
    for method in ('released','older_bound','metric','metric_12','metric_14','metric_16','metric_full'):
        accepted=[c for c in row['candidates'] if c['method']==method and c.get('accepted')]
        if accepted:
            best=min(accepted,key=lambda c:c['pilot']['median_ms']);selected.append(best)
            row['confirmation'].append(dict(method=method,configuration={k:best[k] for k in ('precision','packed','capture')},blocks=[],accepted=True))
    for block in range(5):
        for index in rng.permutation(len(selected)):
            spec=selected[index];record=row['confirmation'][index];cleanup();fn=check=None
            try:
                fn,check=factory(spec);quality=check();stats=timing(fn,31);stats.update(block=block,quality=quality)
                record['blocks'].append(stats);record['accepted'] &=quality['accepted']
                times=[v for b in record['blocks'] for v in b['samples_ms']]
                record['resident']=dict(samples_ms=times,sample_count=len(times),median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)))
            finally:fn=check=None;cleanup();save()
    print('CASE',n,q,sensitivity,row['degrees'],[(c['method'],round(c['resident']['median_ms'],3),c['accepted']) for c in row['confirmation']],flush=True)
    del a,x,bs,plans,metric,extra,reference,reference_by_theta
    if sensitivity:del full
    del direction

report['status']='complete';report['finished_utc']=datetime.now(timezone.utc).isoformat();save()
