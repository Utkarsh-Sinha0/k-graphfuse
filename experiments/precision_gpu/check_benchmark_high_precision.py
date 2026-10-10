"""Exact finite dependency windows validate selected rows of the large benchmarks."""
from pathlib import Path
import sys
base=Path(__file__).parent;sys.path[:0]=[str(base),str(base/'source/core')]
import hashlib,json
from fractions import Fraction as F
import numpy as np
import mpmath as mp
import torch
from scipy.sparse import diags
from threadpoolctl import threadpool_limits
from sparse_c2fuse import compile_sparse
from sensitivity_c2fuse import compile_sensitivity
from cuda_c2fuse import prepare_cuda,evaluate_cuda
from metric_prototype import compile_metric,prepare_metric_cuda,evaluate_metric_cuda
from test_metric_prototype import operator
from check_high_precision import oracle,metrics

torch.set_num_threads(1);threadpool_limits(1);torch.backends.cuda.matmul.allow_tf32=False
mp.mp.prec=384
report=dict(status='running',scope='First 16 output rows of each exact six-case large-benchmark input. Degree-48 finite influence cones fit in the first 64 rows of this specific forward-coupled operator; not exhaustive high-precision verification of every vertex.',precision_bits=[256,384],cases=[],source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),base/'check_high_precision.py',base/'metric_prototype.py')})
def save():(base/'BENCHMARK_HIGH_PRECISION.json').write_text(json.dumps(report,indent=2,allow_nan=False))
for n,q,derivative in ((10000,4,False),(100000,4,False),(1000000,4,False),(100000,16,False),(100000,4,True),(1000000,4,True)):
    rng=np.random.default_rng(20261010+n+q+int(derivative));x=rng.normal(size=(n,q))
    bs=[np.eye(q,dtype=int)]+[np.array([[F(int(v),10*3**k) for v in row] for row in rng.integers(-2,3,(q,q))],dtype=object) for k in range(1,49)]
    a=operator(n);direction=diags(np.where(np.arange(n)%2,-1.,1.),format='csr') if derivative else None
    plans={'released':compile_sensitivity(a,direction,[bs],F(1,10**8),F(1,10**8),parameter_radius=F(1,64)) if derivative else compile_sparse(a,[bs],F(1,10**8),bounds='fast')}
    for name,budget in (('metric',F(1,10**8)),('metric_12',F(1,10**12)),('metric_14',F(1,10**14)),('metric_16',F(1,10**16)),('metric_full',F(0))):
        plans[name]=compile_metric(a,[bs],budget,direction=direction,derivative_tolerance=budget if derivative else None,parameter_radius=F(1,64) if derivative else 0)
    prepared={name:prepare_metric_cuda(plan,packed=True) if name.startswith('metric') else prepare_cuda(plan,packed=True) for name,plan in plans.items()}
    tensor=torch.tensor(x,device='cuda',dtype=torch.float64)
    for theta in (-F(1,64),F(0),F(1,64)) if derivative else (F(0),):
        first=oracle(operator(64),bs,x[:64],derivative,theta,256)
        second=oracle(operator(128),bs,x[:128],derivative,theta,384)
        high=tuple(target[:16] for target in second)
        discrepancy=max(abs(v-w) for a1,a2 in zip(first,high) for row1,row2 in zip(a1[:16],a2) for v,w in zip(row1,row2))
        assert discrepancy<mp.mpf('1e-65')
        row=dict(n=n,q=q,sensitivity=derivative,theta=str(theta),oracle_window_discrepancy=float(discrepancy),methods=[],normalization='Frobenius norm of the first 16 input rows; sampled-row error, distinct from full-graph error')
        rounded=tuple(np.array([[float(v) for v in r] for r in target]) for target in high)
        row['nearest_float64_floor']=metrics(rounded,high,x[:16])
        for name,item in prepared.items():
            actual=evaluate_metric_cuda(item,tensor,theta=theta) if name.startswith('metric') else evaluate_cuda(item,tensor,theta=theta)
            values=actual if isinstance(actual,tuple) else (actual,)
            row['methods'].append(dict(method=name,quality=metrics(tuple(v[:16].cpu().numpy() for v in values),high,x[:16])))
        report['cases'].append(row);save()
    print('ORACLE',n,q,derivative,flush=True)
    del a,direction,plans,prepared,tensor,x
    torch.cuda.empty_cache()
report['status']='complete';save()
print('Large-input high-precision sampled validation complete',flush=True)
