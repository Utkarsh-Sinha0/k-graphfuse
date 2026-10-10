"""Independent sparse Horner oracle; full small cases, 256/384-bit cross-check."""
from pathlib import Path
import sys
base=Path(__file__).parent
sys.path[:0]=[str(base),str(base/'source/core')]
import hashlib,json
from fractions import Fraction as F
import numpy as np
import mpmath as mp
import torch
from scipy.sparse import diags
from sparse_c2fuse import compile_sparse,evaluate_sparse
from sensitivity_c2fuse import compile_sensitivity,evaluate_sensitivity
from cuda_c2fuse import prepare_cuda,evaluate_cuda
from metric_prototype import compile_metric,evaluate_metric,prepare_metric_cuda,evaluate_metric_cuda
from test_metric_prototype import operator,coefficients

mp.mp.prec=384
def oracle(a,bs,x,derivative,theta,precision):
    with mp.workprec(precision):
        xx=[[mp.mpf(float(v)) for v in row] for row in x]
        cc=[[[mp.mpf(int(F(v).numerator))/int(F(v).denominator) for v in row] for row in b] for b in bs]
        rows=[[(int(a.indices[p]),mp.mpf(float(a.data[p]))) for p in range(a.indptr[i],a.indptr[i+1])] for i in range(a.shape[0])]
        th=mp.mpf(F(theta).numerator)/F(theta).denominator
        ee=[mp.mpf(-1 if i%2 else 1) for i in range(a.shape[0])]
        def right(k):return [[mp.fsum(v*cc[k][j][l] for l,v in enumerate(row)) for j in range(len(row))] for row in xx]
        def left(y):return [[mp.fsum(w*y[col][j] for col,w in row)+(th*ee[i]*y[i][j] if derivative else 0) for j in range(x.shape[1])] for i,row in enumerate(rows)]
        y=right(len(cc)-1);dy=[[mp.mpf(0) for _ in row] for row in y]
        for k in range(len(cc)-2,-1,-1):
            if derivative:
                previous=left(dy);dy=[[previous[i][j]+ee[i]*y[i][j] for j in range(x.shape[1])] for i in range(len(y))]
            previous=left(y);term=right(k);y=[[p+t for p,t in zip(row,trow)] for row,trow in zip(previous,term)]
        return (y,dy) if derivative else (y,)

def metrics(values,reference,x):
    norm=mp.sqrt(mp.fsum(mp.mpf(float(v))**2 for v in x.ravel()))
    errors=[];absolute=[]
    for value,target in zip(values,reference):
        deltas=[abs(mp.mpf(float(value[i,j]))-target[i][j]) for i in range(value.shape[0]) for j in range(value.shape[1])]
        errors.append(float(mp.sqrt(mp.fsum(e*e for e in deltas))/norm));absolute.append(float(max(deltas)))
    return dict(error_per_input_norm=errors,max_absolute_error=absolute)

def run_small_oracle():
    report=dict(scope='Full 32-vertex, four-channel constructed nonnormal operators; independent high-precision accuracy only, not large-workload latency',precision_bits=[256,384],cases=[],source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),base/'metric_prototype.py')})
    for derivative in (False,True):
        n=32;q=4;a=operator(n);bs=coefficients(q);x=np.random.default_rng(20261010).normal(size=(n,q))
        direction=diags(np.where(np.arange(n)%2,-1.,1.),format='csr') if derivative else None
        plans={}
        plans['released']=compile_sensitivity(a,direction,[bs],F(1,10**8),F(1,10**8),parameter_radius=F(1,64)) if derivative else compile_sparse(a,[bs],F(1,10**8),bounds='fast')
        for name,budget in (('metric',F(1,10**8)),('metric_12',F(1,10**12)),('metric_14',F(1,10**14)),('metric_16',F(1,10**16)),('metric_full',F(0))):
            plans[name]=compile_metric(a,[bs],budget,direction=direction,derivative_tolerance=budget if derivative else None,parameter_radius=F(1,64) if derivative else 0)
        tensor=torch.tensor(x,device='cuda',dtype=torch.float64)
        for theta in (-F(1,64),F(0),F(1,64)) if derivative else (F(0),):
            low=oracle(a,bs,x,derivative,theta,256);high=oracle(a,bs,x,derivative,theta,384)
            discrepancy=max(abs(v-w) for a1,a2 in zip(low,high) for row1,row2 in zip(a1,a2) for v,w in zip(row1,row2))
            assert discrepancy<mp.mpf('1e-65')
            rounded=tuple(np.array([[float(v) for v in row] for row in target]) for target in high)
            row=dict(sensitivity=derivative,theta=str(theta),oracle_max_discrepancy=float(discrepancy),nearest_float64_floor=metrics(rounded,high,x),methods=[])
            for name,plan in plans.items():
                cpu=evaluate_metric(plan,x,theta=theta) if name.startswith('metric') else evaluate_sensitivity(plan,x,theta=theta) if derivative else evaluate_sparse(plan,x)
                row['methods'].append(dict(method=name,backend='CPU',degree=plan['plan']['degree'] if name.startswith('metric') else plan['degree'],quality=metrics(cpu if isinstance(cpu,tuple) else (cpu,),high,x)))
                for packed in (False,True):
                    prepared=prepare_metric_cuda(plan,packed=packed) if name.startswith('metric') else prepare_cuda(plan,packed=packed)
                    actual=evaluate_metric_cuda(prepared,tensor,theta=theta) if name.startswith('metric') else evaluate_cuda(prepared,tensor,theta=theta)
                    values=actual if isinstance(actual,tuple) else (actual,)
                    row['methods'].append(dict(method=name,backend='CUDA',packed=packed,degree=plan['plan']['degree'] if name.startswith('metric') else plan['degree'],quality=metrics(tuple(v.cpu().numpy() for v in values),high,x)))
            report['cases'].append(row)
    report['status']='complete'
    (base/'HIGH_PRECISION_RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Independent 256/384-bit oracle: four complete cases, all cross-checks below 1e-65')

if __name__=='__main__':run_small_oracle()
