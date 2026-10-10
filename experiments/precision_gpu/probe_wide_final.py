"""Pilot only: compensate the final high-only projected accumulation."""
from pathlib import Path
import sys,json,hashlib
base=Path(__file__).parent;sys.path[:0]=[str(base),str(base/'source/core')]
from fused_acc import prepare_acc,evaluate_acc
from metric_prototype import compile_metric
from test_metric_prototype import operator
from fractions import Fraction as F
from check_high_precision import oracle,metrics
import cupy as cp,numpy as np
from time import perf_counter
n=100000;q=16;rng=np.random.default_rng(20261010+n+q);x=rng.normal(size=(n,q))
b=[np.eye(q,dtype=int)]+[np.array([[F(int(v),10*3**k) for v in row] for row in rng.integers(-2,3,(q,q))],dtype=object) for k in range(1,49)]
reference=tuple(t[:16] for t in oracle(operator(128),b,x[:128],False,0,384));p=compile_metric(operator(n),[b],F(1,10**15))
prepared=prepare_acc(p,compensated=False,projected=True,coefficient_correction=False,final_compensated=True);tensor=cp.asarray(x)
v=evaluate_acc(prepared,tensor);quality=metrics((v.get()[:16],),reference,x[:16]);times=[]
for _ in range(11):
    cp.cuda.Device().synchronize();start=perf_counter();v=evaluate_acc(prepared,tensor,checked=False).copy();cp.cuda.Device().synchronize();times.append(1000*(perf_counter()-start))
r=dict(scope='Pilot only, no held-out comparison',degree=p['plan']['degree'],quality=quality,median_ms=float(np.median(times)),samples_ms=times,source_hashes={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ('probe_wide_final.py','fused_dd.py','fused_acc.py','fused_acc.cu')});print(r)
(base/'WIDE_FINAL_COMP_PROBE.json').write_text(json.dumps(r,indent=2))
