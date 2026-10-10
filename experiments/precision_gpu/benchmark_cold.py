"""Cold full-polynomial preparation: same loaded operator, degree and executor."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'outputs/kgf_math_upgrade'),str(root/'outputs/k-graphfuse/src')]
import gc,json
from fractions import Fraction as F
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
import torch
from scipy.sparse import csr_array
from threadpoolctl import threadpool_limits
from sparse_c2fuse import compile_sparse,evaluate_sparse
from cuda_c2fuse import prepare_cuda,CudaExecutor
from metric_prototype import compile_full

torch.set_num_threads(1);threadpool_limits(1);torch.backends.cuda.matmul.allow_tf32=False
report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(0),
    protocol='11 randomized cold samples/method; loaded normalized CSR through validation, full compilation, upload, first execution and host output; fixed FP64 eager executor; excludes file I/O and initial graph normalization',cases=[])
output=Path(__file__).parent/'COLD_RESULTS.json'
def save():output.write_text(json.dumps(report,indent=2,allow_nan=False))
for n in (10000,100000,1000000):
    rng=np.random.default_rng(20261010+n);columns=np.repeat(np.arange(n),5);rows=rng.integers(0,n,len(columns));weights=rng.uniform(.1,1.,len(columns))
    weights/=weights.reshape(n,5).sum(axis=1).repeat(5)
    a=csr_array((weights,(rows,columns)),shape=(n,n));a.sum_duplicates();a.sort_indices()
    x=np.full((n,1),1/n);beta=F(85,100)
    bs=[[[((1-beta)*beta**k if k<10 else beta**10)]] for k in range(11)]
    reference=evaluate_sparse(compile_full(a,[bs]),x);norm=np.linalg.norm(x)
    case=dict(n=n,stored_entries=a.nnz,degree=10,methods={name:[] for name in ('released','zero_tail_fastpath')});report['cases'].append(case);save()
    for sample in range(11):
        for method in rng.permutation(['released','zero_tail_fastpath']):
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();start=perf_counter()
            plan=compile_full(a,[bs]) if method=='zero_tail_fastpath' else compile_sparse(a,[bs],0,bounds='fast')
            compiled=perf_counter();prepared=prepare_cuda(plan,dtype=torch.float64,packed=False)
            tensor=torch.tensor(x,dtype=torch.float64,device='cuda');executor=CudaExecutor(prepared,tensor,capture=False)
            torch.cuda.synchronize();ready=perf_counter();actual=executor(tensor).cpu().numpy();torch.cuda.synchronize();end=perf_counter()
            error=float(np.linalg.norm(actual-reference)/norm)
            record=dict(sample=sample,total_ms=1000*(end-start),compile_ms=1000*(compiled-start),prepare_ms=1000*(ready-compiled),execute_host_ms=1000*(end-ready),error_per_input_norm=error,accepted=bool(np.isfinite(error) and error<=1e-12))
            assert plan['degree']==10 and record['accepted']
            case['methods'][method].append(record);del plan,prepared,tensor,executor,actual;save()
    print('COLD',n,{name:round(float(np.median([r['total_ms'] for r in samples])),3) for name,samples in case['methods'].items()},flush=True)
report['status']='complete';report['finished_utc']=datetime.now(timezone.utc).isoformat();save()
