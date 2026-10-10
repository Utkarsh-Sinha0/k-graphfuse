"""Application confirmation against a specialized scalar diffusion control."""
from pathlib import Path
import argparse, hashlib, json, time
import study as s

HERE=Path(__file__).resolve().parent
METHODS=('cupy_restart','cupy_restart_captured','fused_onepass','torch_identity_packed_captured')


def restart(a,x,captured):
    matrix=csr_matrix((cp.asarray(a.data),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.indptr,dtype=cp.int32)),shape=a.shape)
    validator=dict(kernels={'check_finite':module.get_function('check_finite')},finite_flag=cp.empty(1,dtype=cp.int32))
    alpha=float(s.F(s.PROTOCOL['application']['alpha']));degree=s.PROTOCOL['application']['degree']
    def raw():
        injection=(1-alpha)*x;y=injection.copy()
        for _ in range(degree):y=alpha*(matrix@y)+injection
        return y
    owned=raw
    if captured:
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            for _ in range(3):raw()
        stream.synchronize()
        with stream:
            stream.begin_capture();static=raw();graph=stream.end_capture()
        def owned():graph.launch();return static.copy()
    def checked():
        if not _finite(validator,x):raise ValueError('Nonfinite input')
        y=owned()
        if not _finite(validator,y):raise ArithmeticError('Nonfinite output')
        return (y,)
    return checked


def freeze():
    paths=[HERE/'application.py',HERE/'FREEZE.json',HERE/'runtime.py',HERE/'RUNTIME_FREEZE.json']
    payload=dict(study_id='kgf-recommendation-20261010-v1',source_sha256={p.name:s.digest(p) for p in paths},
        methods=list(METHODS),protocol=s.PROTOCOL['application'],blocks=5,quality_noninferiority_margin=0.001,
        order='Shuffle methods inside each whole-workflow block, fixed seed; no selection of the K profile.',
        equation='y_0=(1-alpha)X; y_{j+1}=alpha A y_j+(1-alpha)X; after d steps this equals the same finite polynomial with c_k=(1-alpha)alpha^k.',
        scope='Specialized baseline uses scalar channel coefficients, avoiding unnecessary GEMMs. Includes checked compute, transfers and top-10 ranking. No recommendation quality-SOTA claim.')
    (HERE/'APPLICATION_FREEZE.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--selftest',action='store_true');p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--output',type=Path);p.add_argument('--cache',type=Path);args=p.parse_args()
    if args.freeze:freeze();return
    if args.selftest:
        a=s.graph('random',64);x=s.np.random.default_rng(1).normal(size=(64,4));alpha=s.F(17,20);d=12
        bs=[s.np.eye(4,dtype=object)*(1-alpha)*alpha**k for k in range(d+1)]
        y=float(1-alpha)*x
        for _ in range(d):y=float(alpha)*(a@y)+float(1-alpha)*x
        s.np.testing.assert_allclose(y,s.cpu(s.compile_case(a,bs,'identity',s.F(0),full=True),x)[0],rtol=1e-13,atol=1e-13)
        print('PASS: specialized restart and finite polynomial represent the same application.');return
    lock=json.loads((HERE/'APPLICATION_FREEZE.json').read_text(encoding='utf-8'))
    assert all(s.digest(HERE/name)==digest for name,digest in lock['source_sha256'].items());s.check_freeze()
    assert args.seed in s.PROTOCOL['seeds'] and args.output is not None
    global cp, torch, csr_matrix, _finite, module
    import cupy as cp
    cp.cuda.nvrtc.getVersion()
    import torch
    torch.set_num_threads(1);torch.cuda.init()
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    from cupyx.scipy.sparse import csr_matrix
    from fused_dd import _finite, module
    started=time.perf_counter();a,offset,items,users,seen,test,data_hash=s.movielens(args.cache);load_ms=1000*(time.perf_counter()-started)
    q=s.PROTOCOL['application']['channels'];alpha=s.F(s.PROTOCOL['application']['alpha']);degree=s.PROTOCOL['application']['degree']
    bs=[s.np.eye(q,dtype=object)*(1-alpha)*alpha**k for k in range(degree+1)]
    batches=[users[i:i+q] for i in range(0,len(users),q)];x=s.np.zeros((a.shape[0],q));reference_lists=[];rr=[];nn=[]
    reference=s.compile_case(a,bs,'identity',s.F(0),full=True)
    for batch in batches:
        x.fill(0);x[batch,s.np.arange(len(batch))]=1
        values=s.cpu(reference,x)[0];r,n,top=s.ranking(values,batch,offset,seen,test)
        rr.extend(r);nn.extend(n);reference_lists.extend(top)
    report=dict(study_id=lock['study_id'],freeze_sha256=s.digest(HERE/'APPLICATION_FREEZE.json'),seed=args.seed,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cupy=cp.__version__,nvrtc=list(cp.cuda.nvrtc.getVersion()),
        dataset_sha256=data_hash,vertices=a.shape[0],nnz=a.nnz,evaluated_users=len(users),load_build_ms=load_ms,
        cpu_reference_recall=float(s.np.mean(rr)),cpu_reference_ndcg=float(s.np.mean(nn)),status='running',methods=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    for name in METHODS:report['methods'].append(dict(method=name,request_batch_ms=[]))
    rng=s.np.random.default_rng(args.seed)
    for block in range(lock['blocks']):
        for index in rng.permutation(len(METHODS)):
            record=report['methods'][index];method=record['method']
            if 'error' in record:continue
            try:
                start=time.perf_counter();compiled=None if method.startswith('cupy_restart') else s.compile_case(a,bs,'onepass' if method=='fused_onepass' else 'identity',s.F(0),full=True)
                compile_ms=0.0 if compiled is None else 1000*(time.perf_counter()-start);xgpu=cp.zeros_like(cp.asarray(x));start=time.perf_counter()
                checked=restart(a,xgpu,method.endswith('_captured')) if method.startswith('cupy_restart') else s.executor(compiled,xgpu,method)[1]
                cp.cuda.get_current_stream().synchronize();prepare_ms=1000*(time.perf_counter()-start)
                rec=[];ndcg=[];lists=[];start=time.perf_counter()
                for batch in batches:
                    x.fill(0);x[batch,s.np.arange(len(batch))]=1;xgpu.set(x)
                    values=checked()[0].get();r,n,top=s.ranking(values,batch,offset,seen,test)
                    rec.extend(r);ndcg.extend(n);lists.extend(top)
                record['request_batch_ms'].append(1000*(time.perf_counter()-start))
                record.update(compile_ms=compile_ms,prepare_ms=prepare_ms,recall_at_10=float(s.np.mean(rec)),ndcg_at_10=float(s.np.mean(ndcg)),
                    top10_exact_agreement=float(s.np.mean(s.np.all(s.np.asarray(lists)==reference_lists,axis=1))))
                record['quality_within_margin']=all(record[key]>=report[ref]-lock['quality_noninferiority_margin'] for key,ref in (
                    ('recall_at_10','cpu_reference_recall'),('ndcg_at_10','cpu_reference_ndcg')))
                del checked,xgpu;s.clean()
            except Exception as error:record['error']=type(error).__name__
            save()
    report['status']='complete';save();print('Application confirmation:',[(r['method'],s.np.median(r['request_batch_ms'])) for r in report['methods'] if r['request_batch_ms']],flush=True)


if __name__=='__main__':main()
