"""Frozen same-budget ablations and a held-out recommendation workflow."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gc, hashlib, io, json, os, platform, sys, time, zipfile
from urllib.request import urlopen
import numpy as np
from scipy.sparse import csr_array, diags, kron, eye

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'experiments/precision_gpu'
sys.path[:0] = [str(EXP), str(EXP / 'source/core')]
from metric_prototype import _metric, _similarity, compile_full
from sparse_c2fuse import compile_sparse, evaluate_sparse
from sensitivity_c2fuse import compile_sensitivity, evaluate_sensitivity
PROTOCOL = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = [HERE / 'study.py', HERE / 'protocol.json']
    paths += [p for p in (EXP / 'source/core').glob('*.py')]
    paths += [EXP / n for n in ('metric_prototype.py', 'fused_acc.py', 'fused_acc.cu', 'fused_dd.py', 'fused_dd.cu', 'check_high_precision.py', 'test_metric_prototype.py')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def freeze():
    (HERE / 'FREEZE.json').write_text(json.dumps({'study_id': PROTOCOL['study_id'], 'source_sha256': sources()}, indent=2)+'\n', encoding='utf-8', newline='\n')


def check_freeze():
    lock = json.loads((HERE / 'FREEZE.json').read_text(encoding='utf-8'))
    assert lock['source_sha256'] == sources(), 'Source differs from the pre-confirmation freeze'
    return digest(HERE / 'FREEZE.json')


def graph(family, n):
    rng = np.random.default_rng(PROTOCOL['graph_seed'])
    if family in ('alternating', 'balanced'):
        rows=np.arange(n);even=rows[::2];odd=rows[1::2]
        return csr_array((np.r_[np.tile([-.2,-.1],n//2),np.ones(n//2),np.full(n//2,1/64 if family=='alternating' else 1.)],
            (np.r_[rows,even,odd],np.r_[rows,odd,(odd+1)%n])),shape=(n,n))
    if family=='grid':
        side=int(np.sqrt(n));assert side*side==n
        line=diags([np.ones(side-1),np.ones(side-1)],[-1,1],shape=(side,side),format='csr')
        a=csr_array((kron(eye(side),line)+kron(line,eye(side)))*.1)
    else:
        if family=='rmat':
            bits=int(np.log2(n));assert 2**bits==n
            rows=np.zeros(8*n,dtype=np.int64);cols=rows.copy()
            for bit in range(bits):
                v=rng.random(len(rows));rows|=(v>=.76).astype(np.int64)<<bit
                cols|=((v>=.57)&(v<.76)|(v>=.95)).astype(np.int64)<<bit
        else: rows=np.repeat(np.arange(n),8);cols=rng.integers(0,n,len(rows))
        a=csr_array((np.ones(len(rows)),(rows,cols)),shape=(n,n));a.sum_duplicates()
        sums=np.asarray(a.sum(axis=1)).ravel();a=csr_array(diags(.4/np.maximum(sums,1))@a)
        if family=='similarity_random':
            exponents=rng.integers(-3,4,n,dtype=np.int32);a=_similarity(a,-exponents)
    return csr_array(a-.1*eye(n,format='csr'))


def coefficients(q, seed):
    rng=np.random.default_rng(seed)
    return [np.eye(q,dtype=int)]+[np.array([[F(int(v),10*3**k) for v in row] for row in rng.integers(-2,3,(q,q))],dtype=object)
        for k in range(1,PROTOCOL['degree']+1)]


def coordinates(a, mode):
    exponents=np.zeros(a.shape[0],dtype=np.int32)
    for _ in range(0 if mode=='identity' else 1 if mode=='onepass' else 8):
        step,_,_=_metric(_similarity(a,exponents));exponents+=step
        exponents-=int(np.max(exponents))
    assert int(np.max(exponents)-np.min(exponents))<900
    scales=np.ldexp(np.ones(a.shape[0]),exponents)
    return _similarity(a,exponents),exponents,scales,F(2)**int(np.max(exponents)-np.min(exponents))


def compile_case(a, bs, mode, budget, derivative=False, full=False):
    b,e,scales,penalty=coordinates(a,mode)
    direction=diags(np.where(np.arange(len(scales))%2,-1.,1.),format='csr') if derivative else None
    if derivative:
        plan=compile_sensitivity(b,direction,[bs],budget/penalty,budget/penalty,parameter_radius=F(1,64),bounds='fast')
    else: plan=compile_full(b,[bs]) if full else compile_sparse(b,[bs],budget/penalty,bounds='fast')
    return dict(plan=plan,scales=scales,condition_penalty=penalty,original_error_bound=penalty*plan['error_bound'],
        original_derivative_error_bound=penalty*plan['derivative_error_bound'] if derivative else None)


def cpu(compiled, x):
    plan=compiled['plan'];z=x/compiled['scales'][:,None]
    ys=evaluate_sensitivity(plan,z) if 'direction' in plan else (evaluate_sparse(plan,z),)
    return tuple(y*compiled['scales'][:,None] for y in ys)


def quality(values, reference, x):
    norm=float(np.linalg.norm(x))
    return dict(frobenius=[float(np.linalg.norm(v-r)/norm) for v,r in zip(values,reference)],
        max_absolute=[float(np.max(np.abs(v-r))) for v,r in zip(values,reference)],finite=all(np.isfinite(v).all() for v in values))


def executor(compiled, x, method):
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    from fused_acc import prepare_acc, evaluate_acc
    from fused_dd import module, _finite
    plan=compiled['plan'];scale=cp.asarray(compiled['scales'])[:,None];n,q=x.shape
    derivative='direction' in plan
    validator=dict(kernels={'check_finite':module.get_function('check_finite')},finite_flag=cp.empty(1,dtype=cp.int32))
    if method.startswith('torch'):
        import torch
        from cuda_c2fuse import prepare_cuda, CudaExecutor
        tensor=torch.from_dlpack(x);prepared=prepare_cuda(plan,dtype=torch.float64,packed='_packed' in method)
        runner=CudaExecutor(prepared,tensor,capture=method.endswith('_captured'))
        def convert(values):return tuple(cp.from_dlpack(v) for v in (values if derivative else (values,)))
        def owned():
            runner.input.copy_(tensor);values=runner.replay_prevalidated()
            return convert(tuple(v.clone() for v in values) if derivative else values.clone())
        def checked():return convert(runner(tensor))
        return owned,checked
    if method.startswith('fused'):
        wide=q>4
        prepared=prepare_acc(compiled,compensated=method=='fused_compensated',derivative_compensated=derivative,
            projected=wide,coefficient_correction=not wide,final_compensated=wide)
        def raw():
            result=evaluate_acc(prepared,x,checked=False)
            return result if derivative else (result,)
        def checked():
            result=evaluate_acc(prepared,x,checked=True)
            return result if derivative else (result,)
        def owned(): return tuple(v.copy() for v in raw())
    else:
        a=plan['a'];matrix=csr_matrix((cp.asarray(a.data),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.indptr,dtype=cp.int32)),shape=a.shape)
        cc=[cp.asarray(v.T) for v in plan['float_coefficients']]
        packed=cp.concatenate(cc,axis=1) if '_packed' in method else None
        identity0=np.array_equal(plan['float_coefficients'][0],np.eye(q))
        direction=cp.asarray(plan['direction'].diagonal())[:,None] if derivative else None
        state={}
        def raw():
            z=x/scale;state['z']=z;terms=(z@packed).reshape(n,len(cc),q) if packed is not None else None
            def term(k):return terms[:,k,:] if terms is not None else z if k==0 and identity0 else z@cc[k]
            y=term(len(cc)-1);dy=cp.zeros_like(y) if derivative else None
            for k in range(len(cc)-2,-1,-1):
                if derivative:dy=matrix@dy+direction*y
                y=matrix@y+term(k)
            return (y*scale,dy*scale) if derivative else (y*scale,)
        owned=raw
        def checked():
            assert x.dtype==cp.float64 and x.flags.c_contiguous and x.shape==(n,q)
            if not _finite(validator,x):raise ValueError('Nonfinite input')
            values=owned()
            if not _finite(validator,state['z']) or any(not _finite(validator,v) for v in values):raise ArithmeticError('Nonfinite output')
            return values
    if method.endswith('_captured'):
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            for _ in range(3):static=raw()
        stream.synchronize()
        with stream:
            stream.begin_capture();static=raw();capture=stream.end_capture()
        def owned():
            capture.launch();return tuple(v.copy() for v in static)
    return owned,checked


def samples(fn, count):
    import cupy as cp
    out=[]
    for _ in range(count):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter_ns();values=fn()
        cp.cuda.get_current_stream().synchronize();out.append((time.perf_counter_ns()-start)/1e6)
        del values
    return out


def clean():
    import cupy as cp
    gc.collect();cp.get_default_memory_pool().free_all_blocks()


def ablations(seed, report, save):
    import cupy as cp
    rng=np.random.default_rng(seed)
    for spec in PROTOCOL['cases']:
        a=graph(spec['family'],spec['n']);q=spec['q'];derivative=spec.get('derivative',False)
        bs=coefficients(q,seed);x=np.random.default_rng(seed+spec['n']+q).normal(size=(spec['n'],q))
        budget=F(PROTOCOL['budget']);reference=cpu(compile_case(a,bs,'identity',F(0),derivative,True),x)
        row=dict(**spec,nnz=a.nnz,operator_sha256=hashlib.sha256(a.data.tobytes()+a.indices.tobytes()+a.indptr.tobytes()).hexdigest(),methods=[])
        report['cases'].append(row);save();plans={}
        for mode in ('identity','onepass','iterated'):
            start=time.perf_counter();compiled=compile_case(a,bs,mode,budget,derivative)
            plans[mode]=(compiled,1000*(time.perf_counter()-start))
        for method in PROTOCOL['methods']:
            mode='iterated' if 'iterated' in method else 'identity' if 'identity' in method else 'onepass'
            compiled,compile_ms=plans[mode];record=dict(method=method,degree=compiled['plan']['degree'],condition_penalty=float(compiled['condition_penalty']),
                tail_bound=float(compiled['original_error_bound']),compile_ms=compile_ms,blocks=[])
            row['methods'].append(record)
            try:
                xgpu=cp.asarray(x);start=time.perf_counter();fn,checked=executor(compiled,xgpu,method)
                cp.cuda.get_current_stream().synchronize();record['prepare_ms']=1000*(time.perf_counter()-start)
                values=checked();record['quality']=quality(tuple(v.get() for v in values),reference,x);del values
                record['accepted']=record['quality']['finite'] and max(record['quality']['frobenius'])<=float(budget)+PROTOCOL['rounding_screen_allowance']
                record['checked_call_ms']=samples(checked,3)
                for _ in range(PROTOCOL['warmups']):fn()
                del fn,checked,xgpu
            except Exception as error:record['error']=type(error).__name__+': '+str(error)
            clean();save()
        for block in range(PROTOCOL['blocks']):
            for index in rng.permutation(len(row['methods'])):
                record=row['methods'][index]
                if 'error' in record:continue
                mode='iterated' if 'iterated' in record['method'] else 'identity' if 'identity' in record['method'] else 'onepass'
                compiled=plans[mode][0];xgpu=cp.asarray(x);fn,checked=executor(compiled,xgpu,record['method'])
                for _ in range(PROTOCOL['warmups']):fn()
                record['blocks'].append(samples(fn,PROTOCOL['samples_per_block']))
                del fn,checked,xgpu;clean();save()
        print('CASE',spec,[(r['method'],r['degree'],round(float(np.median(r['blocks'])),3)) for r in row['methods'] if r['blocks']],flush=True)


def movielens(cache):
    cache.mkdir(parents=True,exist_ok=True);path=cache/'ml-1m.zip'
    if not path.exists():
        import ssl, certifi
        with urlopen(PROTOCOL['application']['url'],timeout=90,context=ssl.create_default_context(cafile=certifi.where())) as response:path.write_bytes(response.read())
    assert digest(path)==PROTOCOL['application']['archive_sha256'], 'MovieLens archive hash mismatch'
    with zipfile.ZipFile(path) as archive:text=archive.read('ml-1m/ratings.dat').decode('ascii')
    data=np.fromstring(text.replace('::',' '),sep=' ',dtype=np.int64).reshape(-1,4)
    assert len(data)==1000209
    data=data[np.lexsort((data[:,3],data[:,0]))];u=int(data[:,0].max());items=int(data[:,1].max())
    train=np.ones(len(data),dtype=bool);test={};seen={}
    starts=np.r_[0,np.flatnonzero(np.diff(data[:,0]))+1,len(data)]
    for lo,hi in zip(starts[:-1],starts[1:]):
        cut=hi-max(1,(hi-lo)//5);train[cut:hi]=False;uid=int(data[lo,0]-1)
        seen[uid]=data[lo:cut,1]-1;test[uid]=set((data[cut:hi,1][data[cut:hi,2]>=4]-1).tolist())-set(seen[uid].tolist())
    pos=data[train&(data[:,2]>=4)];rows=pos[:,0]-1;cols=u+pos[:,1]-1
    b=csr_array((np.ones(2*len(pos)),(np.r_[rows,cols],np.r_[cols,rows])),shape=(u+items,u+items));b.sum_duplicates()
    degrees=np.asarray(b.sum(axis=0)).ravel();a=csr_array(b@diags(1/np.maximum(degrees,1)))
    eligible=np.array([uid for uid in range(u) if test[uid]],dtype=int)
    users=np.random.default_rng(PROTOCOL['graph_seed']).permutation(eligible)[:PROTOCOL['application']['users']]
    return a,u,items,users,seen,test,digest(path)


def ranking(scores, users, offset, seen, test):
    recalls=[];ndcgs=[];lists=[];k=PROTOCOL['application']['k']
    for j,uid in enumerate(users):
        values=scores[offset:,j].copy();values[seen[int(uid)]]=-np.inf
        # Stable item-id tie breaking; identical zero/tied scores must give identical lists.
        top=np.lexsort((np.arange(len(values)),-values))[:k];truth=test[int(uid)]
        hits=np.array([int(v) in truth for v in top],dtype=float)
        recalls.append(float(hits.sum()/len(truth)));discount=1/np.log2(np.arange(2,len(top)+2))
        ndcgs.append(float(hits@discount/discount[:min(k,len(truth))].sum()));lists.append(top)
    return recalls,ndcgs,np.asarray(lists)


def application(seed, report, save, cache):
    import cupy as cp
    start=time.perf_counter();a,offset,items,users,seen,test,data_hash=movielens(cache);load_ms=1000*(time.perf_counter()-start)
    q=PROTOCOL['application']['channels'];alpha=F(PROTOCOL['application']['alpha'])
    bs=[np.eye(q,dtype=object)*(1-alpha)*alpha**k for k in range(PROTOCOL['application']['degree']+1)]
    batches=[users[i:i+q] for i in range(0,len(users),q)]
    x=np.zeros((a.shape[0],q));reference_lists=[];rr=[];nn=[]
    reference_plan=compile_case(a,bs,'identity',F(0),full=True)
    for batch in batches:
        x.fill(0);x[batch,np.arange(len(batch))]=1
        values=cpu(reference_plan,x)[0];r,n,lists=ranking(values,batch,offset,seen,test)
        rr.extend(r);nn.extend(n);reference_lists.extend(lists)
    result=dict(dataset_sha256=data_hash,ratings=1000209,vertices=a.shape[0],nnz=a.nnz,evaluated_users=len(users),
        load_build_ms=load_ms,cpu_reference_recall=float(np.mean(rr)),cpu_reference_ndcg=float(np.mean(nn)),methods=[])
    report['application']=result;save()
    for method in PROTOCOL['application']['methods']:
        mode='identity' if 'identity' in method else 'onepass';start=time.perf_counter()
        compiled=compile_case(a,bs,mode,F(0),full=True);compile_ms=1000*(time.perf_counter()-start)
        xgpu=cp.zeros_like(cp.asarray(x));start=time.perf_counter();fn,checked=executor(compiled,xgpu,method)
        cp.cuda.get_current_stream().synchronize();prepare_ms=1000*(time.perf_counter()-start)
        durations=[];metrics=None
        for repeat in range(PROTOCOL['application']['repeats']):
            rec=[];ndcg=[];lists=[];start=time.perf_counter()
            for batch in batches:
                x.fill(0);x[batch,np.arange(len(batch))]=1;xgpu.set(x)
                scores=checked()[0].get();r,n,top=ranking(scores,batch,offset,seen,test)
                rec.extend(r);ndcg.extend(n);lists.extend(top)
            durations.append(1000*(time.perf_counter()-start));metrics=(rec,ndcg,lists)
        rec,ndcg,lists=metrics
        result['methods'].append(dict(method=method,degree=compiled['plan']['degree'],condition_penalty=float(compiled['condition_penalty']),
            compile_ms=compile_ms,prepare_ms=prepare_ms,request_batch_ms=durations,recall_at_10=float(np.mean(rec)),ndcg_at_10=float(np.mean(ndcg)),
            top10_exact_agreement=float(np.mean(np.all(np.asarray(lists)==reference_lists,axis=1)))))
        print('APPLICATION',method,round(float(np.median(durations)),2),np.mean(rec),np.mean(ndcg),flush=True)
        del fn,checked,xgpu;clean();save()


def selftest():
    for family in ('alternating','balanced','grid','random','rmat','similarity_random'):
        a=graph(family,64);bs=coefficients(2,91);x=np.random.default_rng(92).normal(size=(64,2))
        expected=cpu(compile_case(a,bs,'identity',F(0),full=True),x)
        for mode in ('identity','onepass','iterated'):
            c=compile_case(a,bs,mode,F(1,10**12));actual=cpu(c,x)
            assert max(quality(actual,expected,x)['frobenius'])<1.1e-12
            assert c['original_error_bound']<=F(1,10**12)
    scores=np.array([[0.,0.],[3.,3.],[2.,2.],[1.,1.]])
    r,n,top=ranking(scores,np.array([0,1]),1,{0:np.array([1]),1:np.array([1])},{0:{0},1:{0}})
    assert r==[1.,1.] and n==[1.,1.] and np.array_equal(top[0],top[1])
    print('PASS: six graph families, original-coordinate bounds and held-out ranking metric checks.')


def small_accuracy(seed, report, save, smoke=False):
    import cupy as cp
    import mpmath as mp
    from check_high_precision import oracle, metrics
    report['small_accuracy']=[]
    for family in (('alternating',) if smoke else ('alternating','balanced','grid','random','rmat','similarity_random')):
        a=graph(family,64);bs=coefficients(4,seed);x=np.random.default_rng(seed).normal(size=(64,4))
        low=oracle(a,bs,x,False,F(0),256);truth=oracle(a,bs,x,False,F(0),384)
        discrepancy=max(abs(v-w) for yy,zz in zip(low,truth) for y,z in zip(yy,zz) for v,w in zip(y,z))
        assert discrepancy<mp.mpf('1e-65')
        row=dict(family=family,rows=64,precision_bits=[256,384],oracle_discrepancy=float(discrepancy),methods=[])
        report['small_accuracy'].append(row)
        for method in PROTOCOL['methods']:
            mode='iterated' if 'iterated' in method else 'identity' if 'identity' in method else 'onepass'
            compiled=compile_case(a,bs,mode,F(PROTOCOL['budget']));xgpu=cp.asarray(x)
            fn,checked=executor(compiled,xgpu,method);values=tuple(v.get() for v in checked())
            observed=metrics(values,truth,x)
            assert max(observed['error_per_input_norm'])<=float(F(PROTOCOL['budget']))+PROTOCOL['rounding_screen_allowance']
            np.testing.assert_allclose(fn()[0].get(),values[0],rtol=1e-12,atol=1e-12)
            row['methods'].append(dict(method=method,degree=compiled['plan']['degree'],quality=observed))
            del fn,checked,xgpu;clean()
        save()
    print('PASS: full small-fixture GPU outputs against independent 256/384-bit references.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--gpu-smoke',action='store_true')
    parser.add_argument('--seed',type=int,default=PROTOCOL['seeds'][0]);parser.add_argument('--output',type=Path);parser.add_argument('--cache',type=Path)
    args=parser.parse_args()
    if args.selftest:selftest();return
    if args.freeze:freeze();return
    if args.gpu_smoke:
        import torch
        torch.cuda.init();small_accuracy(71,{},lambda:None,smoke=True);return
    lock=check_freeze();assert args.seed in PROTOCOL['seeds'];assert args.output is not None and args.cache is not None
    import torch
    torch.set_num_threads(1);torch.cuda.init()
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    import cupy as cp
    import scipy, mpmath
    report=dict(study_id=PROTOCOL['study_id'],freeze_sha256=lock,source_sha256=sources(),seed=args.seed,
        gpu=torch.cuda.get_device_name(0),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cupy=cp.__version__,
        scipy=scipy.__version__,mpmath=mpmath.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
        status='running',cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');temporary.replace(args.output)
    save()
    try:
        small_accuracy(args.seed,report,save);ablations(args.seed,report,save);application(args.seed,report,save,args.cache);report['status']='complete'
    except Exception as error:report['status']='failed';report['error']=type(error).__name__+': '+str(error);raise
    finally:save()


if __name__=='__main__':main()
