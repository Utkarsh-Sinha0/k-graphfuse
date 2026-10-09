"""Exact public Graphalytics task, original probability outputs, real released libraries."""
from pathlib import Path
from fractions import Fraction as F
from urllib.request import Request, urlopen
from time import perf_counter
from datetime import datetime, timezone
import gc, hashlib, importlib.metadata, inspect, json, os, platform, subprocess, tarfile, traceback, unittest
import numpy as np
import scipy
from scipy.sparse import coo_array, diags
import torch
import cupy as cp
import pylibcugraph as plc
import zstandard
from sparse_c2fuse import compile_sparse, evaluate_sparse
from cuda_c2fuse import prepare_cuda, CudaExecutor
from sinha_industry_bench import Monitor, timing, cleanup

BASE = Path('/kaggle/working/sinha_public_reference')
RESULT = BASE / 'evidence'
REPORT = {'project':'Sinha GraphFuse', 'suite':'public_industry_reference', 'cases':[], 'failures':[],
          'started_utc':datetime.now(timezone.utc).isoformat(),
          'contract':'Original probability-score vectors; unweighted undirected global PageRank; uniform initialization; damping0.85; exactly10 iterations',
          'published_hardware_is_different':True, 'published_speedup_ratios_allowed':False}

def save():
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT/'industry_reference.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False))

def download(url, path):
    start=perf_counter()
    with urlopen(Request(url,headers={'User-Agent':'Sinha-GraphFuse-Benchmark'}),timeout=180) as incoming, path.open('wb') as target:
        while chunk:=incoming.read(1<<20): target.write(chunk)
    return {'url':url,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seconds':perf_counter()-start}

def extract(path, destination, zstd=False):
    destination.mkdir(parents=True,exist_ok=True)
    with path.open('rb') as handle:
        if zstd:
            with zstandard.ZstdDecompressor().stream_reader(handle) as reader, tarfile.open(fileobj=reader,mode='r|') as archive:
                archive.extractall(destination,filter='data')
        else:
            with tarfile.open(fileobj=handle,mode='r:gz') as archive: archive.extractall(destination,filter='data')

def quality(ids, values, vertices, reference):
    order=np.argsort(ids)
    assert np.array_equal(ids[order],vertices)
    value=np.asarray(values)[order]
    relative=np.abs(value-reference)/np.maximum(np.abs(reference),np.finfo(float).tiny)
    return {'max_relative_per_vertex_error':float(relative.max()),'l1_error':float(np.abs(value-reference).sum()),
            'mass':float(value.sum()),'accepted':bool(np.isfinite(value).all() and relative.max()<=1e-4)}

def to_host(value):
    return tuple(x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else cp.asnumpy(x) for x in value)

def gpu_measure(name,factory,vertices,reference):
    cleanup(); row={'method':name,'gpu':True}
    try:
        with Monitor() as monitor:
            start=perf_counter(); fn,host=factory(); torch.cuda.synchronize()
            row['prepare_seconds']=perf_counter()-start
            start=perf_counter(); value=fn(); torch.cuda.synchronize()
            row['first_call_ms']=1000*(perf_counter()-start)
            row['quality']=quality(*to_host(value),vertices,reference); del value
            row['accepted']=row['quality']['accepted']
            row['resident']=timing(fn,31,True)
            row['host_round_trip']=timing(host,11,True)
        row['resources']=monitor.result()
    except Exception as error:
        row.update(accepted=False,error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    finally:
        if 'fn' in locals(): del fn,host
        cleanup()
    REPORT['cases'].append(row); save()
    print('CANDIDATE',name,row.get('accepted'),row.get('resident',{}).get('median_ms'),flush=True)

def main():
    BASE.mkdir(exist_ok=True); save()
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    REPORT['environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'torch':torch.__version__,'cupy':cp.__version__,'pylibcugraph':importlib.metadata.version('pylibcugraph-cu12'),
        'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'cpu_count':os.cpu_count(),
        'cpu_model':sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')}),
        'nvidia_smi':subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout}
    REPORT['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.java','.class')}
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py'))
    REPORT['unit_tests']={'run':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors),'skipped':len(tests.skipped)}
    assert tests.wasSuccessful() and not tests.skipped; save()
    print('DOWNLOADING PUBLIC DATASET',flush=True)
    package=BASE/'datagen.tar.zst'
    REPORT['dataset_archive']=download('https://datasets.ldbcouncil.org/graphalytics/datagen-7_5-fb.tar.zst',package)
    data=BASE/'dataset'; extract(package,data,zstd=True)
    vfile=next(data.rglob('datagen-7_5-fb.v')); data=vfile.parent
    efile=data/'datagen-7_5-fb.e'; rfile=data/'datagen-7_5-fb-PR'
    properties=(data/'datagen-7_5-fb.properties').read_text()
    REPORT['dataset_properties']=properties; save()
    vertices=np.sort(np.loadtxt(vfile,dtype=np.int64))
    pairs=np.loadtxt(efile,dtype=np.int64,usecols=(0,1))
    assert len(vertices)==633432 and len(pairs)==34185747
    src,dst=np.searchsorted(vertices,pairs[:,0]),np.searchsorted(vertices,pairs[:,1])
    assert np.array_equal(vertices[src],pairs[:,0]) and np.array_equal(vertices[dst],pairs[:,1])
    adjacency=coo_array((np.ones(len(src)),(src,dst)),shape=(len(vertices),len(vertices))).tocsr()
    adjacency=adjacency+adjacency.T; adjacency.sum_duplicates(); adjacency.sort_indices()
    del pairs,src,dst; gc.collect()
    degree=np.asarray(adjacency.sum(axis=1)).ravel()
    assert np.all(degree>0),'Dangling correction is outside this finite-polynomial case; do not silently change the graph'
    reference_ids=np.loadtxt(rfile,dtype=np.int64,usecols=0)
    reference=np.loadtxt(rfile,usecols=1)
    order=np.argsort(reference_ids); reference=reference[order]
    assert np.array_equal(reference_ids[order],vertices)
    del reference_ids,order
    rank=np.full(len(vertices),1/len(vertices))
    for _ in range(10): rank=.15/len(vertices)+.85*(adjacency@(rank/degree))
    REPORT['independent_reference_quality']=quality(vertices,rank,vertices,reference)
    assert REPORT['independent_reference_quality']['accepted'],REPORT['independent_reference_quality']
    sqrt_degree=np.sqrt(degree)
    s=diags(1/sqrt_degree)@adjacency@diags(1/sqrt_degree)
    coefficients=[np.array([[F(3,20)*F(17,20)**k]],dtype=object) for k in range(10)]+[np.array([[F(17,20)**10]],dtype=object)]
    plan=compile_sparse(s.tocsr(),[coefficients],0,bounds='fast')
    x=(1/len(vertices)/sqrt_degree).reshape(-1,1)
    check=evaluate_sparse(plan,x).ravel()*sqrt_degree
    np.testing.assert_allclose(check,rank,rtol=1e-12,atol=1e-18)
    REPORT['dataset']={'name':'datagen-7_5-fb','vertices':len(vertices),'file_edges':34185747,'stored_symmetric_entries':int(adjacency.nnz),
        'vertex_file_sha256':hashlib.sha256(vfile.read_bytes()).hexdigest(),'edge_file_sha256':hashlib.sha256(efile.read_bytes()).hexdigest(),
        'reference_file_sha256':hashlib.sha256(rfile.read_bytes()).hexdigest(),'degree':plan['degree']}
    save(); print('REFERENCE AND POLYNOMIAL SELFCHECK PASSED',flush=True)
    for dtype in (torch.float32,torch.float64):
        for packed,capture in ((False,False),(True,False),(False,True),(True,True)):
            def factory(dtype=dtype,packed=packed,capture=capture):
                incoming=torch.tensor(x,device='cuda',dtype=dtype)
                roots=torch.tensor(sqrt_degree,device='cuda',dtype=dtype)
                ids=torch.tensor(vertices,device='cuda')
                executor=CudaExecutor(prepare_cuda(plan,dtype=dtype,packed=packed),incoming,capture=capture)
                def fn(): return ids,executor(incoming).ravel()*roots
                def host(): return to_host(fn())
                return fn,host
            gpu_measure(f'Sinha GraphFuse {dtype} packed={packed} capture={capture}',factory,vertices,reference)
    for dtype in (cp.float32,cp.float64):
        def factory(dtype=dtype):
            coo=adjacency.tocoo()
            handle=plc.ResourceHandle()
            id_type=cp.int32 if vertices.min()>=0 and vertices.max()<2**31 else cp.int64
            graph=plc.SGGraph(handle,plc.GraphProperties(is_symmetric=True,is_multigraph=False),
                cp.asarray(vertices[coo.row],dtype=id_type),cp.asarray(vertices[coo.col],dtype=id_type),weight_array=cp.asarray(coo.data,dtype=dtype),
                store_transposed=True,renumber=True,do_expensive_check=True)
            verts=cp.asarray(vertices,dtype=id_type); sums=cp.asarray(degree,dtype=dtype)
            def fn():
                answer=plc.pagerank(handle,graph,verts,sums,None,None,alpha=.85,epsilon=1e-30,max_iterations=10,
                    do_expensive_check=False,fail_on_nonconvergence=False)
                return answer[0],answer[1]
            def host(): return to_host(fn())
            return fn,host
        gpu_measure(f'NVIDIA pylibcugraph native probability PageRank {np.dtype(dtype).name}',factory,vertices,reference)
    del adjacency,s,plan,x,degree,sqrt_degree; cleanup(); gc.collect()
    try:
        print('DOWNLOADING RELEASED ARCADEDB',flush=True)
        archive=BASE/'arcadedb.tar.gz'
        url='https://github.com/ArcadeData/arcadedb/releases/download/26.10.1/arcadedb-26.10.1-base.tar.gz'
        REPORT['arcadedb_archive']=download(url,archive)
        with urlopen(Request(url+'.sha256',headers={'User-Agent':'Sinha-GraphFuse-Benchmark'}),timeout=60) as h: expected=h.read().decode().split()[0]
        assert REPORT['arcadedb_archive']['sha256']==expected
        distribution=BASE/'arcadedb'; extract(archive,distribution)
        java_candidates=list(distribution.rglob('bin/java'))
        if not java_candidates:
            jdk_archive=BASE/'temurin25.tar.gz'
            REPORT['jdk_archive']=download('https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.4.1%2B1/OpenJDK25U-jdk_x64_linux_hotspot_25.0.4.1_1.tar.gz',jdk_archive)
            assert REPORT['jdk_archive']['sha256']=='dbb698396d478e7fa2b1e50f4103324b2a99b90569ee27c33f2261f9215cf41e'
            runtime=BASE/'jdk'; extract(jdk_archive,runtime)
            java_candidates=list(runtime.rglob('bin/java'))
        assert java_candidates,'Verified Java runtime could not be located'
        java=str(java_candidates[0]); REPORT['java_version']=subprocess.run([java,'-version'],capture_output=True,text=True).stderr
        jars=list(distribution.rglob('arcadedb-engine-26.10.1.jar')); assert len(jars)==1,'Expected one released engine jar'
        REPORT['arcadedb_engine_jar_sha256']=hashlib.sha256(jars[0].read_bytes()).hexdigest()
        classpath=str(jars[0].parent/'*')
        classpath=classpath+os.pathsep+str(Path(__file__).parent)
        command=[java,'--add-modules','jdk.incubator.vector','-Xms6g','-Xmx12g','-cp',classpath,'SinhaArcadePR',str(data),str(BASE/'arcade_db'),str(RESULT/'arcade_scores.bin'),str(RESULT/'arcade_timing.json')]
        if Path('/usr/bin/time').exists(): command=['/usr/bin/time','-v','-o',str(RESULT/'arcade_resources.txt')]+command
        with (RESULT/'arcade_run.log').open('w') as log:
            outcome=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        assert outcome.returncode==0,'Released ArcadeDB wrapper failed; see arcade_run.log'
        measured=json.loads((RESULT/'arcade_timing.json').read_text())
        values=np.fromfile(RESULT/'arcade_scores.bin',dtype=np.dtype([('id','>i8'),('score','>f8')]))
        audit=quality(values['id'],values['score'],vertices,reference)
        times=measured['samples_ms']
        REPORT['cases'].append({'method':'ArcadeDB 26.10.1 released embedded PageRank','gpu':False,'accepted':audit['accepted'],'quality':audit,
            'prepare_seconds':measured['prepare_seconds'],'cpu_processors':measured['cpu_processors'],
            'resident':{'samples_ms':times,'sample_count':len(times),'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95))}})
        save(); print('ARCADE BENCHMARK SAVED',audit,flush=True)
    except Exception as error:
        REPORT['failures'].append({'phase':'ArcadeDB released comparator','error':str(error),'traceback':traceback.format_exc()}); save()
        print('ARCADE COMPARATOR FAILURE',str(error),flush=True)
    REPORT['status']='complete' if not REPORT['failures'] else 'partial'
    REPORT['finished_utc']=datetime.now(timezone.utc).isoformat(); save()
    print('PUBLIC INDUSTRY REFERENCE BENCHMARK FINISHED',REPORT['status'],flush=True)

if __name__=='__main__':
    try: main()
    except Exception as error:
        REPORT['failures'].append({'phase':'harness','error':str(error),'traceback':traceback.format_exc()}); REPORT['status']='partial'; save(); raise
