"""Recompute theorem-constant attribution from unchanged published measurements."""
from pathlib import Path
import argparse, ast, base64, hashlib, json, math, re, statistics
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def extract():
    rows=[];sources={};drivers=[];degree_cases=[]
    for family in ('matrix-filters','sensitivity'):
        folder=ROOT/f'evidence/2026-10-10/detailed/raw/{family}'
        data=read(folder/'results.json');manifest=read(folder/'source-manifest.json')
        tree=ast.parse((folder/'run.py').read_text(encoding='utf-8'))
        assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in n.targets))
        files=json.loads(ast.literal_eval(assignment.value.args[0]))
        for name,expected in manifest.items():assert hashlib.sha256(base64.b64decode(files[name])).hexdigest()==expected,name
        for name,expected in data['source_hashes'].items():assert manifest[name]==expected
        driver=base64.b64decode(files['sinha_industry_bench.py']).decode('utf-8')
        assert 'constant=1+sqrt_upper(2)' in driver and 'for mode in ("full", "sinha", "prior")' in driver
        assert 'rng.shuffle(specifications)' in driver and 'torch_factory(p, x, dtype, packed, capture)' in driver
        assert 'constant=Fraction(2)' in base64.b64decode(files['sparse_c2fuse.py']).decode('utf-8')
        drivers.append(manifest['sinha_industry_bench.py'])
        assert 'Tesla T4' in json.dumps(data['environment'])
        for p in (folder/'results.json',folder/'source-manifest.json',folder/'run.py'):
            sources[p.relative_to(ROOT).as_posix()]=digest(p)
        keys=[(r['case'],r['budget'],r['operator_sha256'],r['input_sha256']) for r in data['cases']]
        assert len(keys)==len(set(keys))
        for case in data['cases']:
            degree_cases.append(dict(case=case['case'],vertices=case['vertices'],channels=case['features'],budget=case['budget'],
                old_degree=case['older_complete_constant_degree'],new_degree=case['degree'],
                source=f'evidence/2026-10-10/detailed/raw/{family}/results.json',operator_sha256=case['operator_sha256'],input_sha256=case['input_sha256']))
            if case['degree']==case['older_complete_constant_degree']:continue
            assert case['status']=='complete' and case['degree']+1==case['older_complete_constant_degree']
            methods={}
            for m in case['candidates']:
                match=re.fullmatch(r'(Sinha GraphFuse|Optimized older complete-bound control) float64 packed=(True|False) capture=(True|False)',m['method'])
                if match and m.get('accepted'):
                    assert m['budget']==case['budget'] and m['quality']['finite']
                    assert all(math.isfinite(v) and 0<=v<=case['budget'] for v in m['quality']['error_per_input_frobenius_norm'])
                    if case['kind']=='sensitivity':
                        assert m['additional_quality']['accepted']
                        assert all(math.isfinite(v) and 0<=v<=case['budget'] for v in m['additional_quality']['endpoint_errors'])
                    samples=m['resident']['samples_ms']
                    assert len(samples)==m['resident']['sample_count']==31 and all(math.isfinite(v) and v>0 for v in samples)
                    assert math.isclose(statistics.median(samples),m['resident']['median_ms'],rel_tol=1e-12)
                    assert math.isclose(float(np.quantile(samples,.95)),m['resident']['p95_ms'],rel_tol=1e-12)
                    methods[match.groups()]=m
            for packed in ('False','True'):
                for capture in ('False','True'):
                    old=methods[('Optimized older complete-bound control',packed,capture)]
                    new=methods[('Sinha GraphFuse',packed,capture)]
                    a=old['resident']['median_ms'];b=new['resident']['median_ms']
                    rows.append(dict(case=case['case'],vertices=case['vertices'],channels=case['features'],budget=case['budget'],
                        old_degree=case['older_complete_constant_degree'],new_degree=case['degree'],packed=packed=='True',capture=capture=='True',
                        old_median_ms=a,new_median_ms=b,time_reduction_pct=100*(1-b/a),
                        old_p95_ms=old['resident']['p95_ms'],new_p95_ms=new['resident']['p95_ms'],
                        old_error=old['quality']['error_per_input_frobenius_norm'],new_error=new['quality']['error_per_input_frobenius_norm'],
                        source=f'evidence/2026-10-10/detailed/raw/{family}/results.json',
                        operator_sha256=case['operator_sha256'],input_sha256=case['input_sha256']))
    assert len(set(drivers))==1 and len(rows)==8 and len(degree_cases)==11
    assert sum(r['new_degree']<r['old_degree'] for r in degree_cases)==2
    assert all(r['new_p95_ms']<r['old_p95_ms'] and r['time_reduction_pct']>0 for r in rows)
    local_path=ROOT/'experiments/precision_gpu/LOCAL_RESULTS.json';local=read(local_path)
    assert local['gpu']=='NVIDIA GeForce RTX 3060 Laptop GPU' and local['status']=='complete'
    local_rows=[{k:r[k] for k in ('n','q','sensitivity','degrees')} for r in local['cases']]
    assert len(local_rows)==6 and all(r['degrees']['released']==r['degrees']['older_bound'] for r in local_rows)
    cpu_path=HERE/'cpu_sparse_original.json';boundary=read(cpu_path)['constant_only_boundary']
    assert (boundary['older_complete_bound_degree'],boundary['new_degree'])==(9,8)
    assert not any('ms' in key or 'second' in key for key in boundary)
    for p in (local_path,cpu_path):sources[p.relative_to(ROOT).as_posix()]=digest(p)
    reduction=100*(1-2/(1+math.sqrt(2)))
    assert math.isclose(reduction,17.157287525380983,rel_tol=1e-14)
    return dict(analysis_date='2026-10-11',measurement_date='2026-10-10',
        source_sha256=sources,driver_sha256=drivers[0],matched_fp64_pairs=rows,
        examined_workload_records=11,degree_reducing_records=2,all_degree_cases=degree_cases,
        constant_value_tail_factor_reduction_pct=reduction,
        median_time_reduction_range_pct=[min(r['time_reduction_pct'] for r in rows),max(r['time_reduction_pct'] for r in rows)],
        laptop_gpu=local['gpu'],laptop_constant_cases=local_rows,cpu_degree_boundary=boundary,
        exact_degree_9_to_8_latency_found=False,
        scope='Two constructed T4 workloads; four matched packing/capture settings each; 31 consecutive synchronized samples/configuration, shuffled configuration order. Observed comparisons below the 10% strong-speed bar; no paired multi-block interval or independent repeat. Truncation certificates exclude execution rounding. This analysis creates no new GPU timing.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    summary=extract();target=HERE/'SUMMARY.json'
    if args.verify:
        assert read(target)==summary
        for name,expected in read(HERE/'MANIFEST.json')['sha256'].items():assert digest(ROOT/name)==expected,name
        claims={c['id']:c for c in read(ROOT/'docs/claims.json')['claims']}
        assert claims['bound_attribution_time_range_20261011']['value']==summary['median_time_reduction_range_pct']
        assert claims['bound_attribution_matched_pairs_20261011']['value']==len(summary['matched_fp64_pairs'])==8
        assert claims['bound_attribution_factor_reduction_20261011']['value']==summary['constant_value_tail_factor_reduction_pct']
        assert claims['bound_attribution_degree_reductions_20261011']['value']==summary['degree_reducing_records']==2
        assert claims['bound_attribution_degree_reductions_20261011']['denominator']==summary['examined_workload_records']==11
        print('PASS: theorem-constant sources, eight accepted matching FP64 pairs, all raw medians/p95, laptop equal-degree cases and untimed CPU boundary.')
    else:
        target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
        paths=[Path(__file__),target,HERE/'cpu_sparse_original.json']
        record={'sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}
        (HERE/'MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
        print('Recomputed all eight matched profiles and laptop/boundary limitations.')
