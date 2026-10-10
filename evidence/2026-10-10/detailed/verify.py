"""Offline consistency audit. Does not rerun GPU calculations or attest execution."""
from pathlib import Path
import ast
import base64
import csv
import hashlib
import json
import math
import numpy as np

root=Path(__file__).resolve().parent
summary=json.loads((root/'verification_summary.json').read_text())
confirmed={};count=0

def stats(value):
    samples=np.asarray(value['samples_ms'],dtype=float)
    assert len(samples)==value['sample_count'] and np.isfinite(samples).all() and (samples>0).all()
    assert math.isclose(float(np.median(samples)),value['median_ms'],rel_tol=1e-11)
    assert math.isclose(float(np.quantile(samples,.95)),value['p95_ms'],rel_tol=1e-11)

for check in summary['verified_suites']:
    path=root/check['source_path'];assert path.resolve().is_relative_to(root.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==check['sha256']
    folder=path.parent;data=json.loads(path.read_text())
    assert json.loads((folder/'unit-tests.json').read_text())==dict(run=29,failures=0,errors=0,skipped=0)
    profile=json.loads((folder/'profile.json').read_text());assert profile==data['protocol']['configuration']
    manifest=json.loads((folder/'source-manifest.json').read_text())
    assert hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()==check['runner_sha256']
    tree=ast.parse((folder/'run.py').read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in n.targets))
    files=json.loads(ast.literal_eval(assignment.value.args[0]))
    for name,digest in manifest.items():assert hashlib.sha256(base64.b64decode(files[name])).hexdigest()==digest
    assert json.loads(base64.b64decode(files['profile.json']))==profile
    for name,digest in data['source_hashes'].items():assert manifest[name]==digest
    for case in data['cases']:
        count+=1
        for candidate in case.get('candidates',[]):
            for key in ('pilot','resident','host_output','host_round_trip'):
                if key in candidate:stats(candidate[key])
        for cold in case.get('cold_jobs',[]):stats(cold)
        if 'native_resident' in case:stats(case['native_resident'])
        for result in case.get('confirmation',[]):
            for block in result['blocks']:
                stats(block)
                if result.get('accepted'):
                    quality=block['quality'];assert quality['accepted']
                    assert 0<=quality['l1_error']<=case['accuracy_l1_budget']
                    assert 0<=quality['max_relative_error']<=case['relative_budget']
            if 'resident' in result:
                stats(result['resident'])
                assert result['resident']['samples_ms']==[v for b in result['blocks'] for v in b['samples_ms']]
            if 'host_output' in result:stats(result['host_output'])
            if result.get('accepted') and result.get('resident',{}).get('sample_count')==155:
                assert len(result['blocks'])==5 and all(b['sample_count']==31 for b in result['blocks'])
                confirmed[(data['shard'],case['case'],result['group'])]=result

rows=list(csv.DictReader((root/'confirmed_comparisons.csv').open()))
for row in rows:
    own=confirmed[(row['suite'],row['case'],'K_GPU')];other=confirmed[(row['suite'],row['case'],row['comparator_group'])]
    assert row['K_method']==own['method'] and row['comparator']==other['method']
    assert json.loads(row['K_configuration'])==own['configuration']
    assert json.loads(row['comparator_configuration'])==other['configuration']
    controls=[v for (suite,name,group),v in confirmed.items() if suite==row['suite'] and name==row['case'] and group in ('NATIVE_GPU','CUPY_GPU','SCIPY_CPU')]
    best=min(controls,key=lambda v:v['resident']['median_ms'])
    gpu=[v for v in controls if v['group'].endswith('_GPU')]
    best_gpu=min(gpu,key=lambda v:v['resident']['median_ms']) if gpu else None
    assert row['strongest_tested_control']==str(other is best)
    assert row['strongest_tested_gpu_control']==str(other is best_gpu)
    assert math.isclose(float(row['K_median_ms']),own['resident']['median_ms'],rel_tol=1e-11)
    assert math.isclose(float(row['comparator_median_ms']),other['resident']['median_ms'],rel_tol=1e-11)
    ratio=other['resident']['median_ms']/own['resident']['median_ms']
    assert math.isclose(float(row['speedup']),ratio,rel_tol=1e-11)
    rng=np.random.default_rng(20261010);selection=rng.integers(5,size=(5000,5))
    a=np.asarray([b['samples_ms'] for b in own['blocks']]);b=np.asarray([b['samples_ms'] for b in other['blocks']])
    lo,hi=np.quantile(np.median(b[selection].reshape(5000,-1),axis=1)/np.median(a[selection].reshape(5000,-1),axis=1),[.025,.975])
    assert math.isclose(float(row['paired_block_95ci_low']),lo,rel_tol=1e-11)
    assert math.isclose(float(row['paired_block_95ci_high']),hi,rel_tol=1e-11)
    passes=ratio>=1/.9 and lo>1 and own['resident']['p95_ms']<=other['resident']['p95_ms']
    assert row['passes_scoped_10pct_bar']==str(bool(passes)) and row['global_sota']=='False'
assert count==summary['case_records'] and len(rows)==summary['confirmed_comparisons']
print(f'PASS: {len(summary["verified_suites"])} source snapshots, {count} case records, {len(rows)} confirmed comparisons')
