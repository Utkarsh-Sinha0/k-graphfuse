"""Offline evidence audit; rerun the Kaggle script to reproduce performance."""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np

base=Path(__file__).resolve().parents[1] / "evidence"
manifest=json.loads((base/'MANIFEST.json').read_text())
for name,entry in manifest.items():
    p=base/name
    assert p.stat().st_size==entry['bytes'],name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
raw=json.loads((base/'SINHA_GRAPHFUSE_PUBLIC_REFERENCE_RAW.json').read_text())
assert raw['status']=='complete' and not raw['failures']
assert raw['unit_tests']=={'run':27,'failures':0,'errors':0,'skipped':0}
assert len(raw['cases'])==11 and all(c['accepted'] for c in raw['cases'])
evidence=base/'sinha_public_reference_evidence'
assert json.loads((evidence/'industry_reference.json').read_text())==raw
source=json.loads((evidence/'source-manifest.json').read_text())
assert source==raw['source_hashes']
for name,value in source.items(): assert hashlib.sha256((evidence/'source'/name).read_bytes()).hexdigest()==value
for c in raw['cases']:
    assert c['quality']['max_relative_per_vertex_error']<=1e-4
    for key in ('resident','host_round_trip'):
        if key not in c:continue
        t=c[key];a=np.asarray(t['samples_ms'])
        assert len(a)==t['sample_count'] and np.isfinite(a).all() and (a>0).all()
        assert float(np.median(a))==t['median_ms'] and float(np.quantile(a,.95))==t['p95_ms']
own=min((c for c in raw['cases'] if c['method'].startswith('Sinha')),key=lambda c:c['resident']['median_ms'])
for row in csv.DictReader((base/'SINHA_GRAPHFUSE_PUBLIC_REFERENCE_COMPARISONS.csv').open(encoding='utf-8-sig')):
    c=next(c for c in raw['cases'] if c['method']==row['comparator'])
    a,b=np.asarray(c['resident']['samples_ms']),np.asarray(own['resident']['samples_ms']);rng=np.random.default_rng(20261009)
    z=np.median(a[rng.integers(len(a),size=(5000,len(a)))],axis=1)/np.median(b[rng.integers(len(b),size=(5000,len(b)))],axis=1)
    ci=np.quantile(z,[.025,.975])
    assert math.isclose(float(row['comparator_to_sinha_ratio']),np.median(a)/np.median(b),rel_tol=1e-12)
    assert math.isclose(float(row['speedup_95ci_low']),ci[0],rel_tol=1e-12)
    assert math.isclose(float(row['speedup_95ci_high']),ci[1],rel_tol=1e-12)
    passed=own['resident']['median_ms']<=.9*c['resident']['median_ms'] and ci[0]>1 and own['resident']['p95_ms']<=c['resident']['p95_ms']
    assert row['passes_scoped_10pct_bar']==str(bool(passed))

import argparse
parser = argparse.ArgumentParser(description="Audit saved public-reference scores against separately acquired official data")
parser.add_argument("--reference", type=Path)
args = parser.parse_args()
if args.reference is None:
    print("PASS: public-reference sources, accepted configurations, statistics and bootstrap comparisons. Optional third-party reference output check not requested.")
else:
    reference = args.reference

    assert hashlib.sha256(reference.read_bytes()).hexdigest()==raw['dataset']['reference_file_sha256']
    ids=np.loadtxt(reference,dtype=np.int64,usecols=0);expected=np.loadtxt(reference,usecols=1)
    order=np.argsort(ids);ids,expected=ids[order],expected[order]
    scores=np.fromfile(evidence/'arcade_scores.bin',dtype=np.dtype([('id','>i8'),('score','>f8')]))
    order=np.argsort(scores['id']);assert np.array_equal(scores['id'][order],ids)
    error=float(np.max(np.abs(scores['score'][order]-expected)/np.abs(expected)))
    c=next(c for c in raw['cases'] if c['method'].startswith('ArcadeDB'))
    assert error<=1e-4 and abs(error-c['quality']['max_relative_per_vertex_error'])<1e-15
    assert json.loads((evidence/'arcade_timing.json').read_text())['samples_ms']==c['resident']['samples_ms']
    print('PASS: all file hashes, source hashes, 11 accepted configurations, raw statistics, bootstrap comparisons and released CPU reference output.')
