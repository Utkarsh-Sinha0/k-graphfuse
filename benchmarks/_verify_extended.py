"""Read-only extended evidence audit; no CUDA imports or timing executions."""
from pathlib import Path
import csv, hashlib, json, math, re, runpy, shutil, subprocess, sys, tempfile

repo=Path(__file__).resolve().parents[1]
detail=repo/'evidence/2026-10-10/detailed'
exp=repo/'experiments/precision_gpu'
runpy.run_path(str(detail/'verify.py'),run_name='__main__')
runpy.run_path(str(exp/'verify_package.py'),run_name='__main__')
def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as file:return list(csv.DictReader(file))
rows=read_csv(detail/'confirmed_comparisons.csv')
native=[r for r in rows if r['comparator_group']=='NATIVE_GPU']
coverage=read_csv(detail/'case_coverage.csv')
attempted=[r for r in coverage if 'NATIVE_GPU' in r['accepted_control_groups'].split(';')]
failures=[r for r in attempted if r['K_status']=='NO_ACCEPTED_K_CONFIRMATION']
def accuracy(r):
    return all(float(r['K_worst_block_'+k])<=float(r['comparator_worst_block_'+k]) for k in ('l1_error','relative_error'))
def speed(r):return r['passes_scoped_10pct_bar']=='True'
same=[r for r in native if json.loads(r['K_configuration'])['precision']==json.loads(r['comparator_configuration'])['precision']]
assert len(native)==len({(r['suite'],r['case']) for r in native})==85
assert len(attempted)==len(native)+len(failures)==89
counts=dict(attempted_cases=len(attempted),comparable_cases=len(native),
    lower_median=sum(float(r['speedup'])>1 for r in native),speed_gate_wins=sum(speed(r) for r in native),
    both_reference_errors_no_worse=sum(accuracy(r) for r in native),
    speed_and_both_reference_errors_wins=sum(speed(r) and accuracy(r) for r in native),
    K_accuracy_failures=len(failures),same_precision_cases=len(same),
    same_precision_speed_wins=sum(speed(r) for r in same),same_precision_speed_and_error_wins=sum(speed(r) and accuracy(r) for r in same))
summary=json.loads((detail.parent/'comparison_summary.json').read_text(encoding='utf-8'))
assert summary['native_cugraph']==counts
published=read_csv(detail.parent/'native_comparisons.csv')
assert len(published)==len(native)
for original,public in zip(native,published):
    assert all(public[k]==v for k,v in original.items())
    assert public['both_reference_errors_no_worse']==str(accuracy(original))
    assert public['speed_and_both_reference_errors_win']==str(speed(original) and accuracy(original))
assert json.loads((detail.parent/'K_accuracy_failures.json').read_text(encoding='utf-8'))==failures
for field,key in (('strongest_tested_control','strongest_tested_overall'),('strongest_tested_gpu_control','strongest_tested_gpu')):
    controls=[r for r in rows if r[field]=='True']
    assert summary[key]==dict(wins=sum(speed(r) for r in controls),cases=len(controls))
metrics=read_csv(repo/'evidence/measured_metrics.csv')
matched=[r for r in metrics if r['metric'] not in ('GPU frontend preparation','Sampled average board power')]
assert len(matched)==9 and sum(float(r['lower_value_pct'])>0 for r in matched)==8
for name,digest in summary['source_sha256'].items():
    path=(repo/name).resolve();assert path.is_relative_to(repo)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name

# Reuse the complete recorded precision calculation in an isolated temporary copy.
with tempfile.TemporaryDirectory(prefix='kgf-evidence-') as folder:
    temporary=Path(folder)/'precision';shutil.copytree(exp,temporary,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    subprocess.run([sys.executable,str(temporary/'audit_product.py')],check=True,stdout=subprocess.DEVNULL)
    recomputed=json.loads((temporary/'PRODUCT_VERIFIED.json').read_text(encoding='utf-8'))
expected=json.loads((exp/'PRODUCT_VERIFIED.json').read_text(encoding='utf-8'))
def equal(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-30)
    elif isinstance(a,str):assert a.replace('\\','/')==b.replace('\\','/')
    else:assert a==b
equal(recomputed,expected)
assert expected['t4_dual_gates']==len(expected['t4_cases'])==6
for r in expected['t4_cases']:
    assert r['new_ms']<=.9*r['old_ms'] and r['ci_low']>1 and r['new_p95_ms']<=r['old_p95_ms']
    assert all(n<=o for n,o in zip(r['new_sampled_errors'],r['old_sampled_errors']))
    assert all(n<=o for n,o in zip(r['new_sampled_max_absolute'],r['old_sampled_max_absolute']))

claims={c['id']:c for c in json.loads((repo/'docs/claims.json').read_text(encoding='utf-8'))['claims']}
for identity,value,denominator in (('native_speed_wins_20261010',59,89),('native_speed_error_wins_20261010',26,89),
    ('strongest_overall_wins_20261010',33,85),('strongest_gpu_wins_20261010',38,85),
    ('precision_dual_gates_20261010',6,6),('original_favorable_metrics',8,9)):
    assert claims[identity]['value']==value and claims[identity]['denominator']==denominator
equal(claims['precision_speedup_range_20261010']['value'],[min(r['speedup'] for r in expected['t4_cases']),max(r['speedup'] for r in expected['t4_cases'])])
for c in claims.values():assert (repo/c['source']).exists(),c['source']
print('PASS: 59/89 native speed wins, 26/89 speed/error wins, four failures, six precision gates and structured claims.')

# Check published conversions against absolute timings, independent of rendering libraries.
presentation=json.loads((repo/'docs/assets/presentation_metrics.json').read_text(encoding='utf-8'))
for name,digest in presentation['source_sha256'].items():
    path=(repo/name).resolve();assert path.is_relative_to(repo)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
for name in presentation['artifacts']:assert (repo/name).is_file(),name
assert set(presentation['artifact_sha256'])==set(presentation['artifacts'])
for name,digest in presentation['artifact_sha256'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
for key,value in counts.items():assert presentation['native_coverage'][key]==value,key
shown=presentation['precision_portfolio']['cases']
assert len(shown)==len(expected['t4_cases'])
for saved,displayed in zip(expected['t4_cases'],shown):
    assert (saved['n'],saved['q'],saved['old_ms'],saved['new_ms'])==(
        displayed['n'],displayed['q'],displayed['old_ms'],displayed['new_ms'])
    ratio=saved['old_ms']/saved['new_ms']
    equal(displayed['speedup_ratio'],ratio)
    equal(displayed['compute_time_reduction_percent'],100*(1-1/ratio))
    equal(displayed['derived_serial_compute_rate_increase_percent'],100*(ratio-1))
    intervals=[saved['ci_low'],saved['ci_high']]
    equal(displayed['ci_speedup_ratio'],intervals)
    equal(displayed['compute_time_reduction_ci_percent'],[100*(1-1/r) for r in intervals])
    equal(displayed['derived_serial_rate_increase_ci_percent'],[100*(r-1) for r in intervals])
for identity,field in (
    ('precision_time_reduction_range_20261010','compute_time_reduction_percent'),
    ('precision_derived_rate_range_20261010','derived_serial_compute_rate_increase_percent')):
    equal(claims[identity]['value'],[min(r[field] for r in shown),max(r[field] for r in shown)])
assert math.isclose(100*(1.21-1),21) and math.isclose(100*(1-1/1.21),17.35537190082645)
record=(repo/'research/PDF_RENDERING.md').read_text(encoding='utf-8')
for label,name in (('Source','docs/research/paper.md'),('PDF','docs/research/k-graphfuse-paper.pdf')):
    digest=re.search(label+r' SHA-256: `([0-9a-f]{64})`',record)
    assert digest and hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest[1],name
pdf_figures=re.findall(r'`(docs/assets/[^`]+)` — SHA-256 `([0-9a-f]{64})`',record)
controlled=json.loads((repo/'studies/controlled_20261010/VISUALS.json').read_text(encoding='utf-8'))
bound=json.loads((repo/'studies/bound_attribution_20261011/VISUALS.json').read_text(encoding='utf-8'))
for visual in (controlled,bound):
    for key in ('source_sha256','artifact_sha256'):
        for name,digest in visual[key].items():
            assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
expected_png={name for name in presentation['artifacts'] if name.endswith('.png')}
for visual in (controlled,bound):expected_png.update(name for name in visual['artifact_sha256'] if name.endswith('.png'))
assert {name for name,_ in pdf_figures}==expected_png
for name,digest in pdf_figures:
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
tex_record=json.loads((repo/'research/TEX_COMPILATION.json').read_text(encoding='utf-8'))
assert tex_record['status']=='success'
assert hashlib.sha256((repo/tex_record['source']).read_bytes()).hexdigest()==tex_record['source_sha256']
paper=(repo/'docs/research/paper.md').read_text(encoding='utf-8')
assert 'not peer-reviewed' in paper and 'Metric definitions' in paper
assert 'timing' in paper and '\\theta=0' in paper
print('PASS: percentage figures, native coverage, interval conversions, manuscript/PDF provenance and claim ranges.')
