"""Recompute claims from frozen samples; reject incomplete or mismatched evidence."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

base=Path(__file__).parent
rng=np.random.default_rng(20261010)
summary=dict(status='verified',local=[],cloud=[],cold=[])
def interval(old,new):
    old=np.array(old);new=np.array(new);indices=rng.integers(0,len(old),(20000,len(old)))
    values=np.median(old[indices],axis=1)/np.median(new[indices],axis=1)
    return [float(v) for v in np.quantile(values,[.025,.975])]

def ranking(path,destination):
    report=json.loads(path.read_text());assert report['status']=='complete'
    for name,digest in report['source_hashes'].items():
        candidate=base/'source'/name
        if not candidate.exists():candidate=base/'source/core'/name
        assert candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest()==digest,(name,'source mismatch')
    for case in report['cases']:
        expected=32 if destination=='cloud' else 24
        assert len(case['candidates'])==expected
        accepted={r['method']:r for r in case['confirmation'] if r['accepted']}
        assert all(name in accepted for name in ('released','older_bound','metric'))
        for record in accepted.values():
            assert len(record['blocks'])==5 and record['resident']['sample_count']==155
            samples=[]
            for block in record['blocks']:
                assert len(block['samples_ms'])==31 and block['quality']['accepted']
                assert all(np.isfinite(e) and e<=case['budget'] for e in block['quality']['errors'])
                samples.extend(block['samples_ms'])
            assert np.isclose(np.median(samples),record['resident']['median_ms'],rtol=0,atol=1e-12)
            assert np.isclose(np.quantile(samples,.95),record['resident']['p95_ms'],rtol=0,atol=1e-12)
        controls=[r for name,r in accepted.items() if name in ('released','older_bound','cupy_released')]
        old=min(controls,key=lambda r:r['resident']['median_ms']);new=accepted['metric']
        ratio=old['resident']['median_ms']/new['resident']['median_ms']
        ci=interval([b['median_ms'] for b in old['blocks']],[b['median_ms'] for b in new['blocks']])
        candidates=case['candidates']
        row=dict(n=case['n'],q=case['q'],sensitivity=case['sensitivity'],gpu=report['gpu'],
            old_method=old['method'],old_degree=case['degrees']['released'],new_degree=case['degrees']['metric'],
            old_ms=old['resident']['median_ms'],new_ms=new['resident']['median_ms'],speedup=ratio,
            time_reduction_percent=100*(1-1/ratio),ci_low=ci[0],ci_high=ci[1],
            old_p95_ms=old['resident']['p95_ms'],new_p95_ms=new['resident']['p95_ms'],
            gate=bool(ratio>=1.1 and ci[0]>1 and new['resident']['p95_ms']<=old['resident']['p95_ms']),
            accepted_configurations=sum(bool(c.get('accepted')) for c in candidates),
            accuracy_rejected=sum('errors' in c and not c['accepted'] for c in candidates),
            exceptions=sum('error' in c for c in candidates),not_run=sum('not_run' in c for c in candidates),
            maximum_metric_error=max(e for b in new['blocks'] for e in b['quality']['errors']),
            original_tail_bound=case['metric_original_bound'],original_derivative_bound=case['metric_derivative_bound'],
            selected_configuration=new['configuration'])
        if 'cupy_metric' in accepted:row['cupy_metric_ms']=accepted['cupy_metric']['resident']['median_ms']
        summary[destination].append(row)

ranking(base/'LOCAL_RESULTS.json','local')
for family in ('filters','sensitivity'):
    cloud=base/'kaggle'/family/'kgf_math_upgrade/RESULTS.json'
    if cloud.exists():ranking(cloud,'cloud')

cold=json.loads((base/'COLD_RESULTS.json').read_text());assert cold['status']=='complete'
for case in cold['cases']:
    assert all(len(v)==11 and all(r['accepted'] for r in v) for v in case['methods'].values())
    old=case['methods']['released'];new=case['methods']['zero_tail_fastpath']
    ratio=float(np.median([r['total_ms'] for r in old])/np.median([r['total_ms'] for r in new]))
    summary['cold'].append(dict(n=case['n'],old_ms=float(np.median([r['total_ms'] for r in old])),new_ms=float(np.median([r['total_ms'] for r in new])),speedup=ratio,
        old_compile_ms=float(np.median([r['compile_ms'] for r in old])),new_compile_ms=float(np.median([r['compile_ms'] for r in new])),
        ci=interval([r['total_ms'] for r in old],[r['total_ms'] for r in new]),maximum_error=max(r['error_per_input_norm'] for r in new)))
(base/'VERIFIED_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
for name in ('local','cloud','cold'):
    rows=summary[name]
    if not rows:continue
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (base/f'{name}_comparisons.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def table(rows):
    lines=['| Vertices | Channels | Calculation | Fastest old control | Old ms | Metric ms | Speedup | 95% block interval | Gate |',
           '|---:|---:|---|---|---:|---:|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['n']:,} | {r['q']} | {'Value + derivative' if r['sensitivity'] else 'Polynomial filter'} | {r['old_method']} | {r['old_ms']:.3f} | {r['new_ms']:.3f} | {r['speedup']:.2f}x | {r['ci_low']:.2f}–{r['ci_high']:.2f}x | {'PASS' if r['gate'] else 'FAIL'} |")
    return '\n'.join(lines)

text='''# K GraphFuse private mathematics upgrade: measured results

Two targeted changes are implemented and tested: similarity-aware polynomial/derivative planning, and skipping unnecessary range bounds when retaining every coefficient. These are improvements to the current implementation, not a claim of new mathematical priority or universal superiority.

## Local resident GPU results

Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB. Every case uses the same alternating asymmetric directed operator family, constructed features and degree-48 matrix-valued coefficients. Budget: 1e-8 output Frobenius error per input Frobenius norm. Five randomized held-out blocks of 31 synchronized samples per selected method, after seven pilot samples per accuracy-accepted configuration. Each method independently searches FP32/FP64, packed/unpacked and eager/captured configurations. Input/output similarity scaling is inside the timed query.

'''+table(summary['local'])+'''

The old control is whichever is faster in confirmation: the released constant-two planner or the optimized older 1+sqrt(2) planner. It is not native cuGraph. Reported intervals resample the five paired whole-block medians 20,000 times; they do not express confidence over arbitrary graphs, hardware or customer workloads. Gate: at least 10% lower median, lower interval endpoint above 1, and no worse p95.

## Private matched Kaggle results

'''+(table(summary['cloud']) if summary['cloud'] else 'Two independent private T4 jobs are running. No T4 improvement is claimed before completion and audit.')+'''

T4 follow-up adds FP32/FP64 ordinary and packed CuPy Horner controls with the same polynomial/derivative semantics. Both original-coordinate and metric-coordinate CuPy adapters are recorded. The primary metric comparison in this table uses the PyTorch prototype against the fastest accepted original-coordinate control, including CuPy. CuPy cannot capture the tested sparse CSR operation in the installed version, so only its eager variants are searched. This is not a rerun of the 104 earlier ranking/filter cases or a native cuGraph PageRank comparison.

## Local cold full-polynomial preparation

Fixed ten-update, directed weighted column-stochastic propagation, FP64 eager execution, both methods retain degree 10. Eleven randomized samples/method from a loaded normalized CSR through validation, compilation, upload, first execution and full host output. File import and graph normalization are outside the timing. This is a diagnostic with fixed settings, not a held-out configuration competition.

| Vertices | Current total ms | Zero-tail fast path ms | Speedup | Current compile ms | Fast compile ms |
|---:|---:|---:|---:|---:|---:|
'''
for r in summary['cold']:text+=f"| {r['n']:,} | {r['old_ms']:.3f} | {r['new_ms']:.3f} | {r['speedup']:.2f}x | {r['old_compile_ms']:.3f} | {r['new_compile_ms']:.3f} |\n"
text+='''
The fast path reuses exactly the same executor and all coefficients. Its truncation error is zero because it discards no terms. It does not certify zero floating error or claim a warm-runtime reduction. Every measured output passed a full CPU reference with an observed normalized error <=1e-12.

## Meaning and boundaries

- Repeated asymmetric multi-feature propagation: fewer sparse passes at the same accepted numerical budget.
- Affine parameter sensitivity: compute a value and parameter derivative jointly with fewer retained terms; possible use in repeated network-response or simulation sweeps.
- Frequently rebuilt finite-propagation operators: reduce validation-to-first-output preparation overhead by omitting a mathematically unnecessary tail enclosure.
- All application and cloud-bill benefits still require workflow measurements. No customer deployment, energy reduction or money saved was measured.
- No automatic guarantee of improvement on already balanced graphs, huge condition penalties, real public graphs, or all earlier PageRank cases. Do not enable the metric blindly.
- Four earlier PageRank accuracy gaps, Graph500 native-cuGraph losses and unsupported graph algorithms remain unresolved.

## Reproduce and inspect

`LOCAL_RESULTS.json` and `COLD_RESULTS.json` preserve samples and rejected configurations. `VERIFIED_SUMMARY.json` and CSV files are derived by `audit_results.py`. `source/` contains the frozen prototype, regression checks, and local/T4 benchmark harnesses. `DERIVATION.md` contains the coordinate and derivative error transfer; `RESEARCH.md` links the selected research and classical prior art.

From the original workspace, run `work/.venv/Scripts/python.exe outputs/kgf_math_upgrade/audit_results.py`. Re-running timings requires the recorded numerical core, NumPy/SciPy/PyTorch and a CUDA device. T4 runs are independently frozen private Kaggle scripts; only returned completed evidence enters the report. Public repository and historical benchmark files are unchanged.
'''
text=text.replace('Input/output similarity scaling is inside the timed query.','Input/output similarity scaling is inside the timed query. These are prevalidated resident-compute timings; host-side input/output validation is outside the timed kernel loop. The checked functional API is separately regression-tested and is not the timed replay adapter.')
text=text.replace('T4 follow-up adds FP32/FP64 ordinary and packed CuPy Horner controls','T4 follow-up uses PyTorch 2.11.0+cu128, NumPy 2.1.3, SciPy 1.16.3 and CuPy 14.0.1. It adds FP32/FP64 ordinary and packed CuPy Horner controls')
(base/'RESULTS.md').write_text(text,encoding='utf-8')
print(json.dumps(dict(local_cases=len(summary['local']),local_gates=sum(r['gate'] for r in summary['local']),cloud_cases=len(summary['cloud']),cloud_gates=sum(r['gate'] for r in summary['cloud']),cold_cases=len(summary['cold']),local_speedup_range=[min(r['speedup'] for r in summary['local']),max(r['speedup'] for r in summary['local'])]),indent=2))
