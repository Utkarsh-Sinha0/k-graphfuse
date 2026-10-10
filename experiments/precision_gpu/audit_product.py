"""Conservative speed/accuracy gates for the frozen product experiments."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

base=Path(__file__).parent;rng=np.random.default_rng(20261010)
records=[];all_methods=[]
def interval(old,new):
    indices=rng.integers(0,5,(20000,5));a=np.asarray(old);b=np.asarray(new)
    return [float(v) for v in np.quantile(np.median(a[indices],axis=1)/np.median(b[indices],axis=1),[.025,.975])]
def quality(record,key):return np.max([b['quality'][key] for b in record['blocks']],axis=0)
def pilot(case,record):
    matches=[c for c in case['candidates'] if c['method']==record['method'] and all(c[k]==v for k,v in record['configuration'].items())]
    assert len(matches)==1 and matches[0]['quality']['accepted'];return matches[0]['pilot']['median_ms']
def process(path,source,platform):
    report=json.loads(path.read_text());assert report['status']=='complete'
    for name,digest in report['source_hashes'].items():
        options=[source/name,base/'source/product_v1'/name,base/'source/core'/name]
        assert any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==digest for p in options),(name,'source mismatch')
    for case in report['cases']:
        assert len(case['candidates']) in (8,14)
        for item in case['confirmation']:
            assert len(item['blocks'])==5
            for block in item['blocks']:
                assert len(block['samples_ms'])==31 and block['quality']['accepted'] and block['quality']['finite']
                assert max(block['quality']['global_reference_errors'])<=1e-13
            samples=[v for block in item['blocks'] for v in block['samples_ms']]
            item['median_ms']=float(np.median(samples));item['p95_ms']=float(np.quantile(samples,.95))
        controls=[r for r in case['confirmation'] if r['method'] in ('torch','cupy')];assert len(controls)==2
        old=min(controls,key=lambda r:r['median_ms']);old_error=quality(old,'sampled_errors');old_absolute=quality(old,'sampled_max_absolute')
        eligible=[]
        for item in case['confirmation']:
            if not item['method'].startswith('fused_'):continue
            error=quality(item,'sampled_errors');absolute=quality(item,'sampled_max_absolute')
            improved=bool(np.all(error<=old_error) and np.all(absolute<=old_absolute) and (np.any(error<old_error) or np.any(absolute<old_absolute)))
            ci=interval([b['median_ms'] for b in old['blocks']],[b['median_ms'] for b in item['blocks']])
            ratio=old['median_ms']/item['median_ms'];speed=bool(ratio>=1/.9 and ci[0]>1 and item['p95_ms']<=old['p95_ms'])
            row=dict(platform=platform,n=case['n'],q=case['q'],sensitivity=case['sensitivity'],method=item['method'],old_method=old['method'],old_ms=old['median_ms'],new_ms=item['median_ms'],speedup=ratio,ci_low=ci[0],ci_high=ci[1],old_p95_ms=old['p95_ms'],new_p95_ms=item['p95_ms'],accuracy_gate=improved,speed_gate=speed,dual_gate=bool(improved and speed),pilot_ms=pilot(case,item),old_sampled_errors=old_error.tolist(),new_sampled_errors=error.tolist(),old_sampled_max_absolute=old_absolute.tolist(),new_sampled_max_absolute=absolute.tolist(),worst_component_error_reduction_percent=100*(1-float(np.max(error)/np.max(old_error))),max_global_reference_error=max(v for b in item['blocks'] for v in b['quality']['global_reference_errors']),configuration=item['configuration'],source_report=str(path.relative_to(base)))
            all_methods.append(row)
            if improved:eligible.append(row)
        if eligible:
            # Accuracy screen is numerical, and method choice uses pilot timings.
            chosen=min(eligible,key=lambda r:r['pilot_ms']);records.append(chosen)
        else:records.append(dict(platform=platform,n=case['n'],q=case['q'],sensitivity=case['sensitivity'],dual_gate=False,method='no eligible candidate',source_report=str(path.relative_to(base))))

process(base/'PRODUCT_LOCAL_RESULTS.json',base/'source/product_v1','RTX 3060')
process(base/'WIDE_LOCAL_RESULTS.json',base/'source/wide_v1','RTX 3060 wide follow-up')
for family in ('filters','sensitivity'):
    process(base/f'kaggle/product-{family}/kgf_product/RESULTS.json',base/'source/product_v1','T4')
process(base/'kaggle/product-wide/kgf_product/RESULTS.json',base/'source/wide_v1','T4 wide follow-up')
final={}
for row in records:
    if row['platform'].startswith('T4'):final[(row['n'],row['q'],row['sensitivity'])]=row
chosen=list(final.values());assert len(chosen)==6
for name,rows in (('product_selected',records),('product_all_methods',all_methods)):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (base/f'{name}.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
summary=dict(status='verified',selection_policy='Accuracy screen: every sampled Frobenius and max-absolute component no worse, at least one improves; fastest accuracy-eligible pilot method. Speed: >=10% lower median, paired five-block 95% bootstrap interval above 1, p95 no worse. Strongest control selected conservatively by held-out median.',t4_cases=chosen,t4_dual_gates=sum(r['dual_gate'] for r in chosen),all_selected=records,all_methods=all_methods,
    limits=['One constructed asymmetric graph family with geometrically decaying constructed coefficients','Full FP64 CPU reference checks for every output; 384-bit truth checks on first 16 rows, not all vertices','Resident validated-input compute; excludes I/O, finite-state checks and compilation/JIT','Profile selection is reconstructed from recorded pilots; policy and wide follow-up were developed after exploratory results','No current native-cuGraph PageRank comparison, broad ranking rerun, customer savings or global SOTA claim'])
(base/'PRODUCT_VERIFIED.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
lines=['# K GraphFuse precision GPU portfolio — verified results','','The new implementation combines a similarity-aware mathematical plan with fused GPU execution and selective higher-precision accumulation. It is an experimental GPU upgrade; the released repository and historical evidence are unchanged.','','## Independent Tesla T4 comparisons','','Each row compares the selected accuracy-eligible profile with the fastest accepted original-coordinate FP64 control, independently tuned for packed/unpacked coefficients and eager/captured execution where supported. Controls are CuPy 14.2.0 and PyTorch 2.11.0+cu128 on the same T4. The wide row is a separately frozen confirmation after the initial wide-feature candidates lost.','','| Vertices | Channels | Calculation | Control | Control ms | New ms | Speedup | 95% block interval | Worst sampled component error decrease | Speed + accuracy gate |','|---:|---:|---|---|---:|---:|---:|---|---:|---|']
for r in chosen:
    if 'speedup' not in r:continue
    lines.append(f"| {r['n']:,} | {r['q']} | {'Value + derivative' if r['sensitivity'] else 'Polynomial filter'} | {r['old_method']} | {r['old_ms']:.3f} | {r['new_ms']:.3f} | {r['speedup']:.2f}x | {r['ci_low']:.2f}–{r['ci_high']:.2f}x | {r['worst_component_error_reduction_percent']:.4f}% | {'PASS' if r['dual_gate'] else 'FAIL'} |")
lines+=['','Five randomized confirmation blocks of 31 synchronized samples per selected method, following seven pilot samples/configuration. Fourteen FP64 configurations per main case and eight in the separate wide confirmation. Input/output similarity scaling and an owned GPU output copy are inside timing. Validation, preparation, JIT and host transfer are outside it. Every rejected or slower candidate is retained.','','Accuracy acceptance compares **every** sampled output/derivative Frobenius error component and maximum absolute error against the fastest control; at least one must improve. The displayed percentage uses the largest Frobenius component, so it does not describe every component or every vertex. Values already lie near the FP64 rounding floor: small value-error improvements are real but modest. Derivative components can improve substantially more.','','## What changed and why','','- Transform the operator with a fixed dyadic similarity, paying its exact condition penalty in both value and derivative budgets. The smaller geometry permits a stricter truncation target without a very long polynomial.','- Fuse sparse products, coefficient injection and joint derivative updates into fewer GPU launches and fewer intermediate arrays.','- Store coefficients as high plus low parts where that helps; use FMA, compensated sums or mixed value/derivative precision according to the measured accuracy contract.','- For wide features, project coefficients with standard BLAS and use a separately compiled compensated final kernel. Compensation inside every iteration increased register pressure and latency; a final-only kernel avoids that cost.','- Reuse state buffers and perform final scaling in place internally. Checked API calls still return owned outputs and reject invalid states.','','## Mathematical and numerical evidence','','`DERIVATION.md` proves the coordinate error transfer. `coefficient_certificate.py` bounds coefficient-encoding error exactly using rational arithmetic and the existing complete-Crouzeix/derivative bounds. Main fused plans target 1e-16 truncation; the high-only wide follow-up targets 1e-15. These certificates cover exact-arithmetic truncation and encoding, not every floating operation.','','Every full output was compared against full degree-48 FP64 CPU evaluation with an accepted normalized difference <=1e-13. Independent 256/384-bit references and two finite influence windows cross-check the first 16 rows of the exact large inputs. Window/precision discrepancies were below 1e-65. The oracle distinguishes real error from merely matching an already rounded FP64 answer. See `BENCHMARK_HIGH_PRECISION.json` and the raw product reports.','','## Complexity and useful workloads','','For n vertices, m stored edges, q channels and retained degree d, work remains O(d m q + d n q^2). This iteration improves constants, memory traffic and numerical evaluation; it does not implement a new matrix-multiplication exponent. Narrow fused workspace is O(m + n q + d q^2); wide dense projection adds O(d n q), matching the packed control class. No energy or cloud-bill reduction was measured.','','Best-fit hypotheses are repeated sparse graph-feature refreshes, ranking/retrieval feature stages, linear diffusion filters and diagonal-parameter network/simulation sweeps that need both value and sensitivity. Downstream fraud detection, recommendation quality and production savings still require workflow validation. These are not BFS, shortest paths, graph transactions or HFT trading-alpha results.','','## Robustness and reproduction','','The current tests cover canonical sparse input checks, finite real states, dtype/shape/device/contiguity boundaries, parameter intervals, projection index limits, owned outputs, capture correctness, noncommuting coefficient fusion, exact zero operators, general asymmetric/symmetric controls and an independent high-precision rounding-floor check. The fused backend is calibrated for 1–16 channels and diagonal parameter directions. One prepared workspace must not run concurrent queries.','','Use `README.md` for CPU and checked GPU examples. `source/product_v1` and `source/wide_v1` preserve exact timed versions; each returned Kaggle run has a source manifest, complete numerical results and test outcomes. `product_all_methods.csv` includes losses; `PRODUCT_VERIFIED.json` contains the gate calculations. Run `audit_product.py` to recompute claims without rerunning CUDA.','','## Scope before adoption','','This is one constructed asymmetric graph family at 10,000–1,000,000 vertices with 4/16 channels and constructed polynomial coefficients. It is not a broad public-graph validation or a global SOTA certification. The previous 104-case ranking/filter suite, four PageRank accuracy gaps, native Graph500 losses and unsupported graph algorithms have not been rerun or resolved by this package. JIT, cold startup, end-to-end transfers, resilience, energy and customer value remain separate gates.','','The product uses classical prior art and a derived application of the cited mathematical premise. It makes no claim of new mathematical priority or absence of competitors. These experimental records are published separately from the installed package API.']
lines += ['', '## Final API repair and chart', '',
    'The final checked API uses a self-contained finite-check kernel and explicit stream ordering for default-stream CuPy/PyTorch projection. The 29 core and 14 upgrade tests passed together after these repairs; see `FINAL_LOCAL_TESTS.json`. Exact performance versions remain frozen in `source/product_v1` and `source/wide_v1`. The large latency runs were not repeated on the repaired API.', '',
    '![Six audited T4 comparisons](figures/precision_t4_comparison.png)', '',
    'The chart is derived from `PRODUCT_VERIFIED.json`; `make_chart.py` regenerates the PNG and SVG without running a benchmark.']
(base/'PRODUCT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(t4_cases=6,t4_dual_gates=summary['t4_dual_gates'],selected=[(r['n'],r['q'],r['sensitivity'],r['method'],round(r.get('speedup',0),2),r['dual_gate']) for r in chosen]),indent=2))
