"""Recompute the frozen studies without a GPU or dataset download."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare(old, new):
    """Paired whole-block bootstrap, stratified by separate process/input seed."""
    old, new = np.asarray(old), np.asarray(new)
    assert old.shape == new.shape and old.ndim == 3
    assert np.isfinite(old).all() and np.isfinite(new).all() and np.all(old > 0) and np.all(new > 0)
    ratio = float(np.median(old) / np.median(new))
    rng = np.random.default_rng(47119)
    index = rng.integers(0, old.shape[1], size=(2000, old.shape[0], old.shape[1]))
    run = np.arange(old.shape[0])[None, :, None]
    draws = np.median(old[run, index].reshape(2000, -1), axis=1) / np.median(new[run, index].reshape(2000, -1), axis=1)
    return dict(control_ms=float(np.median(old)), k_ms=float(np.median(new)),
        control_p95_ms=float(np.quantile(old, .95)), k_p95_ms=float(np.quantile(new, .95)),
        ratio=ratio, ratio_ci95=np.quantile(draws, [.025, .975]).tolist(),
        time_reduction_pct=100 * (1 - 1 / ratio), derived_rate_increase_pct=100 * (ratio - 1))


def selfcheck():
    row = compare(np.full((2, 5, 11), 2.), np.ones((2, 5, 11)))
    assert row['ratio'] == 2 and row['time_reduction_pct'] == 50 and row['derived_rate_increase_pct'] == 100
    assert row['ratio_ci95'] == [2, 2]
    assert compare(np.ones((2, 5, 11)), np.full((2, 5, 11), 2.))['time_reduction_pct'] == -100


def summarise():
    selfcheck()
    protocol = read(HERE / 'protocol.json')
    freeze = read(HERE / 'FREEZE.json')
    for path, expected in freeze['source_sha256'].items():
        assert digest(ROOT / path) == expected, path
    for name, field in [('RUNTIME_FREEZE.json', 'wrapper_sha256'), ('CONTROL_FREEZE.json', 'runner_sha256')]:
        lock = read(HERE / name)
        target = 'runtime.py' if name.startswith('RUNTIME') else 'controls.py'
        assert digest(HERE / target) == lock[field]
        assert lock.get('base_freeze_sha256', lock.get('runner_freeze_sha256')) == digest(HERE / 'FREEZE.json')
    app_lock = read(HERE / 'APPLICATION_FREEZE.json')
    for path, expected in app_lock['source_sha256'].items():
        assert digest(HERE / path) == expected
    groups = dict(primary=[], packed_balanced=[], application=[])
    runtime = []
    for path in sorted(RESULTS.glob('*.json')):
        data = read(path)
        if path.name == 'excluded_attempt.json':
            assert data['status'] == 'excluded_environment_failure'
            continue
        if '.runtime.' in path.name:
            assert data['wrapper_sha256'] == digest(HERE / 'runtime.py')
            assert data['cupy_accelerators'] == 'default CUB enabled'
            runtime.append(data)
            continue
        assert data['status'] == 'complete', path.name
        kind = 'application' if data['study_id'].startswith('kgf-recommendation') else 'packed_balanced' if data['study_id'].startswith('kgf-packed') else 'primary'
        lock = 'APPLICATION_FREEZE.json' if kind == 'application' else 'CONTROL_FREEZE.json' if kind == 'packed_balanced' else 'FREEZE.json'
        assert data['freeze_sha256'] == digest(HERE / lock)
        device = 'T4' if data['gpu'] == 'Tesla T4' else 'RTX 3060 Laptop'
        assert device == 'T4' or 'RTX 3060 Laptop' in data['gpu']
        if kind != 'application':
            assert data['source_sha256'] == freeze['source_sha256']
            assert len(data['cases']) == len(protocol['cases'])
            names = protocol['methods'] if kind == 'primary' else read(HERE / 'CONTROL_FREEZE.json')['methods']
            for case, spec in zip(data['cases'], protocol['cases']):
                assert all(case.get(k) == v for k, v in spec.items())
                assert [m['method'] for m in case['methods']] == names
                for m in case['methods']:
                    assert m['accepted'] and 'error' not in m
                    assert np.asarray(m['blocks']).shape == (5, 11)
                    assert m['tail_bound'] <= 1e-12
                    assert m['quality']['finite'] and max(m['quality']['frobenius']) <= 1.1e-12
            if kind == 'primary':
                assert len(data['small_accuracy']) == 6
                for case in data['small_accuracy']:
                    assert case['rows'] == 64 and case['precision_bits'] == [256, 384]
                    assert case['oracle_discrepancy'] < 1e-65
                    assert [m['method'] for m in case['methods']] == names
                    assert all(max(m['quality']['error_per_input_norm']) <= 1.1e-12 for m in case['methods'])
        else:
            assert data['dataset_sha256'] == protocol['application']['archive_sha256']
            assert data['evaluated_users'] == 256 and data['vertices'] == 9992 and data['nnz'] == 944258
            assert [m['method'] for m in data['methods']] == app_lock['methods']
            for m in data['methods']:
                if 'error' in m:
                    assert m['method'] == 'cupy_restart_captured' and m['error'] == 'NotImplementedError'
                    continue
                assert len(m['request_batch_ms']) == 5 and m['quality_within_margin']
                assert m['top10_exact_agreement'] == 1
                assert m['recall_at_10'] == data['cpu_reference_recall'] and m['ndcg_at_10'] == data['cpu_reference_ndcg']
        groups[kind].append((device, data))
    result = dict(study_id=protocol['study_id'], primary=[], packed_balanced=[], components=[], application=[],
        counts=dict(primary_case_runs=40, primary_method_checks=480, small_whole_fixture_checks=288,
            packed_balanced_case_runs=40, packed_balanced_method_checks=120, application_runs=4),
        estimator='Pooled median ratio; 2000 paired whole-block bootstrap draws, stratified by process/input seed. Descriptive, conditional on observed fastest accepted control; no multiplicity adjustment.',
        runtime=runtime)
    identity = [m for m in protocol['methods'] if 'identity' in m and not m.startswith('fused')]
    for kind in ('primary', 'packed_balanced', 'application'):
        assert len(groups[kind]) == 4
        for device in ('RTX 3060 Laptop', 'T4'):
            runs = sorted([r for d, r in groups[kind] if d == device], key=lambda r:r['seed'])
            assert [r['seed'] for r in runs] == ([20261013, 20261014] if kind == 'packed_balanced' else protocol['seeds'])
            if kind == 'application':
                methods = [{m['method']:m for m in r['methods']} for r in runs]
                old = [m['cupy_restart']['request_batch_ms'] for m in methods]
                new = [m['fused_onepass']['request_batch_ms'] for m in methods]
                row = dict(device=device, **compare(np.asarray(old)[:, :, None], np.asarray(new)[:, :, None]),
                    recall_at_10=runs[0]['cpu_reference_recall'], ndcg_at_10=runs[0]['cpu_reference_ndcg'], top10_exact_agreement=1.0,
                    setup=[dict(seed=r['seed'], load_build_ms=r['load_build_ms'], methods=[{k:v for k,v in m.items() if k in ('method','compile_ms','prepare_ms','error')} for m in r['methods']]) for r in runs])
                result[kind].append(row)
                continue
            for index, spec in enumerate(protocol['cases']):
                rows = [r['cases'][index] for r in runs]
                assert len({r['operator_sha256'] for r in rows}) == 1
                methods = [{m['method']:m for m in r['methods']} for r in rows]
                controls = identity if kind == 'primary' else ['cupy_onepass_packed', 'cupy_iterated_packed']
                control = min(controls, key=lambda name:np.median([m[name]['blocks'] for m in methods]))
                def pair(old, new):
                    return dict(**compare([m[old]['blocks'] for m in methods], [m[new]['blocks'] for m in methods]),
                        control_degree=[m[old]['degree'] for m in methods], k_degree=[m[new]['degree'] for m in methods],
                        control_penalty=[m[old]['condition_penalty'] for m in methods], k_penalty=[m[new]['condition_penalty'] for m in methods],
                        control_error=max(max(m[old]['quality']['frobenius']) for m in methods),
                        k_error=max(max(m[new]['quality']['frobenius']) for m in methods))
                measured = pair(control, 'fused_onepass_captured')
                if kind == 'packed_balanced':
                    assert measured['control_degree'] == measured['k_degree']
                result[kind].append(dict(device=device, **spec, control=control, **measured))
                if kind == 'primary':
                    for old, new, label in protocol['comparison_pairs']:
                        result['components'].append(dict(device=device, **spec, comparison=label, control=old, candidate=new, **pair(old, new)))
    for kind in ('primary', 'packed_balanced'):
        result['counts'][kind + '_pooled_faster'] = sum(r['ratio'] > 1 for r in result[kind])
        result['counts'][kind + '_pooled_strong_speed'] = sum(r['ratio'] >= 1/.9 and r['ratio_ci95'][0] > 1 and r['k_p95_ms'] <= r['control_p95_ms'] for r in result[kind])
    assert len(runtime) == 4
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    summary = summarise()
    path = HERE / 'SUMMARY.json'
    if args.verify:
        assert read(path) == summary, 'Saved summary differs from raw measurements'
        claims = {c['id']:c for c in read(ROOT / 'docs/claims.json')['claims']}
        wins = [r for r in summary['packed_balanced'] if r['ratio'] >= 1/.9 and r['ratio_ci95'][0] > 1 and r['k_p95_ms'] <= r['control_p95_ms']]
        assert claims['controlled_packed_balanced_strong_speed_20261010']['value'] == len(wins) == 18
        assert claims['controlled_packed_balanced_time_range_20261010']['value'] == [min(r['time_reduction_pct'] for r in wins), max(r['time_reduction_pct'] for r in wins)]
        assert claims['controlled_application_time_reduction_20261010']['value'] == [r['time_reduction_pct'] for r in summary['application']]
        assert claims['controlled_application_top10_agreement_20261010']['value'] == 1.0
        for name, expected in read(HERE / 'RESULTS_MANIFEST.json')['sha256'].items():
            assert digest(ROOT / name) == expected, name
        print('PASS: frozen sources, complete raw runs, matched budgets, same-estimator intervals, application quality and summary.')
    else:
        path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
        files = sorted(RESULTS.glob('*.json')) + [path, Path(__file__)]
        record = dict(sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files})
        (HERE / 'RESULTS_MANIFEST.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps(summary['counts']))
