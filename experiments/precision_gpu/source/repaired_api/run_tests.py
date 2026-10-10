"""Combined import-order regression; requires a real CUDA GPU and no skipped checks."""
from pathlib import Path
import hashlib, json, os, sys, unittest

base = Path(__file__).resolve().parent
paths = [str(base), str(base / 'source/core')]
sys.path[:0] = paths
os.environ['PYTHONPATH'] = os.pathsep.join(paths)
# Reproduce the preloaded compiler and core-before-upgrade stream conditions.
import torch
torch.cuda.init()

report = {}
for name, directory in (('core', base / 'source/core_tests'), ('upgrade', base)):
    suite = unittest.TestLoader().discover(str(directory), top_level_dir=str(directory))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report[name] = dict(run=result.testsRun, failures=len(result.failures),
                        errors=len(result.errors), skipped=len(result.skipped),
                        passed=result.wasSuccessful() and not result.skipped)
report['gpu'] = torch.cuda.get_device_name(0)
report['torch'] = torch.__version__
import cupy
report['cupy'] = cupy.__version__
tested = {'run_tests.py', 'metric_prototype.py', 'fused_dd.py', 'fused_dd.cu', 'fused_acc.py',
          'fused_acc.cu', 'coefficient_certificate.py', 'check_high_precision.py'}
report['source_hashes'] = {str(p.relative_to(base)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
    and ((p.parent == base and (p.name in tested or p.name.startswith('test_')))
         or 'core' in p.parts or 'core_tests' in p.parts)
    and p.suffix in ('.py', '.cu')}
(base / 'FINAL_LOCAL_TESTS.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
assert all(report[name]['passed'] for name in ('core', 'upgrade')), 'Combined checks failed or skipped'
