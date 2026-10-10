"""Offline completeness and checksum check; run after extracting the package."""
from pathlib import Path
import hashlib, json

base = Path(__file__).resolve().parent
manifest = json.loads((base / 'PACKAGE_SHA256.json').read_text())
files = {p.relative_to(base).as_posix() for p in base.rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and p.name != 'PACKAGE_SHA256.json'}
assert files == set(manifest), 'Package files differ from the manifest'
for name, digest in manifest.items():
    path = (base / name).resolve()
    assert path.is_relative_to(base), 'Manifest path escapes the package'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Checksum mismatch: {name}'
tests = json.loads((base / 'FINAL_LOCAL_TESTS.json').read_text())
for name, digest in tests['source_hashes'].items():
    assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, f'Tested source changed: {name}'
assert all(tests[k]['passed'] and not tests[k]['skipped'] for k in ('core', 'upgrade'))
print(f'Verified {len(manifest)} file hashes; tested API/core sources match')
