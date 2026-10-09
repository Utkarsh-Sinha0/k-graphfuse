"""Offline evidence audit; does not substitute for independent benchmark reruns."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import zipfile
import numpy as np

root = Path(__file__).resolve().parents[1] / "evidence"
if (root / "MANIFEST.json").exists():
    manifest = json.loads((root / "MANIFEST.json").read_text())
else:
    with zipfile.ZipFile(root / "sinha_graphfuse_kaggle_release.zip") as bundle:
        assert bundle.testzip() is None, "ZIP integrity failure"
        manifest = json.loads(bundle.read("MANIFEST.json"))
for name, entry in manifest.items():
    content = (root / name).read_bytes()
    assert len(content) == entry["bytes"] and hashlib.sha256(content).hexdigest() == entry["sha256"], name

report = json.loads((root / "SINHA_GRAPHFUSE_KAGGLE_RAW.json").read_text())
for run in report["runs"]:
    raw = (root / run["raw_file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == run["raw_sha256"]
    original = json.loads(raw)
    assert original["unit_tests"] == {"run": 27, "failures": 0, "errors": 0, "skipped": 0}
    folder = (root / run["raw_file"]).parent
    tree = ast.parse((folder / "run.py").read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "files" for t in node.targets))
    files = json.loads(ast.literal_eval(assignment.value.args[0]))
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    assert hashes == original["source_hashes"] == json.loads((folder / "source-manifest.json").read_text())
    selected = [c for c in report["cases"] if c["run_id"] == run["id"]]
    assert [{k: v for k, v in c.items() if k != "run_id"} for c in selected] == original["cases"]

with (root / "SINHA_GRAPHFUSE_COMPARISONS.csv").open(encoding="utf-8-sig", newline="") as handle:
    comparisons = list(csv.DictReader(handle))
assert len(comparisons) == len(report["cases"])
for case, row in zip(report["cases"], comparisons):
    assert (row["case"], row["run_id"]) == (case["case"], case["run_id"])
    for candidate in case["candidates"]:
        if "resident" in candidate:
            timing = candidate["resident"]
            assert len(timing["samples_ms"]) == timing["sample_count"]
            np.testing.assert_allclose(np.median(timing["samples_ms"]), timing["median_ms"], rtol=1e-12)
            np.testing.assert_allclose(np.quantile(timing["samples_ms"], .95), timing["p95_ms"], rtol=1e-12)
    if not row["gpu_speedup"]:
        assert row["measured_gpu_bar"] == "NO QUALIFIED COMPARISON"
        continue
    eligible = [c for c in case["candidates"] if c.get("accepted") and c.get("gpu") and "resident" in c]
    own = min((c for c in eligible if c["method"].startswith("Sinha GraphFuse")), key=lambda c: c["resident"]["median_ms"])
    other = min((c for c in eligible if not c["method"].startswith("Sinha GraphFuse")), key=lambda c: c["resident"]["median_ms"])
    assert row["sinha_method"] == own["method"] and row["gpu_baseline"] == other["method"]
    a, b = np.asarray(other["resident"]["samples_ms"]), np.asarray(own["resident"]["samples_ms"])
    rng = np.random.default_rng(20261009)
    ratios = np.median(a[rng.integers(len(a), size=(5000, len(a)))], axis=1) / np.median(b[rng.integers(len(b), size=(5000, len(b)))], axis=1)
    interval = np.quantile(ratios, [.025, .975])
    np.testing.assert_allclose(interval, [float(row["gpu_speedup_95ci_low"]), float(row["gpu_speedup_95ci_high"])], rtol=1e-12)
    np.testing.assert_allclose(np.median(a)/np.median(b), float(row["gpu_speedup"]), rtol=1e-12)
    passed = np.median(b) <= .9*np.median(a) and interval[0] > 1 and own["resident"]["p95_ms"] <= other["resident"]["p95_ms"]
    assert passed == (row["measured_gpu_bar"] == "MEASURED GPU BAR PASS")
print(f"PASS: {len(report['runs'])} run snapshots, {len(comparisons)} case records, source and package hashes, timings, comparator selection and GPU-bar verdicts")
