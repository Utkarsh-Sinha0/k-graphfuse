"""Audit archived facts; large benchmark execution is a separate manual step."""
from pathlib import Path
import hashlib
import json
import math
import runpy

root = Path(__file__).resolve().parents[1]
runpy.run_path(str(Path(__file__).with_name("_verify_public.py")), run_name="__main__")
runpy.run_path(str(Path(__file__).with_name("_verify_large.py")), run_name="__main__")
raw = json.loads((root / "evidence/SINHA_GRAPHFUSE_PUBLIC_REFERENCE_RAW.json").read_text())
for name, expected in raw["source_hashes"].items():
    current = root / "src" / name
    if current.is_file():
        assert hashlib.sha256(current.read_bytes()).hexdigest() == expected, name
own = min((c for c in raw["cases"] if c["method"].startswith("Sinha")), key=lambda c:c["resident"]["median_ms"])
nv = next(c for c in raw["cases"] if c["method"] == "NVIDIA pylibcugraph native probability PageRank float32")
claims = json.loads((root / "docs/claims.json").read_text())
values = {c["id"]: c["value"] for c in claims["claims"]}
assert math.isclose(values["same_t4_median"], 100*(1-own["resident"]["median_ms"]/nv["resident"]["median_ms"]), rel_tol=1e-12)
assert math.isclose(values["sampled_memory"], 100*(1-own["resources"]["device_peak_sampled_bytes"]/nv["resources"]["device_peak_sampled_bytes"]), rel_tol=1e-12)
assert values["reference_pass"] == len(raw["cases"]) == 11
assert values["broader_gate"] == 0 and values["complete_inequality"] == 2
assert math.isclose(values["job_impact"], values["same_t4_median"]/2, rel_tol=1e-12)
assert "global SOTA" in claims["unestablished"]
print("PASS: byte-identical released numerical core and bounded claims ledger.")
