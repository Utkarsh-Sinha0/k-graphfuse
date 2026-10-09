# Contributing

Use a focused change and preserve source/evidence provenance. Do not overwrite old benchmark records or change a gate after observing results. Cite mathematical sources and retain all applicable license notices.

```sh
python -m pip install -e ".[docs]"
python -m unittest discover -s tests -v
python examples/quickstart.py
python examples/sensitivity.py
python examples/resolvent.py
python benchmarks/verify_evidence.py
python -m mkdocs build --strict
```

For GPU changes, also run `python examples/gpu.py` and the CUDA tests on a real device. Report explicit skips. Use a small independent regression check for substantive math changes. Benchmarks require matching semantics and complete accepted/rejected records.

New source and documentation contributions use Apache 2.0; changes to the historical MIT components preserve their MIT provenance and notices. No contributor license agreement is required by this release.

Documentation follows task-oriented tutorials/guides, conceptual explanations and explicit API contracts. Include prerequisites, expected outputs, errors, shapes, units and limitations. Avoid unverified global SOTA, customer savings or ranking guarantees.
