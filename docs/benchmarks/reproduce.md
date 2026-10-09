# Reproduce and audit the evidence

An offline audit checks saved facts. A new benchmark execution measures your environment. Keep these two outcomes separate.

## Audit without a GPU or network

From an installed repository checkout:

```sh
python benchmarks/verify_evidence.py
```

The audit checks the publication manifest, unchanged module/frozen-runner hashes, all recorded sample statistics, candidate acceptance, comparator selection, bootstrap intervals, claim calculations and earlier-suite verdicts. It does not rerun GPU timings. Private account metadata, screenshots and installation logs are intentionally excluded; the transformation ledger records the public copy changes.

## Audit the saved released CPU scores

The repository includes the wrapper's generated `arcade_scores.bin` and timings. Obtain `datagen-7_5-fb-PR` from the [official Graphalytics dataset archive](https://datasets.ldbcouncil.org/graphalytics/datagen-7_5-fb.tar.zst), respecting its dataset terms, then run:

```sh
python benchmarks/verify_evidence.py --reference /path/to/datagen-7_5-fb-PR
```

The audit verifies the reference SHA-256 before comparing aligned output identifiers and scores. The original third-party graph/reference files are not redistributed here. Their archive, vertex, edge and reference hashes are in the raw record.

## Rerun the exact public-reference experiment

Use a disposable **Linux CUDA/Kaggle workspace**, internet access, sufficient disk/RAM and time to import the graph. The historical runner uses `/kaggle/working` and downloads official graph data, vendor binaries and Java runtime. It does not use an API credential or private account identifier.

The recorded environment uses Python 3.13, PyTorch 2.11.0+cu128, NumPy 2.1.3, SciPy 1.16.3, CuPy 14.0.1 and pylibcugraph 26.10.0. Install the matching CUDA-enabled PyTorch build for your driver separately. Install the comparator requirements, then execute the frozen bootstrap:

```sh
python -m pip install -r benchmarks/requirements-reference.txt
python evidence/sinha_public_reference_evidence/run.py
```

The bootstrap also checks installation of its pinned cuGraph comparator and zstandard dependency. Do not interpret a different dependency build as an exact environment replication; record the actual versions in a new run.

The bootstrap writes its original module snapshot into the disposable workspace and executes the benchmark. Outputs include `/kaggle/working/sinha_public_reference/evidence/industry_reference.json`, CPU scores, timings and resource samples. It can take substantially longer or fail on a different environment. Archived timing does not guarantee the same output latency.

## Rerun the broader suite

The original large run and two recovery bootstraps are retained:

```sh
python evidence/sinha_benchmark_evidence/large/run.py
python evidence/sinha_benchmark_evidence/community/run.py
python evidence/sinha_benchmark_evidence/scale-gpu/run.py
```

Use separate disposable runs because output locations can overlap. The original large run contains historical community URL failures and a time-limited checkpoint; it is preserved verbatim rather than repaired under the same identity. Recovery runs cover the missing workloads. Read the protocol copies before interpreting any partial result.

## Publish a replication

Record the commit, actual library builds, device/driver, graph and input hashes, complete configuration search, accuracy gates, preparation scopes, all samples, rejected candidates, resource method and run status. Use repeated independent runs and randomized order for a stronger performance study. Do not replace old evidence with a new result; add a separate run identity.
