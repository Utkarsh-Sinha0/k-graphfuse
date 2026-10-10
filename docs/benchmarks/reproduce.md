# Reproduce and audit the evidence

The [new controlled-study protocol](../research/controlled-study.md#reproduction-and-integrity) gives separate primary, packed-balanced and specialized application commands. `python studies/controlled_20261010/analyse.py --verify` recomputes the new raw results, source freezes, budget gates, same-estimator intervals and quality parity without GPU access. The full repository audit runs it automatically.

An offline audit checks saved facts. A new benchmark execution measures your environment. Keep these two outcomes separate.

## Audit without a GPU or network

From an installed repository checkout:

```sh
python benchmarks/verify_evidence.py
```

The audit checks the publication manifest, unchanged module/frozen-runner hashes, all recorded sample statistics, candidate acceptance, comparator selection, bootstrap intervals, claim calculations and earlier-suite verdicts. It does not rerun GPU timings. Private account metadata, screenshots and installation logs are intentionally excluded; the transformation ledger records the public copy changes.

The extended audit also checks all 15 broad-suite source snapshots, 111 case records and 235 confirmed comparison rows; recomputes native denominators and the 59 speed / 26 speed-and-error wins; and reruns the precision study's offline derivation in a temporary copy. It verifies experimental package completeness and checks that the tested API hashes match the published source. CPU CI performs this audit without installing cuGraph, CuPy or PyTorch.

## Check the experimental GPU API

The experimental source is separate from the installed `k_graphfuse` package. Install the recorded compatible CUDA dependencies, then run from the experiment directory:

```sh
cd experiments/precision_gpu
python verify_package.py
python run_tests.py
```

The GPU regression runner requires a real CUDA device and fails on skipped checks. It initializes PyTorch first, runs 29 core tests and then 14 upgrade tests in the same process to exercise compiler/header and stream interoperability. It writes a new local test record; that result is not a new large benchmark. Read the [experiment usage instructions](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/experiments/precision_gpu/README.md) before execution.

## Reproduce frozen precision timings

Use a disposable Linux CUDA environment. The recorded T4 builds are CuPy 14.2.0 and PyTorch 2.11.0+cu128, with NumPy 2.1.3, SciPy 1.16.3, Mpmath 1.3.0 and threadpoolctl. Install a matching CUDA-enabled PyTorch build separately. No private notebook account or credential is needed to run the frozen scripts locally.

Materialize the main frozen source into an empty workspace, preserving the included core layout:

```sh
mkdir -p /tmp/kgf-product/source /tmp/kgf-product/results
cp -R experiments/precision_gpu/source/core /tmp/kgf-product/source/core
cp experiments/precision_gpu/source/product_v1/* /tmp/kgf-product/
cd /tmp/kgf-product
export KGF_PRODUCT_OUTPUT=/tmp/kgf-product/results
export KGF_PRODUCT_CASES='[[10000,4,false],[100000,4,false],[1000000,4,false],[100000,16,false]]'
python benchmark_product_cloud.py
```

For sensitivities, create a second empty workspace with the same main/core files and set `KGF_PRODUCT_CASES='[[100000,4,true],[1000000,4,true]]'`.

For the wide follow-up, create a third empty workspace. Copy the main source first, then overlay `source/wide_v1` so its exact wide executor and script replace their main versions; set `KGF_PRODUCT_CASES='[[100000,16,false]]'`. The recorded wide snapshot intentionally reuses some unchanged main/core files. Use the source hashes in the returned reports to verify the materialized version. The initial wide filter in the first run preserves the losing profiles; the third run supplies the selected wide confirmation.

The frozen timed code predates the final default-stream interoperability repair. Use the current checked API for application evaluation, and the frozen code only to reproduce its recorded experiment in a suitable environment. New hardware, builds, input values or preparation paths require a new run identity. Captured timing replays theta zero; endpoint accuracy checks are separate.

For the broad suite, each `evidence/2026-10-10/detailed/raw/<suite>/run.py` is a standalone frozen Linux/Kaggle bootstrap. It can install/download benchmark dependencies and datasets in a disposable workspace. Review the corresponding `profile.json` before running it. Parallel workers must use separate GPUs or isolated runs; timing simultaneous jobs on one GPU can create contention.

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

## Rebuild the publication figures and PDF

The presentation scripts reuse saved evidence; they do not execute GPU benchmarks. In a working Python environment, install the rendering tools separately from the lightweight package:

```sh
python -m pip install matplotlib markdown reportlab pymupdf pillow pyyaml
python benchmarks/make_presentation_figures.py
python benchmarks/render_research_paper.py
python benchmarks/verify_evidence.py
```

The figure script checks source hashes, derives all displayed percentages from absolute medians and includes an arithmetic self-check. It writes three PNG/SVG pairs and `docs/assets/presentation_metrics.json`. Regenerate it after a referenced precision documentation change.

The PDF renderer reads `docs/research/paper.md`, uses the installed Matplotlib DejaVu fonts, resolves links to the public site, and checks all six result rows, scope text, references and page bounds. Its hashes and dependency versions are in [PDF_RENDERING.md](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/research/PDF_RENDERING.md). The PDF is independently rendered from Markdown; it is not compiled from the [standalone LaTeX source](https://github.com/Utkarsh-Sinha0/k-graphfuse/blob/main/research/k-graphfuse-paper.tex). ReportLab invariant mode removes changing PDF timestamps; use the recorded rendering versions for byte replication. MathText equations are images, so the web manuscript and LaTeX provide accessible/editable mathematical source.

The ordinary offline CI audit needs none of these rendering libraries. It checks figure inputs and conversions, structured percentage claims and saved manuscript/PDF hashes. The web math integration follows [Material for MkDocs' MathJax instructions](https://squidfunk.github.io/mkdocs-material/reference/math/), with a pinned MathJax 3.2.2 browser script.

## Publish a replication

Record the commit, actual library builds, device/driver, graph and input hashes, complete configuration search, accuracy gates, preparation scopes, all samples, rejected candidates, resource method and run status. Use repeated independent runs and randomized order for a stronger performance study. Do not replace old evidence with a new result; add a separate run identity.

## Audit theorem-dependent work and timing

```sh
python studies/bound_attribution_20261011/extract.py --verify
```

This offline audit decodes immutable runner sources, checks manifests and all matched FP64 medians/p95, and verifies equal-degree laptop controls and the untimed CPU boundary. It preserves all four execution settings for both changed-degree records. It does not create a 9→8 time or independent hardware repeat. [Full attribution report](../research/bound-attribution.md).
