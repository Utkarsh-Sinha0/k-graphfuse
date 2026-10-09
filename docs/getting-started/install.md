---
description: Install K GraphFuse for CPU use or optional CUDA execution and verify the environment with runnable examples.
---
# Install K GraphFuse

## Requirements

| Component | Requirement | Purpose |
|---|---|---|
| Python | 3.11 or newer | Package and reference tests |
| NumPy | 1.26 or newer | Channel coefficients and dense feature states |
| SciPy | 1.11 or newer | Sparse graph operators |
| PyTorch | 2.11 or newer, CUDA build | Optional GPU execution |
| NVIDIA device and driver | Compatible with your PyTorch CUDA build | Optional GPU execution |

CPU import does not import PyTorch. cuGraph, CuPy, Java and Graphalytics data are comparator requirements only; they are not required to use the library. Support is limited to the tested combinations described in the [support matrix](../reference/limitations.md).

## Install from the repository

```sh
git clone https://github.com/Utkarsh-Sinha0/k-graphfuse.git
cd k-graphfuse
python -m venv .venv
```

Activate the environment on Linux or macOS:

```sh
source .venv/bin/activate
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install and verify:

```sh
python -m pip install .
python examples/quickstart.py
python examples/sensitivity.py
python examples/resolvent.py
```

Expected quickstart result: retained degree `2`, output `[5.125 5.   ]`.

## Enable CUDA

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a build compatible with your operating system and driver. Then install the optional runtime and verify device availability:

```sh
python -m pip install ".[gpu]"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python examples/gpu.py
```

`True` confirms that PyTorch can access a CUDA device. A package installation alone does not confirm GPU support. Follow the [CUDA execution guide](../guides/gpu.md) before choosing FP32 or CUDA capture.

## Develop locally

```sh
python -m pip install -e ".[docs]"
python -m unittest discover -s tests -v
python -m mkdocs serve
```

The site preview defaults to `http://127.0.0.1:8000/`. Run `python -m mkdocs build --strict` for publication checks. CUDA tests report explicit skips on machines without CUDA; the CPU tests still run.
