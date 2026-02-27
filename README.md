````md
# MM-PHATE Reproducibility

This repository contains code to reproduce the **main figures** from the paper:

**Multiway Multislice PHATE: Visualizing Hidden Dynamics of RNNs through Training**

The code supports two reproducibility modes:

- **Cache mode (fast):** regenerate figures from cached artifacts (no training).
- **Full mode (slow):** train models + extract activations + run embeddings + generate figures from scratch.

---

## Repository layout (high level)

- `src/mmphate_repro/` : reusable library code (all logic lives here; avoid duplication)
- `configs/`           : manifests/configs mapping paper figures to pipelines
- `experiments/`       : thin entry scripts (optional; call into `src/`)
- `tests/`             : pytest tests (run via tox)
- `data/`              : datasets (not committed)
- `results_cache/`     : cached artifacts for fast reproduction (not committed unless small)
- `figures/`           : generated figures (not committed)

---

## Installation

### Windows (PowerShell)

Create a virtual environment:

```powershell
py -3.10 -m venv .venv
````

Activate the environment:

* PowerShell (may require adjusting execution policy):

```powershell
.\.venv\Scripts\Activate.ps1
```

* If PowerShell activation is blocked, use CMD instead:

```bat
.venv\Scripts\activate.bat
```

Install dependencies and the package:

```powershell
pip install -r requirements\base.txt
pip install -r requirements\dev.txt
pip install -e .
```

### macOS / Linux (bash)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
pip install -e .
```

### Sanity check (all platforms)

```bash
python -c "import mmphate_repro; print(mmphate_repro.__version__)"
mmphate-repro -h
```

---

## Reproduce figures

### Cache mode (fast)

Cache mode regenerates the paper figures from precomputed embeddings/metrics stored under `results_cache/`.

```bash
mmphate-repro reproduce --mode cache
```

### Full recomputation (slow)

Full mode trains models and regenerates all embeddings and figures from scratch.

```bash
mmphate-repro reproduce --mode full
```

---

## Testing

Run tests in a clean environment using tox:

```bash
tox
```

---

## Determinism / reproducibility notes

We set random seeds in training and embedding scripts where possible. Some operations (especially GPU kernels and t-SNE) may remain nondeterministic across platforms or library versions. For the most faithful reproduction:

* use the pinned dependency versions listed in `requirements/`
* prefer running on the same OS/hardware when exact numerical equality is required
* expect small numerical differences across platforms to be possible

```
```
