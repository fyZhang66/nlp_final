# nlp-final: End-to-End ABSA with ATE + ASC (SemEval 2014)

This project implements **aspect term extraction (ATE)** and **aspect-level sentiment classification (ASC)** for review text. The `pipeline` module chains them into an **ATE → ASC** end-to-end workflow with **in-domain / cross-domain** experiments and error analysis.

---

## Requirements

- **Python**: `>=3.10` (see `requires-python` in `pyproject.toml`).
- **GPU (recommended)**: Training and inference expect a PyTorch CUDA build. This repo resolves `torch` from the official **CUDA 12.8 (cu128)** index (Windows / Linux), which works well on recent NVIDIA GPUs.
- **Packaging**: **[uv](https://docs.astral.sh/uv/)** is recommended to create the virtual environment and install dependencies.

---

## Virtual environment and dependencies with uv

From the project root:

```bash
# Create a virtual environment (example: Python 3.11)
uv venv --python 3.11

# Install dependencies (matches uv.lock)
uv sync
```

**Activate the environment**

- **Windows (PowerShell)**

  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- **Linux / macOS**

  ```bash
  source .venv/bin/activate
  ```

Run all `python -m ...` commands with the **environment activated** and the **working directory set to the project root**.

### PyTorch and uv

- Dependencies are declared under `[project.dependencies]` in `pyproject.toml`.
- **`torch` is resolved from PyTorch’s cu128 index** (see `[tool.uv.index]` and `[tool.uv.sources]`) so Windows does not accidentally install the CPU-only wheel from PyPI.
- To refresh the lockfile after editing `pyproject.toml`: run `uv lock`, then `uv sync`.

---

## Data layout and paths

- **Raw XML** (SemEval 2014 Task 4): under `data/`, referenced via `pipeline/semeval_data.py` and `RAW_DATA` in `pipeline/config.py`.
- **Hugging Face datasets for training** (from preprocessing scripts): e.g. `ate/ate_data_*`, `asc/asc_data_*`.
- **Trained models**: `ate/ate_output_{domain}_{model}/final`, `asc/asc_output_{domain}_{model}/final` (aligned with `pipeline.config.MODELS`).
- **Pipeline metrics / artifacts**: default `pipeline/outputs/` (`pipeline.config.OUTPUT_DIR`).
- **Figures**: default `pipeline/figures/` (`pipeline.config.FIGURES_DIR`).

`.gitignore` excludes large files under `ate_output_*` / `asc_output_*` except `test_results.json`. Whether to commit `pipeline/outputs` and `pipeline/figures` is up to your team.

---

## End-to-end workflow (recommended order)

All commands assume the **project root** as the current working directory.

### 1. Data preparation + training (ATE and ASC, 2 domains × 2 backbones)

One shot: preprocess, then train all combinations.

```bash
python -m pipeline.train_all
```

Common variants:

```bash
# Only build HF datasets (no training)
python -m pipeline.train_all --prepare-only

# Only train (datasets must already exist)
python -m pipeline.train_all --train-only

# Restrict domain and/or backbone (example: laptop + BERT)
python -m pipeline.train_all --domain laptop --model_name bert
```

Notes: `--domain` / `--model_name` can be `all` (default) or `restaurant` / `laptop`, `bert` / `deberta`.  
In the default **prepare + train** run, **preparation still runs for both restaurant and laptop**; `--domain` / `--model_name` only filter **training**. For preparation only, use `--prepare-only`. To train a single domain without re-running preparation, run `--prepare-only` first, then `--train-only --domain ...`.

### 2. Cross-domain matrix (8 runs) + error analysis + summary tables + figures

After **ATE/ASC training** is done (the `final` dirs referenced by `MODELS` exist):

```bash
python -m pipeline.run_cross_domain
```

Options:

```bash
# Custom pipeline output directory (default matches OUTPUT_DIR in config)
python -m pipeline.run_cross_domain --output_dir pipeline/outputs

# Run experiments only; skip matplotlib figures (faster)
python -m pipeline.run_cross_domain --skip_figures
```

Typical outputs:

- Per-experiment subfolders: `ate_predictions_*.jsonl`, `e2e_predictions_*.jsonl`, `metrics_*.json`, confusion-matrix CSVs, error-analysis JSON/JSONL, etc.
- Aggregate summary: `pipeline/outputs/cross_domain_summary.json`.
- Unless `--skip_figures` is set: PNGs under **`pipeline/figures/`** (confusion matrices, ATE error breakdown, end-to-end attribution) plus summary comparison plots.

### 3. Regenerate figures only (no model inference)

If `pipeline/outputs` already contains the CSV/JSON inputs:

```bash
python -m pipeline.plot_figures
```

Optional: `--outputs_dir`, `--figures_dir` (defaults in `pipeline.config`).

---

## Standalone modules (debugging / custom runs)

### Single end-to-end pass (one ATE/ASC pair and one test set)

```bash
python -m pipeline.run_pipeline ^
  --ate_model_dir ate/ate_output_restaurant_bert/final ^
  --asc_model_dir asc/asc_output_restaurant_bert/final ^
  --test_xml      data/Restaurants_Test_Gold.xml ^
  --domain        restaurant ^
  --model_name    bert ^
  --output_dir    pipeline/outputs
```

(On Linux / macOS, replace line continuation `^` with `\`.)

### Error analysis only (when `e2e_predictions_*.jsonl` already exists)

```bash
python -m pipeline.error_analysis ^
  --e2e_predictions pipeline/outputs/e2e_predictions_restaurant_bert.jsonl ^
  --test_xml        data/Restaurants_Test_Gold.xml ^
  --domain          restaurant ^
  --model_name      bert ^
  --output_dir      pipeline/outputs
```

---

## Cross-domain experiment matrix

`EXPERIMENTS` in `pipeline/run_cross_domain.py` defines **8** settings: **train domain × test domain × {bert, deberta}**, covering in-domain baselines and cross-domain pairs. If data or model directories are missing, that run is skipped and logged.

---

## Configuration

| What | Where |
|------|--------|
| Data paths, model `final` paths, output dirs | `pipeline/config.py` (`RAW_DATA`, `MODELS`, `OUTPUT_DIR`, `FIGURES_DIR`) |
| Backbones and training hyperparameters | `BACKBONE_PRESETS` in `pipeline/config.py` |
| Dependencies and uv / PyTorch index | `pyproject.toml` |

---

## Troubleshooting

- **`torch.cuda.is_available()` is False (Windows)**  
  Ensure dependencies were installed with `uv sync` and that `torch` comes from the cu128 index (see `pyproject.toml`). Install an NVIDIA driver compatible with the CUDA 12.8 runtime used by the wheels.

- **Cross-domain runs are skipped**  
  Check that `ate/.../final` and `asc/.../final` exist for each entry in `pipeline.config.MODELS`, and that test XML files under `data/` are present.

---

## Course notice

Course project. Data and baseline model usage follow SemEval and the respective Hugging Face model cards.
