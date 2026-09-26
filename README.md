# CrystalDLM

**Feedback Learning of Periodic Diffusion Language Models for Crystal Generation**

CrystalDLM uses lattice-conditioned periodic interactions to construct crystal geometry and learns reference-conditioned reconstruction and relative verification from physical feedback.

`Planner → Periodic DLM → Diffusion → Reconstruction & relative verification`

[Results](RESULTS.md) · [Reproduction guide](docs/reproduction.md)

## Models

| Paper name | Components |
| --- | --- |
| DLM | Base constructor + diffusion |
| Periodic DLM | Base constructor + periodic head + diffusion |
| CrystalDLM | Periodic DLM + physical-feedback reconstruction and selection |
| CrystalDLM (draft) | Periodic constructor output before diffusion |

## Install

Use Python 3.11 on Linux. The reference environment uses PyTorch 2.4.1 and CUDA 12.1.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install -r requirements.txt
```

## Quick start

The default is **MP-20 with 1000 fixed H1A2 Plans**. Planner training is optional.

Place `train.csv`, `val.csv`, and `test.csv` in `datasets/mp20/`. Set the dataset and frozen diffusion checkpoint paths in `configs/local.json`.

```bash
cp configs/mp20.json configs/local.json
# Set paths in configs/local.json.
export MP_API_KEY=...
bash scripts/reproduce.sh --config configs/local.json
```

The workflow prepares data, trains the constructor and periodic head, fits the reconstruction model and verifier, and evaluates generated outputs. It uses the configured frozen diffusion model.

```bash
# Inspect commands and settings.
bash scripts/reproduce.sh --config configs/local.json --dry-run
# Generate and evaluate with configured trained checkpoints.
bash scripts/reproduce.sh --config configs/local.json --stage inference
# Resume with prepared data.
bash scripts/reproduce.sh --config configs/local.json --resume --skip-prepare
```

## Stage commands

| Script | Action |
| --- | --- |
| `scripts/01_planner.sh` | Export the default Plan panel |
| `scripts/02_constructor.sh` | Train the base DLM constructor |
| `scripts/03_periodic.sh` | Fit the lattice-conditioned periodic head |
| `scripts/04_feedback.sh` | Fit reference-conditioned reconstruction and relative verification |

Run each with `bash`, adding `--config configs/local.json` and optionally `--skip-prepare`. Planner training uses `scripts/reproduce.sh --stage train-planner`; diffusion training uses `crystaldlm train diffusion` after data preparation.

## Datasets

| Dataset | Argument | Atoms per cell | Plans |
| --- | --- | --- | --- |
| MP-20 | `mp20` | 1–20 | Included H1A2 panel |
| Perov-5 | `perov-5` | 5 | User-supplied JSONL |
| MPTS-52 | `mpts-52` | 1–52 | User-supplied JSONL |

```bash
bash scripts/reproduce.sh --dataset perov-5 --plans /path/to/perov-plans.jsonl
bash scripts/reproduce.sh --dataset mpts-52 --plans /path/to/mpts-plans.jsonl
```

Use the matching dataset config and checkpoints. `--num-samples` changes the default 1000 requests; `--device` and `--set` configure execution.

## Outputs

Under `outputs/<dataset>/samples/`, `raw.jsonl` contains periodic drafts, `refined.jsonl` contains diffusion references, and `edited.jsonl` contains final CrystalDLM outputs. Direct and physical-quality summaries are stored separately in `direct/` and `evaluation/`.

[Reproduction details](docs/reproduction.md) include configuration, metric definitions and saved-record formats. See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream code, data and model terms.
