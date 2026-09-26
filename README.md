# CrystalDLM

Reproduction repository for **CrystalDLM: Feedback Learning of Periodic Diffusion Language Models for Crystal Generation**.

## 1. Install

Python 3.11, Linux, CUDA 12.1.

Run from the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install -r requirements.txt
```

## 2. Prepare files

| Input | Default location |
| --- | --- |
| Dataset splits, with `cif` and `material_id` columns | `datasets/mp20/`: `train.csv`, `val.csv`, `test.csv` |
| Frozen diffusion checkpoint | `checkpoints/mp20/diffusion.pt` |

```bash
cp configs/mp20.json configs/local.json
export MP_API_KEY="YOUR_MATERIALS_PROJECT_API_KEY"
```

Edit paths in `configs/local.json` if your files are stored elsewhere. The default Plan set is included; Planner training is optional.

## 3. Run

```bash
bash scripts/reproduce.sh --config configs/local.json --device cuda:0
```

This prepares data, trains the constructor and feedback models, generates structures, and evaluates the outputs.

With trained checkpoints configured, run inference only:

```bash
bash scripts/reproduce.sh --config configs/local.json --stage inference
```

## Options

| Setting | Default | Purpose |
| --- | --- | --- |
| `--num-samples` | `1000` | Number of generation requests |
| `--device` | `cuda:0` | Execution device |

Perov-5 and MPTS-52 use their dataset configs and a supplied Plan file:

```bash
bash scripts/reproduce.sh --dataset perov-5 --plans /path/to/perov-plans.jsonl
bash scripts/reproduce.sh --dataset mpts-52 --plans /path/to/mpts-plans.jsonl
```

[Detailed commands](docs/reproduction.md) · [Paper results](RESULTS.md) · [Citation](CITATION.cff) · [Third-party notices](THIRD_PARTY_NOTICES.md)
