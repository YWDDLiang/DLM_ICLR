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

With trained checkpoints configured, run inference only:

```bash
bash scripts/reproduce.sh --config configs/local.json --stage inference
```

[Usage](docs/reproduction.md) · [Paper results](RESULTS.md) · [Citation](CITATION.cff) · [Third-party notices](THIRD_PARTY_NOTICES.md)
