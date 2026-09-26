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

Edit paths in `configs/local.json` if your files are stored elsewhere.

## 3. Run

### Train

#### Planner (optional)

Plans are provided, so Planner training can be skipped.

```bash
bash scripts/01_planner.sh --config configs/local.json
```

#### DLM base

Train the base model for Plan-conditioned crystal generation.

```bash
bash scripts/02_constructor.sh --config configs/local.json
```

#### Periodic DLM

Train periodic interactions for crystal construction.

```bash
bash scripts/03_periodic.sh --config configs/local.json
```

#### Diffusion

Train a refiner if needed, then set `models.diffusion` to its checkpoint. Skip training when using a prepared checkpoint.

```bash
bash scripts/04_diffusion.sh --config configs/local.json
```

#### Physical Feedback

Generate feedback data with the provided collection code, then train reconstruction and selection. Collection requires the TRAIN split and trained constructor, periodic and diffusion models; see [required inputs](docs/feedback_data.md).

```bash
# Collect structures and physical labels.
bash scripts/05_collect_feedback.sh --config configs/local.json
# Train from the collected data.
bash scripts/06_train_feedback.sh --config configs/local.json
```

### Inference

With trained checkpoints configured, generate structures using the supplied Plans or a custom file via `--plans`:

```bash
bash scripts/inference.sh --config configs/local.json
```

To generate new Plans with a trained Planner (optional):

```bash
bash scripts/sample_plans.sh --config configs/local.json
```

## 4. Evaluation

The scripts evaluate final outputs by default. Use `--structures FILE` to evaluate another saved collection.

### Direct

Check structural and compositional validity, coverage, and property distributions.

```bash
bash scripts/07_evaluate_direct.sh --config configs/local.json
```

### SUN

Evaluate stability, uniqueness and novelty, reporting S.U.N., M.S.U.N. and V.U.N.

```bash
bash scripts/08_evaluate_sun.sh --config configs/local.json
```

[Usage](docs/reproduction.md) · [Paper results](RESULTS.md) · [Citation](CITATION.cff) · [Third-party notices](THIRD_PARTY_NOTICES.md)
