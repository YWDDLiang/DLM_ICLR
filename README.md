# CrystalDLM

Reproduce crystal generation with a fixed composition Plan, a masked diffusion language model (B0), periodic construction (C1), frozen continuous refinement (F800), and physical-feedback editing (C2).

The default experiment is **MP20 with the included H1A2 1000-Plan panel**. Planner training is optional. Every method uses the same fixed Plan order, prompts and random seeds.

## Install

Use Linux, Python 3.11, and a CUDA GPU with enough memory for the 8B backbone. The reference package set targets PyTorch 2.4.1 / CUDA 12.1:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install -r requirements.txt
```

For another CUDA/PyTorch build, install matching PyTorch/PyG wheels. The Python launcher also works on Windows (`python -X utf8 scripts/reproduce.py ...`); GPU training is supported on Linux. Third-party data and weights retain their original licenses; see [notices](THIRD_PARTY_NOTICES.md).

## One command

Provide the dataset's `train.csv`, `val.csv`, and `test.csv` under `datasets/mp20/` (CIF column `cif`, identifier column `material_id`). Copy `configs/mp20.json` to `configs/local.json` and set `models.diffusion` to a compatible frozen diffusion checkpoint. Foundation model identifiers and local checkpoint paths are configurable. The foundation model and diffusion weights are **not bundled**.

```bash
cp configs/mp20.json configs/local.json
# Edit dataset/checkpoint paths in configs/local.json first.
export MP_API_KEY=...  # Needed when fetching missing reference hull entries.
bash scripts/reproduce.sh --config configs/local.json
```

This runs data preparation → fixed Plan selection → B0 training → C1 training → C2 training → generation/F800/C2 → Direct and SUN/MSUN/VUN evaluation. Each stage finishes before the next starts. Planner and diffusion training are not included by default. C2 training uses prepared TRAIN sources, separately from the fixed evaluation Plans.

```bash
# Inspect every resolved command and setting without loading models.
bash scripts/reproduce.sh --config configs/local.json --dry-run
# Use already trained B0/C1/C2 checkpoints specified in the config.
bash scripts/reproduce.sh --config configs/local.json --stage inference
# Reuse prepared data and resume compatible training/checkpoints.
bash scripts/reproduce.sh --config configs/local.json --resume --skip-prepare
```

## Module commands

All entry points share `--config`, `--dataset`, `--num-samples` (default 1000), `--output`, `--device`, and `--set SECTION.KEY=VALUE`.

| Entry | Action |
| --- | --- |
| `bash scripts/01_planner.sh` | Export the fixed H1A2 panel; no Planner training or model load |
| `bash scripts/02_b0.sh` | Prepare data and train B0 |
| `bash scripts/03_c1.sh` | Prepare data and train C1 using the configured B0 |
| `bash scripts/04_c2.sh` | Prepare data and run C2 warm-up, collection, physical supervision and fitting |
| `bash scripts/reproduce.sh --stage train-planner` | Explicitly prepare and train the optional Planner |
| `bash scripts/train.sh diffusion --config configs/local.json` | Optionally train diffusion from prepared data |

Pass the same `--config configs/local.json` to module commands. With prepared data, add `--skip-prepare`. A module's prerequisite checkpoints must exist, or be produced by an earlier stage. Details and individual sampling/evaluation commands are in the [reproduction guide](docs/reproduction.md).

## Other datasets

| Dataset argument | Atom range | Default evaluation Plan source |
| --- | --- | --- |
| `mp20` | 1–20 | Included H1A2 1000-Plan panel |
| `perov-5` | Exactly 5 | User-supplied Plan JSONL |
| `mpts-52` | 1–52 | User-supplied Plan JSONL |

```bash
bash scripts/reproduce.sh --dataset perov-5 --plans /absolute/path/perov-plans.jsonl
bash scripts/reproduce.sh --dataset mpts-52 --plans /absolute/path/mpts-plans.jsonl
```

Set each dataset's paths in its matching config (or pass `--config`). Use models trained with that dataset's vocabulary/capacity. The loader validates Plan atom counts, the body canvas is `7 + 4N`, and Planner prompts use the dataset's name and atom range. Changing the dataset does not make an MP20-trained checkpoint a trained Perov-5/MPTS-52 checkpoint.

## Outputs

Runs write to `outputs/<dataset>/` by default. `plans/evaluation.manifest.json` records selection and hashes. `samples/raw.jsonl`, `refined.jsonl`, and `edited.jsonl` retain every selected request, including failures. Direct reports are under `samples/direct/`; SUN/MSUN/VUN reports are under `samples/evaluation/`, with separate raw, refined and edited endpoints. Unknown evaluations remain explicit.

See the [evaluation protocol](docs/reproduction.md#evaluation-protocol) for endpoint definitions and selection rules.
