# Reproduction

## Configuration

Configs are provided for MP-20, Perov-5 and MPTS-52. Paths resolve relative to the config file; `@run/` resolves within its output directory, and `hf:owner/model` identifies a Hugging Face asset. Supply dataset splits, the language backbone and a compatible frozen diffusion checkpoint. Inference also needs the trained constructor, periodic head, reconstruction model, verifier and risk model.

The public module names and checkpoint keys are:

| Component | Command | Config key |
| --- | --- | --- |
| Planner | `planner` | `planner` |
| Base DLM constructor | `constructor` | `constructor` |
| Periodic construction | `periodic` | `periodic` |
| Continuous diffusion | `diffusion` | `diffusion` |
| Physical feedback | `feedback` | `feedback`, `verifier`, `risk` |

Use `crystaldlm` for command-line execution. The Python package is `dlm_iclr`.

```bash
bash scripts/reproduce.sh --config configs/local.json --device cuda:0
bash scripts/reproduce.sh --config configs/local.json --set 'runtime.devices=["cuda:0","cuda:1"]'
```

Worker counts are set under `runtime`. Each stage finishes before the next. The launcher saves its resolved config and input identities for resumption. Data preparation reuses matching input/output hashes; `crystaldlm prepare --force` rebuilds it.

## Plans and data

MP-20 uses `preset:mp20_default`. Perov-5 and MPTS-52 use a supplied Plan JSONL. Each row has a unique `source_id` and a `plan_state`; saved prompts and body/refiner seeds retain their original values. Missing prompts and seeds are constructed deterministically.

```json
{"source_id":"example:0","body_eligible":true,"plan_state":{"N":5,"elements":["Ca","Ti","O"],"counts":[1,1,3],"formula":"CaTiO3","reduced_formula":"CaTiO3","charge_bucket":"neutral_plausible","oxidation_candidates":"unknown","anion_framework":"oxide","lattice_system":"cubic","spacegroup_bucket":"sg_195_230","volume_per_atom_bin":"volpa_010_014","prototype_key":"example"}}
```

Plan validation checks typed fields, atom ranges and composition counts. Eligible rows are selected in saved order. The default workflow processes 1000 fixed requests and retains failure records.

Crystal data can be CSV (`cif`, `material_id`), structure JSONL, or CIF directories. Column mappings and TRAIN/VAL/TEST paths are configured under `dataset`. Feedback learning uses prepared TRAIN Plans.

## Training

The full workflow trains the base constructor, fits the periodic head with the constructor frozen, then learns reconstruction and verification from evaluated modifications. Feedback stages are `warmup`, `collect`, `label`, `compile`, `reconstruction`, `refit`, `verifier`, and `risk`.

```bash
bash scripts/reproduce.sh --config configs/local.json
# Optional Planner training and fresh Plan sampling.
bash scripts/reproduce.sh --config configs/local.json --stage train-planner
crystaldlm sample planner --config configs/local.json
# Optional diffusion training from prepared data.
crystaldlm train diffusion --config configs/local.json
# Resume a particular feedback fit.
crystaldlm train feedback --stage reconstruction --resume --config configs/local.json
```

The base constructor uses composition prefill, the axis schedule and native coordinate/lattice masks. Periodic construction adds the learned lattice-conditioned interactions and construction policy. Diffusion uses 800 steps by default. Feedback fits a separate reference-conditioned reconstruction model and relative verifier.

## Sampling

```bash
# Base constructor drafts.
crystaldlm sample constructor --config configs/local.json --plans outputs/mp20/plans/evaluation.jsonl --output outputs/mp20/constructor-samples
# Periodic drafts and diffusion references.
crystaldlm sample periodic --config configs/local.json --plans outputs/mp20/plans/evaluation.jsonl --output outputs/mp20/samples
crystaldlm sample diffusion --config configs/local.json --output outputs/mp20/samples
# Full CrystalDLM inference and evaluation.
bash scripts/reproduce.sh --config configs/local.json --stage inference --skip-prepare
```

Feedback retains confirmed S.U.N. references and applies learned reconstruction/selection to other editable references (`feedback.protect_sun=true`). Selection includes the unchanged reference. Submitted geometry is saved before physical relaxation.

## Evaluation

Direct evaluates saved geometry: structural/compositional validity, coverage Precision/Recall, and Wasserstein-1 distances for Density and distinct element count. Coverage uses the TEST reference and `evaluation.coverage_cutoffs`; composition uses SMACT 3.1.0 with the mixed-valence supplement.

Physical screening uses CHGNet 0.3.0, joint position/cell FIRE relaxation (up to 1000 steps, force tolerance 0.1 eV/Å, stress tolerance 0.5 GPa), and a cached Materials Project competing-phase hull. Stability uses `E_hull <= 0`; metastability uses `E_hull <= 0.1 eV/atom` and includes stability. U compares submitted geometry with earlier saved outputs, and N compares with TRAIN structures of the same reduced composition. StructureMatcher tolerances are 0.2, 0.3 and 5 degrees. V.U.N., S.U.N. and M.S.U.N. combine these indicators.

```bash
crystaldlm evaluate direct --config configs/local.json --structures outputs/mp20/samples/raw.jsonl --output outputs/mp20/direct-draft --full
crystaldlm evaluate direct --config configs/local.json --structures outputs/mp20/samples/refined.jsonl --output outputs/mp20/direct-reference --full
crystaldlm hull --config configs/local.json --structures outputs/mp20/samples/refined.jsonl
crystaldlm evaluate sun --config configs/local.json --structures outputs/mp20/samples/edited.jsonl --output outputs/mp20/physical-final
```

The default evaluation limit is 1000 records; use `--num-samples` to change it. Summaries record the actual denominator and unknown counts. Fingerprint workers use a 60-second per-structure deadline, with unresolved coverage reported as bounds. Outputs are kept separately for drafts, diffusion references and final reconstructions.

## Checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/reproduce.py --dry-run
```
