# Running CrystalDLM

## Files and configuration

Start from `configs/mp20.json`, `configs/perov-5.json`, or `configs/mpts-52.json`. Paths are relative to the config; `@run/` refers to its output directory. Foundation-model identifiers use `hf:owner/model`.

| Input | Config key |
| --- | --- |
| TRAIN/VAL/TEST data | `dataset.splits` |
| Language backbone | `models.dlm` |
| Base constructor checkpoint | `models.constructor` |
| Periodic head | `models.periodic` |
| Frozen diffusion checkpoint | `models.diffusion` |
| Reconstruction model | `models.feedback` |
| Relative verifier and risk model | `models.verifier`, `models.risk` |

The full workflow trains the constructor, periodic head, reconstruction model and verifier. Inference uses the configured trained checkpoints. Supply a diffusion checkpoint for either mode.

CSV data uses `cif` and `material_id` columns. Structure JSONL and CIF directories are also supported; column mappings belong in `dataset.fields`.

## Commands

```bash
# Full workflow.
bash scripts/reproduce.sh --config configs/local.json
# Inspect the commands before execution.
bash scripts/reproduce.sh --config configs/local.json --dry-run
# Resume training with prepared data.
bash scripts/reproduce.sh --config configs/local.json --resume --skip-prepare
# Inference with trained checkpoints.
bash scripts/reproduce.sh --config configs/local.json --stage inference
```

For individual stages, use:

| Entry | Action |
| --- | --- |
| `bash scripts/01_planner.sh` | Export the default Plan set |
| `bash scripts/02_constructor.sh` | Train the base constructor |
| `bash scripts/03_periodic.sh` | Fit the periodic head |
| `bash scripts/04_feedback.sh` | Train reconstruction and relative verification |

Pass `--config configs/local.json` to each entry. Prepared data is reused when input and output hashes match. To rebuild it:

```bash
crystaldlm prepare --config configs/local.json --force
```

Planner and diffusion training are available separately:

```bash
bash scripts/reproduce.sh --config configs/local.json --stage train-planner
crystaldlm sample planner --config configs/local.json
crystaldlm train diffusion --config configs/local.json
```

Feedback fitting stages are `warmup`, `collect`, `label`, `compile`, `reconstruction`, `refit`, `verifier`, and `risk`:

```bash
crystaldlm train feedback --config configs/local.json --stage reconstruction --resume
```

## Feedback options

Both options default to `true`:

```json
{
  "feedback": {
    "protect_sun": true,
    "physical_rollback": true
  }
}
```

`protect_sun` retains references already confirmed S.U.N. and sends other editable references to reconstruction. The learned selector always includes the unchanged reference.

`physical_rollback` evaluates the selected reconstruction, then restores its paired reference when that reference was confirmed M.S.U.N. and the reconstruction is not confirmed S.U.N. or M.S.U.N. Unresolved reconstruction evaluations remain recorded. Direct and N/U-based metrics are recomputed on the final selected collection using the saved physical labels.

```bash
bash scripts/reproduce.sh --config configs/local.json --output outputs/custom \
  --set feedback.protect_sun=false --set feedback.physical_rollback=false
```

Changed policies or model settings use a new output directory. The resolved settings and per-request decisions are saved with the results.

## Data and Plans

MP-20 uses the included `mp20_default` Plan set. Other datasets require `--plans /path/to/plans.jsonl` and matching checkpoints.

| Dataset | Argument | Atoms per cell |
| --- | --- | --- |
| MP-20 | `mp20` | 1–20 |
| Perov-5 | `perov-5` | 5 |
| MPTS-52 | `mpts-52` | 1–52 |

Each Plan row has a unique `source_id` and a `plan_state`. Prompts and seeds may be supplied or derived deterministically. Example:

```json
{"source_id":"example:0","body_eligible":true,"plan_state":{"N":5,"elements":["Ca","Ti","O"],"counts":[1,1,3],"formula":"CaTiO3","reduced_formula":"CaTiO3","charge_bucket":"neutral_plausible","oxidation_candidates":"unknown","anion_framework":"oxide","lattice_system":"cubic","spacegroup_bucket":"sg_195_230","volume_per_atom_bin":"volpa_010_014","prototype_key":"example"}}
```

The default request count is 1000. `--num-samples` takes precedence over the configured count. Every selected request, including failures, remains in the evaluation denominator.

## Sampling and evaluation

```bash
# Base constructor drafts.
crystaldlm sample constructor --config configs/local.json --plans outputs/mp20/plans/evaluation.jsonl --output outputs/mp20/base-samples
# Periodic drafts and diffusion references.
crystaldlm sample periodic --config configs/local.json --plans outputs/mp20/plans/evaluation.jsonl --output outputs/mp20/samples
crystaldlm sample diffusion --config configs/local.json --output outputs/mp20/samples
# Evaluate an existing final collection.
crystaldlm evaluate direct --config configs/local.json --structures outputs/mp20/samples/final.jsonl --output outputs/mp20/direct-final --full
crystaldlm evaluate sun --config configs/local.json --structures outputs/mp20/samples/final.jsonl --output outputs/mp20/physical-final
```

The full workflow fetches missing hull references using `MP_API_KEY`. For standalone evaluation, prepare the hull with `crystaldlm hull --config configs/local.json --structures FILE.jsonl`.

Direct evaluates saved geometry before physical relaxation. Physical screening uses CHGNet 0.3.0 and joint position/cell FIRE relaxation, up to 1000 steps with force/stress tolerances 0.1 eV/Å and 0.5 GPa. S.U.N. and M.S.U.N. use hull thresholds 0 and 0.1 eV/atom. N/U compare saved structures with TRAIN references and earlier outputs, using StructureMatcher tolerances 0.2, 0.3 and 5 degrees. Unresolved evaluations are retained as unknown.

## Outputs

All paths below are relative to `outputs/<dataset>/samples/`.

| File or directory | Content |
| --- | --- |
| `final.jsonl`, `final.summary.json` | Final selected structures and metrics |
| `raw.jsonl`, `refined.jsonl` | Constructor drafts and diffusion references |
| `edited.jsonl` | Reconstructions before physical rollback |
| `rollback.jsonl` | Selected collection when physical rollback is enabled |
| `direct/`, `evaluation/` | Separate metrics for each saved collection |
| `evaluation/rollback/decisions.jsonl` | Per-request reference/reconstruction selection |
| `finalization.settings.json` | Selection policy and evaluation identities |
