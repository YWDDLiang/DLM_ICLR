# Usage

Install the environment and prepare the inputs as described in the [README](../README.md).

## Configuration

Copy the matching dataset config and update its data and checkpoint paths. Paths resolve relative to the config file; `@run/` refers to the output directory.

| Dataset | Config | Plans |
| --- | --- | --- |
| MP-20 | `configs/mp20.json` | Included |
| Perov-5 | `configs/perov-5.json` | User supplied |
| MPTS-52 | `configs/mpts-52.json` | User supplied |

For inference, set the trained checkpoints under `models`: `constructor`, `periodic`, `diffusion`, `feedback`, `verifier`, and `risk`. Parameter defaults are in [defaults.json](../src/dlm_iclr/defaults.json).

Plan files use JSONL records with `source_id` and `plan_state`; use the [included Plan file](../src/dlm_iclr/data/presets/plans/mp20_default.jsonl) as the format reference.

## Run

```bash
# Full workflow.
bash scripts/reproduce.sh --config configs/local.json
# Inference with trained checkpoints.
bash scripts/reproduce.sh --config configs/local.json --stage inference
# Another dataset.
bash scripts/reproduce.sh --dataset perov-5 --plans /path/to/perov-plans.jsonl
bash scripts/reproduce.sh --dataset mpts-52 --plans /path/to/mpts-plans.jsonl
```

Use `--num-samples` to set the request count and `--device` to select the GPU. Use a new `--output` directory when changing the configuration.

## Individual stages

| Script | Action |
| --- | --- |
| `scripts/01_planner.sh` | Load the Plan set |
| `scripts/02_constructor.sh` | Train the constructor |
| `scripts/03_periodic.sh` | Train the periodic head |
| `scripts/04_feedback.sh` | Train reconstruction and verification |

Run each script with `bash` and `--config configs/local.json`. Optional Planner and diffusion training:

```bash
bash scripts/reproduce.sh --config configs/local.json --stage train-planner
crystaldlm train diffusion --config configs/local.json
```

For available arguments, run `bash scripts/reproduce.sh --help` or `crystaldlm --help`.

## Evaluation

The workflow reports Direct and S.U.N./M.S.U.N./V.U.N. results. Failed and unresolved requests remain in the denominator; unresolved rates are reported as intervals. The evaluation settings and per-request records are saved with each run.

See [paper results and metric definitions](../RESULTS.md).
