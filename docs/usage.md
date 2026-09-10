# Usage

## Saved and generated Plans

```bash
dlm-iclr plans R03_256 --output outputs/r03-plans.jsonl
dlm-iclr plans H1A2_1200 --legal-only --requests 1050 --output outputs/h1a2-1050.jsonl
dlm-iclr plans generate --config configs/local.json --requests 1200 --seed 17 \
  --output outputs/new-plans.jsonl
```

Selection is applied to the original Plan sequence before G runs. With `--legal-only`, the requested count is the number of legal Plans; otherwise the count includes failed Planner records. Requesting more available Plans produces a clear input error. It does not silently repeat or replace them.

Custom JSONL records contain `source_id`, `plan_state`, and optionally the canonical `body_prompt`, `body_noise_seed`, `refiner_noise_seed`, and provenance. Export a preset for a complete example. Seeds are exact nonnegative 63-bit integers; use a JSON reader that preserves them. If seeds are omitted for a custom file, they are derived deterministically from the configured seed and source identity.

Fresh generation uses the configured H1 rich Planner: temperature 0.9, top-p 0.95, top-k 50 and at most 96 new tokens. It preserves failed Planner outputs. In `infer --plan-source generate`, `inference.planner_requests` controls the available source pool, with a default of 1200; `--requests` controls the subsequent selected subset.

## Data preparation

`prepare-data` accepts MP-20 CSV, another CSV with `cif`, JSONL with `cif` or a pymatgen `structure` dictionary, and a directory of CIF files. The output preserves the dataset name, original split, source row and material identifier. Entries outside the fixed vocabulary/size representation are recorded in the conversion report.

```bash
dlm-iclr prepare-data --source /data/mp20/train.csv --dataset mp20 --split train \
  --output outputs/data/mp20-train.jsonl
dlm-iclr prepare-data --source /data/mp20/test.csv --dataset mp20 --split test \
  --output outputs/data/mp20-test.jsonl
dlm-iclr prepare-train --source outputs/data/mp20-train.jsonl \
  --exclude outputs/data/mp20-test.jsonl --output outputs/data/clean-train.jsonl
```

`prepare-train` deduplicates reduced compositions, excludes both bundled evaluation presets and any extra `--exclude` sources, and preserves evaluation/dev/test records as excluded sources. It never changes an original held-out source into TRAIN. For a new dataset, choose a name with `--dataset` and provide its original split explicitly.

`CLEAN_TRAIN_1000` selects the included 1000 composition-disjoint, synthetic Planner-generated TRAIN conditions and is the default training source. Its original split is `generated`; source row numbers identify generation requests. It can also be exported with `dlm-iclr plans CLEAN_TRAIN_1000 --output outputs/train-plans.jsonl`.

## Python API

```python
from dlm_iclr.config import load_config
from dlm_iclr.pipeline import run_inference, evaluate_output

config = load_config("configs/local.json")
config.inference.plan_source = "H1A2_1200"
config.inference.requests = 1050
config.inference.legal_only = True

structures = run_inference(config, "outputs/sample", devices=["cuda:0"])
scores, report = evaluate_output(config, "outputs/sample", devices=["cuda:0"])
```

The lower-level `Constructor`, `Refiner`, `Editor`, `ValueNetwork`, and `Labeler` APIs can be used independently. `Editor.edit` takes Plans and continuous/token input views; it does not accept physical results. `Labeler` and score computation are separate modules.

## Execution and restart

`scripts/run.sh` and `scripts/run.sbatch` invoke the same CLI. Set `PYTHON` to choose an interpreter. Pass ordinary `sbatch` options for your partition, GPUs, CPUs, memory and time. The code works without Slurm when a usable CUDA device is available.

G and F process individual structures. E batches independent requests. `--refiner-workers 4` runs four separate F processes per visible GPU while keeping each F sample at batch size one. On the tested A800 example, the matching-seed result remained bitwise identical. Concurrency is an execution option and should be sized to available CPUs and memory.

Completed inference records are written atomically and can be resumed. Use a different output directory when changing source data, model contents or settings. Prefer local model directories for reproducible resumption; remote model identifiers are resolved by the model library.

The self-improvement directory contains `train_plans.jsonl`, `evaluation_plans.jsonl`, `source_split.json`, the complete `S0`–`S3` output directories, and each round's TRAIN rollout, feedback, checkpoints and training records. It records all requested rounds without selecting checkpoints based on an intermediate metric. The current command can restart inference records and physical caches; a repeated training command performs a fresh update from its configured starting checkpoint.

Individual actor training supports `torchrun` through `TRAIN_GPUS`:

```bash
TRAIN_GPUS=4 bash scripts/train.sh E --config configs/local.json \
  --data outputs/feedback/E.jsonl --output outputs/editor
```

Actor batch sizes are per device. Set E's batch size to 4 to retain an effective source batch of 16 across four devices. Value-head training uses one device. Training reports include actual optimizer steps, source exposure and parameter changes.

## Outputs

`structures.jsonl` is the selected structure artifact used for evaluation. `G/`, `F/`, and `E/` preserve stage records, raw refiner matrices, patch bindings, candidate scores and actual DLM calls. `evaluation/scores.jsonl` holds the per-request metrics and `evaluation/summary.json` holds counts and percentages.

The physical cache is keyed by exact structure content, model, implementation, package versions and protocol. Full relaxation trajectories are compressed separately so large candidate banks do not keep every frame in memory. Unresolved physical or matching work is explicit and is not reported as a confirmed unstable structure.

If reference entries are missing, `counts` and `percent` are `null` for each affected metric. `known_counts`, `unknown_counts`, and `count_bounds` preserve what is established over the full requested denominator. `hull_missing` and `hull_error` identify affected records. See [results](results.md) for the frozen reference limitation and formal reporting rule.
