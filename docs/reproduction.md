# Reproduction guide

## Assets and paths

Paths in a config are relative to that config file. `@run/` resolves under its output root; `hf:owner/model` selects a Hugging Face repository. `scripts/reproduce.py` writes a resolved config snapshot before execution and refuses to reuse an output root with different settings. No server username, local drive, private dataset path, or API key is built into the commands.

Required assets:

- Crystal TRAIN/VAL/TEST CSVs with CIFs (or structure JSONL/CIF directories supported by the data adapter).
- The LLaDA foundation model (`models.dlm`), downloaded according to its provider terms.
- A compatible frozen continuous diffusion checkpoint (`models.diffusion`). If training it locally, run `dlm prepare` and `dlm train diffusion`, then point `models.diffusion` to its produced checkpoint.
- For inference-only use, trained B0, C1, C2, relative value and risk checkpoints under the corresponding `models` keys. Full reproduction trains these from the dataset.
- Materials Project access for missing hull entries; use `MP_API_KEY` in the environment. Existing hull cache entries are reused. CHGNet loads its 0.3.0 model through package 0.4.2.

Use `--set runtime.generation_workers_per_device=1` on smaller GPUs. `--device cuda:0` uses one device; multiple visible devices can be configured with `--set 'runtime.devices=["cuda:0","cuda:1"]'`. No GPU reservation or background restoration service is installed.

## Fixed Plans

MP20 defaults to `preset:H1A2_1000`, the included fixed panel of 1000 Plans. Original request IDs, prompts and body/refiner seeds are retained in the data; its manifest provides file identities. All 1000 selected requests remain in the generation/evaluation denominator, including failures. A valid Plan does not guarantee chemical or structural validity of its generated output.

For other datasets, pass `--plans /path/to/plans.jsonl`. Each line must have a unique `source_id`, a `plan_state`, and preferably the original `body_noise_seed` and `refiner_noise_seed` (nonnegative 63-bit integers). Missing seeds are deterministically derived from source identity. An eligible rich `plan_state` includes:

```json
{"N":5,"elements":["Ca","Ti","O"],"counts":[1,1,3],"formula":"CaTiO3","reduced_formula":"CaTiO3","charge_bucket":"neutral_plausible","oxidation_candidates":"unknown","anion_framework":"oxide","lattice_system":"cubic","spacegroup_bucket":"sg_195_230","volume_per_atom_bin":"volpa_010_014","prototype_key":"example"}
```

The wrapper is `{"source_id":"example:0","plan_state":{...},"body_eligible":true}`. Body prompts can be omitted and will be constructed from the Plan; supplied prompts must match. Selection keeps the original order, requires at least the requested number of eligible Plans, and writes every exclusion reason. It never replaces failed structures after generation. Plans derived from TEST truth carry additional information and must be reported as such; do not label them de novo Planner outputs.

```bash
# No model load: export the default 1000 Plans.
dlm plans --config configs/mp20.json
# Explicit alternative Plan input, selected before generation.
dlm plans --config configs/perov-5.json --source /path/to/perov.jsonl
```

## Training and sampling

`reproduce.sh` runs B0 → C1 → C2 by default and then evaluates 1000 requests. `train.sh all` likewise excludes Planner and frozen diffusion training unless explicitly selected. To train Planner, use `reproduce.sh --stage train-planner`; this enables Planner tokenization during data preparation. To sample new Plans after that training:

```bash
dlm sample planner --config configs/local.json --num-samples 1000
```

The optional sampler makes a fixed number of requests and retains failures. New Plans are not automatically substituted into the fixed H1A2 evaluation panel.

Individual generation commands:

```bash
dlm sample b0 --config configs/local.json --plans outputs/mp20/plans/evaluation.jsonl --output outputs/mp20/b0-samples
dlm sample c1 --config configs/local.json --plans outputs/mp20/plans/evaluation.jsonl --output outputs/mp20/samples
dlm sample diffusion --config configs/local.json --output outputs/mp20/samples
```

B0 uses schema/element prefill, the original axis schedule, duplicate-coordinate and lattice-volume masks; it has no learned C1 head, C1 candidate sampler, construction monitor or recovery cascade. C1 uses the trained periodic head and its configured constructor. F800 performs 800 diffusion steps. These are different configurations; separate output roots prevent accidental reuse across B0 and C1.

Full pipeline with existing trained assets:

```bash
dlm run --config configs/local.json --plans outputs/mp20/plans/evaluation.jsonl
# Resume at a completed-input boundary using the same Plans/configuration.
dlm run --config configs/local.json --plans outputs/mp20/plans/evaluation.jsonl --from-stage diffusion
```

Per-stage settings and input Plan identities are checked before cached outputs are reused. Changing sampling parameters or checkpoints requires a new output root. Preserve TRAIN/VAL/TEST splits; C2's physical teacher uses TRAIN Plans, not the fixed evaluation panel.

## Evaluation protocol

Direct measures the saved output **before CHGNet relaxation**. Full mode includes composition/structural validity, fingerprint-based validity, coverage and distribution distances, with TEST as the coverage reference. SMACT 3.1.0 with the named mixed-valence supplement is the configured composition check. Coverage thresholds (structure/composition) are MP20 0.4/10, Perov-5 0.2/4, and the configured MPTS-52 setting 0.4/10. MPTS-52's thresholds are a project setting, not a claim of a universal benchmark standard.

SUN/MSUN use verified CHGNet terminal energies against the cached Materials Project hull (`S <= 0`, `MS <= 0.1 eV/atom`). N and U use the saved output geometry, with TRAIN for novelty. VUN uses composition-and-structure validity, novelty and ordered uniqueness. All selected requests remain in the denominator; unresolved predicates remain unknown and yield count bounds.

The default C2 configuration retains refined outputs already identified as SUN (`c2.protect_sun=true`). The editor can also retain its continuous reference when no eligible improvement is selected. These decisions are saved per request; the final edited endpoint therefore includes retained and edited structures. Setting `c2.protect_sun=false` defines a different experiment. Raw, F800 and final C2 results are evaluated and stored separately; do not attribute a refined-only result to C2.

```bash
dlm evaluate direct --config configs/local.json --structures outputs/mp20/samples/raw.jsonl --output outputs/mp20/direct-raw --full
dlm evaluate direct --config configs/local.json --structures outputs/mp20/samples/refined.jsonl --output outputs/mp20/direct-f800 --full
dlm hull --config configs/local.json --structures outputs/mp20/samples/refined.jsonl
dlm evaluate sun --config configs/local.json --structures outputs/mp20/samples/refined.jsonl --output outputs/mp20/sun-f800
```

Standalone evaluation also defaults to at most 1000 input records, preserving order; set `--num-samples` to change that count. The summary reports the actual denominator, never pads a shorter file. Full reference sets are used regardless of the prediction limit. Fingerprint workers have a 60-second per-structure deadline; resource failures remain unknown, and coverage is reported as bounds rather than an invented point estimate. Reference fingerprint failures stop the full Direct report instead of dropping reference rows.

## Checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/reproduce.py --dry-run
```

Tests cover Plan selection and identities, dataset capacity in fresh workers, CLI orchestration, resume isolation, failure denominators and numerical components. They do not replace a full GPU training run or establish that a newly trained model reproduces a particular historical score.
