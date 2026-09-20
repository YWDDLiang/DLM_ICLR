# CrystalDLM / DLM-ICLR

**Feedback learning for periodic diffusion language models in crystal generation.**

DLM-ICLR combines a material-condition Planner, a crystal-adapted LLaDA model, the C1 periodic relation layer, and frozen continuous refinement. The current C2 route organizes physically checked geometric teachers and supervision at real construction prefixes, then learns that experience back into the draft generator. C2 does not edit diffusion outputs in this route.

```mermaid
flowchart LR
    P[Original Plan / prompt] --> C1[DLM + C1 draft]
    C1 --> F[Frozen continuous diffusion]
    F --> PH[Quantize / decode / physical recheck]
    PH --> C2[C2: registered teachers and prefix feedback]
    C1 --> C2
    C2 --> L[Learn back into DLM]
    L --> C1
    C1 --> EV[Raw and refined evaluation]
```

The current validated recipe preserves the original prompt, registers physically equivalent teacher coordinates to the draft representation, rechecks exact tokens, and mixes complete-teacher fitting with verified corrections at actual visible prefixes. Deployment uses the frozen model and sampling configuration without teacher retrieval or a physical oracle. See [the feedback method and artifact contract](docs/registered-feedback.md).

**Evidence scope:** the current result is a controlled study on 16 selected, seen TRAIN conditions with four fresh sampling streams. At the same body temperature 0.2, the unrelaxed CHGNet stability proxy increased from 7/64 to 35/64. It is not a claim about unseen MP-20 performance, DFT validation, learned online verifier gains, or shorter refinement. [Results and tradeoffs](docs/results/registered-feedback-train.md) include conventional SUN/MSUN, unknowns, and the F800 comparison.

## Quick start

Use Python 3.10–3.12 and install PyTorch/PyG for your CUDA version; see [installation](docs/installation.md).

```bash
python -m pip install -e '.[train,refine,physics,direct]'
dlm config --config configs/mp20.json --output configs/local.json
```

Set dataset and model locations in `configs/local.json`. Local directories and Hugging Face identifiers (`hf:owner/model`) are supported. Paths beginning with `@run/` refer to the configured output directory.

```bash
# Prepare all module datasets together
bash scripts/prepare.sh --config configs/local.json

# Train the complete stack, or pass one module name
bash scripts/train.sh all --config configs/local.json

# Learn from reviewed, exact-token physical supervision artifacts
python -m dlm_iclr.feedback train \
  --config configs/local.json --assets outputs/initial_assets.json \
  --recipe configs/registered_feedback.json \
  --teachers outputs/material/teachers.jsonl \
  --prefix-feedback outputs/material/verified_prefix_supervision.jsonl \
  --material-review outputs/material/ROOT_REVIEW.json \
  --budget-ledger outputs/budget.json --budget-limits configs/feedback_budget.example.json \
  --output outputs/feedback_student

# Evaluate the frozen model on a Plan file fixed before outcomes are observed
python -m dlm_iclr.feedback evaluate \
  --config configs/local.json --assets outputs/feedback_student/assets.json \
  --recipe configs/registered_feedback.json --plans outputs/fixed_plans.jsonl \
  --budget-ledger outputs/budget.json --budget-limits configs/feedback_budget.example.json \
  --panel-scope seen_train --reference-split val --output outputs/feedback_evaluation
```

The material contract and preparation APIs are described in the [feedback guide](docs/registered-feedback.md); the commands do not invent missing physical labels. Set `MP_API_KEY` in the environment when uncached physical references are needed. On Windows, enable UTF-8 mode (`PYTHONUTF8=1`) for the third-party chemistry data files. The shared budget ledger must be reused across preparation, training, and evaluation.

| Module | Learns | Main artifact | Documentation |
|---|---|---|---|
| Planner | Seven-line material conditions with Llama 3 8B | Two-stage LoRA checkpoint | [Planner](docs/modules/planner.md) |
| B0 | Compact crystal vocabulary with masked denoising | LoRA and trained vocabulary tables | [B0](docs/modules/b0.md) |
| C1 | Periodic coordinate compatibility from DLM hidden states | Periodic probability head | [C1](docs/modules/c1.md) |
| Diffusion | Continuous lattice and coordinate denoising | CrysLLMGen diffusion model | [Diffusion](docs/modules/diffusion.md) |
| C2 feedback | Physically verified teachers and actual-prefix supervision | Reviewed material and updated draft adapter | [Feedback](docs/registered-feedback.md) |
| Evaluation | Geometry, stability, novelty and uniqueness | Per-structure and aggregate metrics | [Evaluation](docs/evaluation.md) |

## Change the dataset

CSV, structure JSONL and CIF directories use one adapter. Set split paths and field mappings; the adapter preserves atom blocks, polymorphs and supplied train/validation/test splits.

```bash
dlm config --config configs/custom.json --output configs/my-crystals.json
bash scripts/prepare.sh --config configs/my-crystals.json
bash scripts/train.sh all --config configs/my-crystals.json
```

The default vocabulary represents 1–20 atoms, atomic numbers 1–94, fractional coordinates at 0.01 resolution, lengths at 0.1 Å and angles at 1°. [Data interfaces](docs/data.md) describe source fields, prepared records and model assets.

## Retained baseline and legacy stages

The earlier post-F editor remains available for historical reproduction. It is not the registered-feedback method or the source of the new TRAIN result. The active feedback entry point is `python -m dlm_iclr.feedback`; older `dlm run` and `sample c2` commands below retain their legacy behavior.

```bash
bash scripts/train.sh c2 --stage warmup --config configs/local.json
bash scripts/sample.sh c1 --plans H1A2_1050 --config configs/local.json
bash scripts/sample.sh diffusion --config configs/local.json
bash scripts/query_hull.sh --config configs/local.json \
  --structures outputs/mp20/samples/refined.jsonl
bash scripts/evaluate.sh sun --config configs/local.json \
  --structures outputs/mp20/samples/refined.jsonl \
  --output outputs/mp20/samples/evaluation/refined
bash scripts/sample.sh c2 --config configs/local.json
bash scripts/evaluate.sh direct --config configs/local.json \
  --structures outputs/mp20/samples/edited.jsonl \
  --output outputs/mp20/samples/evaluation/direct
```

`dlm run` connects these stages. Use `--from-stage` and `--to-stage` for a segment. Numerical modules are also ordinary Python APIs.

## Configure compute

Independent requests run in separate workers. C2 batches model queries, F reuses fixed graph topology, and physics and structural matching share caches.

```bash
bash scripts/run.sh --config configs/local.json --plans H1A2_1050 \
  --set 'runtime.devices=["cuda:0","cuda:1"]' \
  --set runtime.refine_workers_per_device=8 \
  --set runtime.physics_workers_per_device=8

torchrun --standalone --nproc_per_node=2 -m dlm_iclr train b0 \
  --config configs/local.json
```

B0 preserves effective batch 16 across single- and dual-process training. Checkpoints retain best and last states; completed stages can be reused with `--resume`. [Configuration](docs/configuration.md) lists settings and outputs.

## Reproducibility

The [SUN / MSUN / VUN result table](docs/results/C2_UNIFIED/RESULT_ZH.md) compares F800 and two C2 variants on the same 1,005 requests with jointly known labels from the historical 1,050-request panel. It includes Stable / MetaStable, validity, uniqueness, novelty, case studies and downloadable per-request records. These are physical-label-informed ablations; the report specifies the selection rules and denominator.

Defaults follow the retained C1 and one-revision C2 execution, including the fitted risk penalty. Saved Plan presets preserve source order and random seeds. The [reference profile](docs/reference.md) and [release validation](docs/validation.md) record initialization, checkpoint roles and optimization settings. Foundation and task checkpoints are supplied as local assets or produced by the training commands.

Detailed Chinese notes cover each module's scientific task, training and inference, mathematical objectives, implementation and paper foundations: [read the technical notes](private/README_ZH.md) or [download the complete notes](private/technical_notes_zh.zip).

## Attribution

Continuous refinement follows [CrysLLMGen](https://github.com/kdmsit/crysllmgen), **NeurIPS 2025**, and its DiffCSP components. The language backbone is [LLaDA](https://github.com/ML-GSAI/LLaDA). C1 adapts the tractable structured-output idea studied by [CoDD](https://arxiv.org/abs/2603.00045). The retained legacy editor draws on remasking and reward-weighted proposal methods; those components are separate from the registered-teacher recipe evaluated here. See [references](docs/references.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

Original project code uses the [MIT license](LICENSE). Model and dataset terms follow their providers.
