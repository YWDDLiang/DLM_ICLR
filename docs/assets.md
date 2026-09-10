# Model and data assets

Copy `configs/example.json` to `configs/local.json` and set paths for the assets you have. Paths beginning with `./` or `../` resolve relative to the configuration file. Environment variables and `~` are expanded for model/data asset paths.

| Configuration key | Required asset |
|---|---|
| `assets.base_model` | LLaDA-8B-Instruct model directory or Hugging Face identifier |
| `assets.generator` | B0 crystal checkpoint, or an updated G checkpoint |
| `assets.editor` | The trained editor checkpoint |
| `assets.value` | `bundled`, or a portable `autonomous_value.pt` |
| `assets.refiner` | The CrysLLMGen `model_494.pt` checkpoint |
| `assets.planner_base` | Llama base model, required for fresh Plan generation |
| `assets.planner` | H1A2 rich-Plan adapter, required for fresh Plan generation |
| `assets.chgnet` | CHGNet 0.3.0 checkpoint, required for physical evaluation |
| `assets.hull_cache` | `official_slim_cache.jsonl`, or its containing directory |
| `assets.novelty_reference` | MP-20 training CSV, or a converted JSONL reference set |
| `training.plans` | Composition-disjoint TRAIN Plan JSONL |

The base model is published by [LLaDA](https://github.com/ML-GSAI/LLaDA). The crystal B0 checkpoint extends its vocabulary and includes trained input/output tables; the unmodified language checkpoint cannot replace B0. A PEFT B0/G checkpoint includes `adapter_model.safetensors`, `adapter_config.json`, and tokenizer files. Both IO tables must be included in its `modules_to_save` payload.

An editor checkpoint adds `expert_edit_config.json`, `expert_edit_modules.pt`, `periodic_state_config.json`, and `periodic_state.pt` to that PEFT checkpoint. Loading requires the actual model tensors and their architectural configuration. It does not require an experiment directory, training receipt, approval marker or historical verification probe.

The repository currently distributes the **4,200,674-byte autonomous value head** and its manifest. It contains the learned hidden projection, normalization statistics and two-output readout; it does not contain the 8B language model or editor backbone. The large crystal, editor, Planner and refiner checkpoints are **external assets and are not included in this Git repository**. Configure their existing local paths to run the released workflow. No download link is claimed for an unpublished checkpoint.

The bundled value model is paired with the minibatch editor trained in the reported component experiment. Replacing the editor changes its feature space; train a matching value model with `dlm-iclr train value` after an editor update. The self-improvement command performs that step automatically.

Saved Plans are included at `src/dlm_iclr/data/plans`:

| Preset | Original requests | Legal Plans | SHA-256 |
|---|---:|---:|---|
| H1A2_1200 | 1200 | 1186 | `f049ab0860656423013f93b56e0a5d101f9473ad344f32cf27c25a8f2bd2c1a1` |
| R03_256 | 256 | 254 | `b18524df7c6168a6570eb61f9e77e26753495f1b465b0f2b4a425257e2fe07cd` |

These are actual saved requests, including failed Planner records, original prompts and full integer seeds. The `CLEAN_TRAIN_1000` alias selects bundled `data/training/clean_train_1000.jsonl`, containing synthetic Planner-generated TRAIN conditions, not MP-20 CSV row indices. Its original split is `generated` and its usage role is `train`. It excludes the evaluation and development compositions used to prepare this release. The file has SHA-256 `85e7b9bca96792a275d686c367a338e031a1a333fd97511b80885bc226832f52`.

Hull reference rows have the form `{"chemsys":"A-B", "entries":[{"composition":{"A":1}, "energy":-1.0, "entry_id":"..."}]}`. Energies are **total entry energies**, not per-atom energies. The retained reference protocol is the official Materials Project `GGA_GGA+U` entry set with `compatible_only=True`, without local compatibility reprocessing. Record the database version when constructing a cache and use one common cache for every compared snapshot. Missing references remain explicit unknowns in evaluation.
