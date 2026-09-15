# Data interfaces

Each dataset supplies training, validation and test sources. A source can be CSV with CIF text, JSONL with a pymatgen `structure` dictionary or `cif`, or a directory of CIF files.

```json
{
  "dataset": {
    "name": "my-crystals",
    "splits": {
      "train": "../datasets/my-crystals/train.jsonl",
      "val": "../datasets/my-crystals/val.jsonl",
      "test": "../datasets/my-crystals/test.jsonl"
    },
    "fields": {"id": "material_id", "structure": "structure"}
  }
}
```

`dlm prepare` produces continuous structures, B0 prompt/answer pairs, Planner prompt/answer pairs, Plan conditions and conversion diagnostics in `OUTPUT/data`. Split, source ID and atom permutation accompany each crystal. Structures sharing a composition remain separate examples.

Planner conditions follow H1A2 preprocessing of quantized source geometry. B0/C1 atom blocks are reordered together to match the Plan element order. Continuous structures remain available for diffusion training and novelty comparisons.

Custom field names belong in `dataset.fields`. Other modules use the prepared representation. Full Direct coverage accepts custom structure/composition cutoffs in `evaluation.coverage_cutoffs`.

## Model assets

| Key | Asset |
|---|---|
| `models.planner_base` | Meta-Llama-3-8B base weights |
| `models.planner` | H1A2 epoch-2 LoRA and tokenizer |
| `models.dlm` | LLaDA base checkpoint |
| `models.b0` | Crystal LoRA, trained vocabulary tables and tokenizer |
| `models.c1` | Periodic head with `head_config` and `state_dict` |
| `models.diffusion` | CrysLLMGen `model` state dictionary |
| `models.c2` | Editor LoRA, periodic state, edit modules and tokenizer |
| `models.value` | Value dimensions and `state_dict` |
| `models.risk` | Normalization statistics and logistic coefficients |
| `models.chgnet` | CHGNet checkpoint; empty selects packaged 0.3.0 weights |

Default `@run/` paths correspond to training outputs. Set an existing compatible location to reuse a trained module.
