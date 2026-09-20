# Continuous diffusion refinement

This module follows [CrysLLMGen, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html), using its [released implementation](https://github.com/kdmsit/crysllmgen) and DiffCSP-based periodic network.

Training adds Gaussian noise to lattice matrices and wrapped Gaussian noise to fractional coordinates. The network learns lattice noise and the wrapped coordinate score. The objective is the sum of the two mean squared errors. Composition stays fixed.

Defaults follow the upstream MP-20 recipe: 1000 noise steps, 500 training epochs, batch 512, Adam LR `1e-3`, seed 1234, gradient-value clipping 1, and ReduceLROnPlateau factor 0.6/patience 30. Validation uses batch 32.

At inference, C1 provides initial coordinates and lattice parameters. The model performs 800 predictor/corrector refinement steps and returns a continuous crystal. Fixed graph topology and lattice-independent work are reused across steps. Each request retains its own random stream.

In the retained [C2 physical-feedback method](c2.md), F is frozen and supplies the continuous reference for conditional reconstruction. The reference and executed alternatives are physically evaluated; their outcomes train a conditional DLM and relative verifier. This gives continuous processing a role in reusable feedback learning as well as in inference. The diffusion output itself is not a physical preference label. [Complete story and teacher mechanism](../PHYSICAL_FEEDBACK_MASTER_STORY_ZH.md).

The separate registered draft-feedback recipe uses F and permitted relaxation to propose complete teachers or prefix-preserving completions. It quantizes, decodes and physically rechecks the exact token structures before supervising the draft DLM. That recipe does not use the post-F conditional editor. [Registered teacher workflow](../C1_C2_METHOD_ZH.md#teacher).

The reporting-data experiment compares F800 and F400 from the same saved raw drafts for both model versions, using `ordered_csr_v1`. In the vendored implementation `time_start = diff_steps`; a change from 800 to 400 therefore changes the diffusion starting index as well as the number of predictor/corrector steps. This is a budget/schedule comparison, not a claim of halving the same trajectory without loss. See [evaluation details](../registered-feedback.md#f400-and-f800-budget-comparison).

```bash
bash scripts/train.sh diffusion --config configs/local.json
bash scripts/sample.sh diffusion --config configs/local.json
bash scripts/query_hull.sh --config configs/local.json \
  --structures outputs/mp20/samples/refined.jsonl
```

Code: [trainer](../../src/dlm_iclr/diffusion/trainer.py), [refinement](../../src/dlm_iclr/diffusion/refinement.py), [vendored numerical model](../../src/dlm_iclr/_vendor/crysllmgen/models_ddpm/diffusion.py). The original copyright and MIT notices are retained in [third-party notices](../../THIRD_PARTY_NOTICES.md).
