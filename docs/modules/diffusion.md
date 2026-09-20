# Continuous diffusion refinement

This module follows [CrysLLMGen, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html), using its [released implementation](https://github.com/kdmsit/crysllmgen) and DiffCSP-based periodic network.

Training adds Gaussian noise to lattice matrices and wrapped Gaussian noise to fractional coordinates. The network learns lattice noise and the wrapped coordinate score. The objective is the sum of the two mean squared errors. Composition stays fixed.

Defaults follow the upstream MP-20 recipe: 1000 noise steps, 500 training epochs, batch 512, Adam LR `1e-3`, seed 1234, gradient-value clipping 1, and ReduceLROnPlateau factor 0.6/patience 30. Validation uses batch 32.

At inference, C1 provides initial coordinates and lattice parameters. The model performs 800 predictor/corrector refinement steps and returns a continuous crystal. Fixed graph topology and lattice-independent work are reused across steps. Each request retains its own random stream.

In the registered C2 feedback route, this continuous model is frozen and provides candidate lattices and coordinates for physical supervision. Permitted physical relaxation can provide additional terminal candidates. C2 registers equivalent complete teachers or proposes completions that preserve the student's actual visible prefix. Each resulting exact-token structure is decoded and physically checked before it supervises the discrete DLM. F's continuous score or its parent's physical label cannot simply be copied across quantization or a changed prefix. C2 does not edit the returned F structure. [Full teacher workflow and equations](../C1_C2_METHOD_ZH.md#teacher).

The reporting-data experiment compares F800 and F400 from the same saved raw drafts for both model versions, using `ordered_csr_v1`. In the vendored implementation `time_start = diff_steps`; a change from 800 to 400 therefore changes the diffusion starting index as well as the number of predictor/corrector steps. This is a budget/schedule comparison, not a claim of halving the same trajectory without loss. See [evaluation details](../registered-feedback.md#f400-and-f800-budget-comparison).

```bash
bash scripts/train.sh diffusion --config configs/local.json
bash scripts/sample.sh diffusion --config configs/local.json
bash scripts/query_hull.sh --config configs/local.json \
  --structures outputs/mp20/samples/refined.jsonl
```

Code: [trainer](../../src/dlm_iclr/diffusion/trainer.py), [refinement](../../src/dlm_iclr/diffusion/refinement.py), [vendored numerical model](../../src/dlm_iclr/_vendor/crysllmgen/models_ddpm/diffusion.py). The original copyright and MIT notices are retained in [third-party notices](../../THIRD_PARTY_NOTICES.md).
