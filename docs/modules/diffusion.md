# Continuous diffusion refinement

This module follows [CrysLLMGen, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html), using its [released implementation](https://github.com/kdmsit/crysllmgen) and DiffCSP-based periodic network.

Training adds Gaussian noise to lattice matrices and wrapped Gaussian noise to fractional coordinates. The network learns lattice noise and the wrapped coordinate score. The objective is the sum of the two mean squared errors. Composition stays fixed.

Defaults follow the upstream MP-20 recipe: 1000 noise steps, 500 training epochs, batch 512, Adam LR `1e-3`, seed 1234, gradient-value clipping 1, and ReduceLROnPlateau factor 0.6/patience 30. Validation uses batch 32.

At inference, C1 provides initial coordinates and lattice parameters. The model performs 800 predictor/corrector refinement steps and returns a continuous crystal. Fixed graph topology and lattice-independent work are reused across steps. Each request retains its own random stream.

```bash
bash scripts/train.sh diffusion --config configs/local.json
bash scripts/sample.sh diffusion --config configs/local.json
bash scripts/query_hull.sh --config configs/local.json \
  --structures outputs/mp20/samples/refined.jsonl
```

Code: [trainer](../../src/dlm_iclr/diffusion/trainer.py), [refinement](../../src/dlm_iclr/diffusion/refinement.py), [vendored numerical model](../../src/dlm_iclr/_vendor/crysllmgen/models_ddpm/diffusion.py). The original copyright and MIT notices are retained in [third-party notices](../../THIRD_PARTY_NOTICES.md).
