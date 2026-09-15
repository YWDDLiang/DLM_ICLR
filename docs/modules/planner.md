# Planner

The H1A2 planner adapts **Meta-Llama-3-8B base** using LoRA. Its supervised output is a seven-line material Plan: formula, anion framework, charge bucket, lattice system, space-group bucket, volume-per-atom bin and end marker. This compact condition drives subsequent crystal generation.

Training minimizes answer-token causal cross-entropy. Stage 1 begins with a fresh adapter; stage 2 loads stage 1's final adapter and starts a new optimizer and cosine schedule. Both stages visit the training split once. Defaults are effective batch 8, maximum length 768, LR `2e-5`, warm-up 100, seed 17, LoRA rank 16/alpha 32/dropout 0.05 on attention and feed-forward projections.

Sampling uses temperature 0.9, top-p 0.95, top-k 50 and at most 96 new tokens. Parsed conditions and request seeds are saved before any structure generation.

```bash
bash scripts/train.sh planner --config configs/local.json
bash scripts/sample.sh planner --config configs/local.json
```

Code: [trainer](../../src/dlm_iclr/planner/trainer.py), [two-stage workflow](../../src/dlm_iclr/planner/workflow.py), [sampling](../../src/dlm_iclr/planner/sampling.py). Background: [Llama 3](https://arxiv.org/abs/2407.21783) and [LoRA, ICLR 2022](https://openreview.net/forum?id=nZeVKeeFYf9).
