# References and implementation scope

| Work | Used for | Primary source |
|---|---|---|
| The Llama 3 Herd of Models (2024) | Planner backbone | [Paper](https://arxiv.org/abs/2407.21783) |
| LoRA (ICLR 2022) | Low-rank task adaptation | [Paper](https://openreview.net/forum?id=nZeVKeeFYf9) |
| Large Language Diffusion Models (2025) | LLaDA and masked conditional modeling | [Paper](https://arxiv.org/abs/2502.09992), [code](https://github.com/ML-GSAI/LLaDA) |
| Simple and Effective Masked Diffusion Language Models (2024) | Masked-denoising formulation | [Paper](https://arxiv.org/abs/2406.07524) |
| Breaking the Factorization Barrier in Diffusion Language Models / CoDD (2026) | Tractable structured output distributions | [Paper](https://arxiv.org/abs/2603.00045) |
| LLM Meets Diffusion / CrysLLMGen (NeurIPS 2025) | Continuous crystal refinement | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html), [code](https://github.com/kdmsit/crysllmgen) |
| Crystal Structure Prediction by Joint Equivariant Diffusion / DiffCSP (NeurIPS 2023) | Periodic network and diffusion components used by CrysLLMGen | [Paper](https://arxiv.org/abs/2309.04475), [code](https://github.com/jiaor17/DiffCSP) |
| Don't Settle Too Early / RemeDi (ICLR 2026) | Remasking with richer context | [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d8bc24ef15b7a808b6dc403c01583e2-Abstract-Conference.html) |
| Iterative Distillation for Reward-Guided Fine-Tuning / VIDD (ICLR 2026) | Reward-weighted conditional supervision | [Paper](https://openreview.net/pdf/698f04aced8dfc906818be819cb717e0cae2b049.pdf) |
| CHGNet (Nature Machine Intelligence 2023) | Learned physical energy and relaxation | [Paper](https://doi.org/10.1038/s42256-023-00716-3), [code](https://github.com/CederGroupHub/chgnet) |
| Materials Project | Competing-phase energy references | [API documentation](https://docs.materialsproject.org/downloading-data/using-the-api) |
| SMACT | Composition screening and mixed-valence enumeration | [Code](https://github.com/WMD-group/SMACT) |

C1 uses a crystal-specific periodic tree head and the existing conditional reveal process. C2 uses a geometric editor, a finite-candidate physical-teacher update and one risk-tilted remasking step. These are adaptations of the cited ideas; full CoDD circuit architectures, RemeDi dual-stream trajectory training and VIDD iterative rollout training are separate algorithms.
