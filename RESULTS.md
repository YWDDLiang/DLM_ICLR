# Results

Results reported in **CrystalDLM: Feedback Learning of Periodic Diffusion Language Models for Crystal Generation**, manuscript v34 (Tables 1–4). Rates are percentages. Density and # Element are Wasserstein-1 distances for mass density and the number of distinct elements.

## De novo crystal generation

CrystalDLM final outputs before physical relaxation; 10,000 parseable outputs per dataset (Table 1).

| Dataset | Structural ↑ | Compositional ↑ | Precision ↑ | Recall ↑ | Density ↓ | # Element ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| MP-20 | 100.00 | 92.31 | 98.95 | 99.44 | 0.79 | 0.07 |
| Perov-5 | 100.00 | 99.83 | 98.60 | 97.08 | 0.21 | 0.06 |
| MPTS-52 | 99.65 | 87.22 | 95.85 | 98.81 | 0.86 | 0.45 |

## Stable and novel crystal generation

MP-20, 5,000 outputs (Table 2).

| Model | V.U.N. ↑ | S.U.N. ↑ | M.S.U.N. ↑ |
| --- | --- | --- | --- |
| CrystalDLM | 81.68 | 10.50 | 51.24 |

## Crystal construction before diffusion

CrystalDLM drafts before diffusion and feedback reconstruction; 10,000 parseable outputs per dataset (Table 3). Sampling success is the fraction of generation attempts yielding a parseable structure.

| Dataset | Sampling success ↑ | Structural ↑ | Compositional ↑ | Precision ↑ | Recall ↑ | Density ↓ | # Element ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MP-20 | 99.46 | 99.83 | 92.31 | 98.53 | 90.60 | 1.21 | 0.07 |
| Perov-5 | 99.99 | 100.00 | 99.83 | 96.43 | 92.60 | 0.14 | 0.06 |
| MPTS-52 | 98.22 | 97.13 | 87.22 | 99.00 | 94.41 | 0.76 | 0.45 |

## Ablation studies

MP-20 variants sharing the planner, condition list, constructor checkpoint and diffusion model (Table 4). Structural validity uses 10,000 outputs; S.U.N. and M.S.U.N. use 5,000 outputs.

| Model | Structural validity ↑ | S.U.N. ↑ | M.S.U.N. ↑ |
| --- | --- | --- | --- |
| DLM | 98.40 | 7.86 | 45.08 |
| Periodic DLM | 100.00 | 8.48 | 47.24 |
| CrystalDLM | 100.00 | 10.50 | 51.24 |

DLM uses the base constructor and diffusion. Periodic DLM adds periodic coordinate interactions. CrystalDLM adds feedback reconstruction and relative verification; the latter two variants share diffusion references.

## Metrics

- **Structural / Compositional:** geometric validity and composition plausibility.
- **Precision / Recall:** proximity to, and coverage of, the reference distribution.
- **V.U.N.:** valid, unique and novel.
- **S.U.N. / M.S.U.N.:** unique and novel with CHGNet-screened energy above hull ≤ 0 / ≤ 0.1 eV/atom. Metastability includes stability.
- **U / N:** structural uniqueness within the saved output order / novelty relative to the training-reference set.
