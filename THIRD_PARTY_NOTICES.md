# Third-party notices

Original DLM-ICLR project code is distributed under the repository's MIT license. External datasets and model weights keep their own terms.

The vendored refinement components in `src/dlm_iclr/_vendor/crysllmgen` are derived from [CrysLLMGen](https://github.com/kdmsit/crysllmgen), copyright 2025 Kishalay Das (KD), under the MIT license. The retained license is included beside the code and in [licenses/CrysLLMGen.txt](licenses/CrysLLMGen.txt). The numerical model, validity metrics and Direct composition scaler are included with namespaced imports. `src/dlm_iclr/diffusion/trainer.py` adapts the upstream training loop; `src/dlm_iclr/evaluation/direct.py` and `features.py` adapt generation metrics and coverage formulas from commit `94bb287751cd20a882c7c1df7ca736633d78e5e1`. Please cite *LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation*, NeurIPS 2025.

The CSP diffusion/network and geometric utilities build on [DiffCSP](https://github.com/jiaor17/DiffCSP), copyright 2023 Rui Jiao, under the MIT license. Its notice is retained in [licenses/DiffCSP.txt](licenses/DiffCSP.txt).

The language backbone and masked-diffusion sampling formulation build on [LLaDA](https://github.com/ML-GSAI/LLaDA). The upstream README identifies LLaDA-8B-Base and LLaDA-8B-Instruct as MIT-licensed. These foundation-model weights are loaded as external assets, not copied into this repository. Please cite *Large Language Diffusion Models* (Nie et al., 2025), [arXiv:2502.09992](https://arxiv.org/abs/2502.09992), when using that model family.

Physical evaluation uses [CHGNet](https://github.com/CederGroupHub/chgnet), [ASE](https://gitlab.com/ase/ase), and [pymatgen](https://github.com/materialsproject/pymatgen). Composition validity uses [SMACT](https://github.com/WMD-group/SMACT). These packages are dependencies and their source code is not vendored here. Materials Project reference entries and MP-20 structures remain external data assets; cite their original sources when reporting derived results.

Full Direct evaluation additionally uses [matminer](https://github.com/hackingmaterials/matminer) for Magpie/CrystalNN fingerprints and [SciPy](https://github.com/scipy/scipy) for Wasserstein and fingerprint distances. Their package source is not vendored here. Matminer incorporates feature data with its own attribution; cite the package and underlying descriptors when reporting those metrics.

The Planner uses external Meta-Llama-3-8B weights under the provider's Llama 3 license. The training and preprocessing code adapt the project's H1A2 implementation. The periodic C1 head and C2 teacher/revision components build on ideas discussed in CoDD, RemeDi and VIDD; their implementation scope and primary references are listed in [docs/reproduction.md](docs/reproduction.md).
